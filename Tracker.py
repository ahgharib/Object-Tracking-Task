import cv2 as cv
import numpy as np
import torch
from PIL import Image
import open_clip

# track stuff
from ultralytics import YOLO
from supertracker import ByteTrack, Detections

# ---- Configs ----
M_WEIGHTS = "yolo26m.pt"
MIN_CONF = 0.3
REID_THRESH = 0.65
C_THRESH = 0.32

# global state dict (ugh, I know, but it works for now)
target_info = {
    'id': None,
    'cls': None,
    'hist_data': None,
    'clip_feat': None
}
user_click = None

def on_mouse_click(evt, x, y, flags, prm):
    global user_click
    if evt == cv.EVENT_LBUTTONDOWN:
        user_click = (x, y)

def grab_hist(img_crop, b=(16,16)):
    hsv_img = cv.cvtColor(img_crop, cv.COLOR_BGR2HSV)
    # calc histogram
    h = cv.calcHist([hsv_img], [0,1], None, [b[0], b[1]], [0, 180, 0, 256])
    cv.normalize(h, h)
    return h.flatten()

def check_hist_sim(h_a, h_b):
    return cv.compareHist(h_a.astype("float32"), h_b.astype("float32"), cv.HISTCMP_CORREL)

def extract_clip(crop_img):
    try:
        rgb_img = cv.cvtColor(crop_img, cv.COLOR_BGR2RGB)
        p_img = Image.fromarray(rgb_img)
        t = clip_prep(p_img).unsqueeze(0).cuda()
        with torch.no_grad():
            feat = clip_mod.encode_image(t)
            feat = feat / feat.norm(dim=-1, keepdim=True)
        return feat.cpu().numpy()[0]
    except Exception as e:
        # print(f"clip error: {e}")
        return None

def calc_clip_score(v1, v2):
    if v1 is None or v2 is None: return -1
    return np.dot(v1 / np.linalg.norm(v1), v2 / np.linalg.norm(v2))


# init models
print("-> initializing yolo...")
model_y = YOLO(M_WEIGHTS)

print("-> spinning up bytetrack...")
trk = ByteTrack(track_activation_threshold=0.25, lost_track_buffer=30, frame_rate=30)

print("-> loading CLIP (this might take a sec)")
clip_mod, _, clip_prep = open_clip.create_model_and_transforms('ViT-B-32', pretrained='laion2b_s34b_b79k')
clip_mod = clip_mod.cuda().eval()

cam = cv.VideoCapture(0)
cv.namedWindow("Tracking window")
cv.setMouseCallback("Tracking window", on_mouse_click)

print("Ready! Click on something to lock on. Press ESC to bail.")

while True:
    ok, frm = cam.read()
    if not ok: break

    res = model_y(frm)[0]
    
    # map to bytetrack format
    dets = Detections(
        xyxy=res.boxes.xyxy.cpu().numpy(),
        confidence=res.boxes.conf.cpu().numpy(),
        class_id=res.boxes.cls.cpu().numpy().astype(int)
    )

    trkd = trk.update_with_detections(dets)

    # user selecting a new target
    if user_click and target_info['id'] is None:
        cx, cy = user_click
        for idx, b in enumerate(trkd.xyxy):
            xx1, yy1, xx2, yy2 = b.astype(int)
            
            if xx1 <= cx <= xx2 and yy1 <= cy <= yy2:
                target_info['id'] = int(trkd.tracker_id[idx])
                target_info['cls'] = int(trkd.class_id[idx])
                target_info['hist_data'] = grab_hist(frm[yy1:yy2, xx1:xx2])
                target_info['clip_feat'] = extract_clip(frm[yy1:yy2, xx1:xx2])
                
                print(f"Locked onto ID {target_info['id']} (Class {target_info['cls']})")
                break
        
        user_click = None # reset click

    active_ids = [int(t) for t in trkd.tracker_id]
    
    # recovery logic if we lose the target
    if target_info['id'] is not None and target_info['id'] not in active_ids:
        top_score = -1
        top_idx = -1
        
        for idx, b in enumerate(trkd.xyxy):
            c_id = int(trkd.class_id[idx])
            conf_val = trkd.confidence[idx]

            if c_id != target_info['cls'] or conf_val < MIN_CONF: continue

            xx1, yy1, xx2, yy2 = b.astype(int)
            c_crop = frm[yy1:yy2, xx1:xx2]

            c_emb = extract_clip(c_crop)
            sim_clip = calc_clip_score(target_info['clip_feat'], c_emb)
            sim_hist = check_hist_sim(target_info['hist_data'], grab_hist(c_crop))

            # prefer clip, fallback to hist similarity
            final_score = sim_clip if sim_clip >= 0 else (sim_hist * 0.8)

            if final_score > top_score:
                top_score = final_score
                top_idx = idx

        # re-assign if it passes the threshold
        if top_score >= C_THRESH or top_score >= REID_THRESH:
            target_info['id'] = int(trkd.tracker_id[top_idx])
            xx1, yy1, xx2, yy2 = trkd.xyxy[top_idx].astype(int)
            target_info['hist_data'] = grab_hist(frm[yy1:yy2, xx1:xx2])
            target_info['clip_feat'] = extract_clip(frm[yy1:yy2, xx1:xx2])
            print(f"Target re-acquired! New ID: {target_info['id']} (Score: {top_score:.2f})")

    # rendering
    out_frame = frm.copy()
    for idx, b in enumerate(trkd.xyxy):
        t_id = int(trkd.tracker_id[idx])
        
        # only draw if it's our target (or if nothing is selected yet)
        if target_info['id'] is None or t_id == target_info['id']:
            xx1, yy1, xx2, yy2 = b.astype(int)
            lbl = f"#{t_id} {model_y.names[int(trkd.class_id[idx])]} {trkd.confidence[idx]:.2f}"
            
            cv.rectangle(out_frame, (xx1, yy1), (xx2, yy2), (0, 255, 120), 2)
            cv.putText(out_frame, lbl, (xx1, yy1-8), cv.FONT_HERSHEY_DUPLEX, 0.5, (0, 255, 120), 1)

    cv.imshow("Tracking window", out_frame)
    
    if (cv.waitKey(1) & 0xFF) == 27: # user hit esc
        break

cam.release()
cv.destroyAllWindows()
