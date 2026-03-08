# Object-Tracking-Task
Intern acceptance Task for Eyego company
-----------------------------------------------------------------------------------------------------------------------------------------------------------

## Real-Time Object Tracker with YOLO, ByteTrack & CLIP

This repository implements a **real-time single-object tracker** combining **YOLOv26 object detection**, **ByteTrack multi-object tracking**, and **OpenCLIP-based re-identification**. It allows the user to **select an object via mouse click** and track it robustly, even under occlusion, lighting changes, or partial loss of the target.

---

###  Features

- **Real-time object detection** using YOLOv26.
- **Persistent multi-object tracking** with ByteTrack.
- **User-driven target selection** via mouse click.
- **Intelligent target re-acquisition** using:
  - Color histogram similarity
  - CLIP embedding cosine similarity
- Handles **appearance changes** and **temporary occlusion**.
- Live visualization with OpenCV.

---

### 🛠️ Tech Stack

| Component | Purpose |
|-----------|--------|
| `ultralytics.YOLO` | YOLOv26 object detection for bounding boxes, classes, and confidence |
| `supertracker.ByteTrack` | Maintains persistent object IDs using motion tracking |
| `OpenCLIP ViT-B-32` | Generates deep embeddings for semantic object re-identification |
| `OpenCV` | Video capture, rendering, mouse interaction |
| `NumPy` | Array manipulation, numerical computations |
| `PIL` | Conversion of OpenCV images to formats compatible with CLIP |

---

### ⚙️ Installation


git clone <repo-url>
cd <repo-folder>
pip install -r requirements.txt
pip install torch==2.12.0.dev20260307+cu128 torchvision==0.26.0.dev20260221+cu128 torchaudio==2.11.0.dev20260227+cu128 --pre --extra-index-url https://download.pytorch.org/whl/nightly/cu128

### 🖥️ Usage
python tracker.py

Click on any object in the "Tracking window" to lock on.

Press ESC to exit.

### 🔹 Configuration Variables
| Variable      | Type         | Description                                                   |
| ------------- | ------------ | ------------------------------------------------------------- |
| `M_WEIGHTS`   | str          | YOLOv26 weights path                                           |
| `MIN_CONF`    | float        | Minimum confidence threshold for detections                   |
| `REID_THRESH` | float        | Cosine similarity threshold for re-identifying lost target    |
| `C_THRESH`    | float        | Combined histogram + CLIP score threshold for recovery        |
| `target_info` | dict         | Stores target metadata: `id`, `cls`, `hist_data`, `clip_feat` |
| `user_click`  | tuple / None | Stores user mouse click coordinates                           |

### 🔹 Functions
#### on_mouse_click(evt, x, y, flags, prm)

  Detects left mouse clicks for target selection.

#### grab_hist(img_crop, b=(16,16)) → np.array

  Computes normalized HSV histogram for a cropped image.

#### check_hist_sim(h_a, h_b) → float

  Returns histogram correlation between two objects.

#### extract_clip(crop_img) → np.array

  Generates CLIP embeddings (ViT-B-32) for object re-identification.

#### calc_clip_score(v1, v2) → float

  Computes cosine similarity between two CLIP embeddings.

### 🔹 Workflow Diagram
                 ┌───────────────┐
                 │ Video Capture │
                 └───────┬───────┘
                         │
                         ▼
                  ┌────────────┐
                  │ YOLOv26 Det│
                  │ (BBoxes,   │
                  │ Classes,   │
                  │ Confidence)│
                  └───────┬────┘
                          │
                          ▼
                  ┌────────────┐
                  │ ByteTrack  │
                  │ Multi-Obj  │
                  │ Tracking   │
                  └───────┬────┘
                          │
        ┌─────────────────┴─────────────────┐
        │                                   │
        ▼                                   ▼
    ┌───────────────┐                   ┌──────────────┐
    │ User Click    │                   │ Target Lost? │
    │ Selection     │                   └───────┬──────┘
    └───────┬───────┘                           │
            │                                   ▼
            │                        ┌────────────────────┐
            │                        │ Re-identification   │
            │                        │ CLIP Embedding +    │
            │                        │ Histogram Similarity│
            │                        └─────────┬──────────┘
            ▼                                  │
    ┌───────────────┐                          │
    │ Store Target  │                          │
    │ Info (Hist +  │<─────────────────────────┘
    │ CLIP)         │
    └───────────────┘
            │
            ▼
    ┌───────────────┐
    │ Render BBoxes │
    │ on Frame      │
    └───────────────┘

### 💡 Design Insights

- Hybrid Re-ID: Combines traditional color histograms with deep embeddings for robust re-identification.

- Click-based selection: Ensures user chooses the correct object, useful in multi-object scenarios.

- Fallback mechanisms: Histogram similarity used when CLIP fails (e.g., partial crops, low-quality frames).

- Score weighting: Prefers semantic CLIP similarity; histogram similarity acts as secondary check.


### 🔹 Notes / Limitations

Requires a CUDA-capable GPU for real-time CLIP inference.

Sensitive to extremely small or heavily occluded objects.

YOLOv26 detection quality directly impacts tracking reliability.

Depends on YOLOV26 Classes. Cannot detect objects outside these Classes

------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------

# 🧪 Experiments

I evaluated several approaches for **real-time single-object tracking**, ranging from classical OpenCV algorithms to modern deep learning methods, including segmentation-guided tracking and YOLO-based detection with Re-ID.

---

### 1️⃣ OpenCV Trackers (KCF, CSRT, etc.)

**Technical Details:**

- Used OpenCV's **legacy tracking algorithms**, e.g., `TrackerCSRT` and `TrackerKCF`.
- Workflow:
  1. Capture first frame from webcam.
  2. User selects ROI via `cv2.selectROI`.
  3. Initialize tracker on selected bounding box.
  4. Update tracker for each frame, rendering results.
- Relies purely on **pixel-level appearance** (color, texture) and motion estimation.
- Pros: lightweight, fast, requires no deep learning or GPU.
- Limitations:
  - Sensitive to **occlusion**, **fast motion**, and **appearance changes**.
  - No Real mechanism for re-identifying lost objects.
- Key functions:
  - `cv2.TrackerCSRT.create()`
  - `tracker.init(frame, bbox)`
  - `tracker.update(frame)`

**Results:**
- The object is tracked but in an inaccurate way and the accuracy is very low, cannot handle multiple objects in one frame

**Why it works or not:**  
Classical trackers work well in simple scenarios but fail under occlusion, lighting change, or when the object leaves/re-enters the frame.

---

### 2️⃣ Deep Feature Tracker (Siamese/ResNet-based)

**Technical Details:**

- Combines **Siamese-style tracking** with feature embedding extraction (ResNet18).
- Workflow:
  1. Extract deep embeddings from the object patch.
  2. Maintain multiple templates for appearance variation.
  3. Re-rank candidate regions (Selective Search or sliding window) using embedding similarity.
  4. Re-initialize tracker when the object is lost.
- Fallback using **ORB keypoints** if GPU/torch is unavailable.
- Features:
  - Embeddings: 512-d L2-normalized vectors from ResNet18.
  - Proposal generation: selective search or sliding windows.
  - Tracker smoothing: exponential smoothing on bounding box coordinates.
- Key functions/classes:
  - `FeatureExtractor.embed()`
  - `SingleObjectTracker.re_detect()`
  - `SingleObjectTracker.rerank_proposals()`

**Results:**
- Better than traditional algorithms, but still cannot handle similar objects, size change, or fast motion
- Noticiably slower than Experiment 1


**Why it works or not:**  
Works better than classical trackers for appearance changes due to deep embeddings. Still limited in **real-time performance** and sometimes fails for small or fast-moving objects.

---

### 3️⃣ SAM2.1_tiny + Tracker (Segmentation-Guided Tracking)

**Technical Details:**

- Integrates **MobileSAM** (tiny variant of Segment Anything Model) with KCF tracker.
- Workflow:
  1. Capture webcam frames and allow user ROI selection.
  2. Track object via KCF tracker.
  3. Periodically re-run SAM on first + current frame to **propagate segmentation masks**.
  4. Re-initialize tracker using bounding box derived from predicted mask.
- Pros:
  - Can **track by segmentation**, not just bounding box.
  - Handles **partial occlusion** better than classical trackers.
  - Can track **Any Object**, with no class restrictions
  - works well on recorded videos
- Limitations:
  - Requires periodic mask propagation → introduces **latency**.
  - Heavily reliant on the initial frame and SAM quality.
  - Too slow for real-time
  - Even tiny SAM is slow and not accurate enough
  - Cannot handle multiple objects of the same type in a group
- Observations:
  - When two objects of the same type and size overlap, the system switches the boundary box and cannot differentiate the selected object from the overlapped one  
- Key functions:
  - `build_sam2_video_predictor()`
  - `predictor.propagate_in_video()`
  - `tracker.init(frame, prev_box)`

**Results:** 
  - Best Accuracy so far, but extremely slow and still vulnerable to object overlapping. 

**Why it works or not:**  
Segmentation-based tracking can handle complex shapes and partial occlusion, but performance is **limited by SAM inference speed** and may drift if mask propagation fails.

---

### 4️⃣ YOLOv26 + ByteTrack + CLIP (Current Code)

**Technical Details:**

- Modern pipeline combining:
  1. **YOLOv26**: real-time object detection for bounding boxes, classes, and confidence scores.
  2. **ByteTrack**: multi-object tracker maintaining persistent IDs.
  3. **CLIP embeddings**: semantic re-identification when the target is lost.
  4. **Color histograms**: lightweight fallback for re-acquisition.
- Workflow:
  - User clicks object → save histogram and CLIP embedding.
  - Track object using ByteTrack.
  - If the object is lost, compute similarity using CLIP and histograms to reassign ID.
- Key functions:
  - `extract_clip(crop_img)`
  - `calc_clip_score(v1, v2)`
  - `grab_hist(img_crop)`
  - `trk.update_with_detections(dets)`
- Pros:
  - Robust to **occlusion, lighting changes, and re-entry**.
  - Combines **semantic similarity** (CLIP) with **traditional CV cues** (histograms).
  - Faster than SAM2
- Limitations:
  - Requires **GPU** for real-time CLIP inference.
  - Heavier pipeline than classical trackers.
  - does not match the exact requirements
  - depends on Yolo Classes
  - Cannot track any object unless it is from the COCO dataset (or it mistakenly classifies the object with a COCO label)
  - Still vulnerable to object Overlapping

**Results:**
 - better accuracy than 1, 2
 - better Re-detection than most systems

**Why it works or not:**  
This approach is the most realistic solution for the requirments **real-world** tracking because it combines **detection, motion tracking, and deep re-identification**, solving most of the failure modes seen in classical and single-feature trackers.
