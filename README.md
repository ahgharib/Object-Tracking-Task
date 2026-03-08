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
pip install opencv-python torch ultralytics supertracker open-clip-torch pillow numpy

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
                  │ YOLOv26 Det │
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



# Experiments:
1- OpenCV Algorithms
2- Siem and Tracker
3- SAM2.1_tiny and Tracker
4- YOLOv26 and Tracker
