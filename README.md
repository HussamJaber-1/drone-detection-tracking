# Drone Detection and Tracking Pipeline

YOLOv8 fine-tuned drone detector with ByteTrack multi-object tracking and weighted sensor fusion. Built as a practical skills demonstration for C-UAS (Counter-Unmanned Aircraft Systems) engineering.

## What it does

- Detects drones in video using YOLOv8n fine-tuned on 15,687 images
- Tracks each drone across frames with ByteTrack, maintaining persistent IDs
- Computes a fused threat confidence score combining YOLO confidence (70%) and proximity signal (30%)
- Colour-codes detections: green = high confidence (≥0.70), orange = uncertain

## Architecture mapping to C-UAS

| This project | C-UAS system |
|---|---|
| YOLOv8 detector | EO/IR camera detection |
| ByteTrack persistent IDs | Multi-object track management |
| Weighted fusion (0.70/0.30) | Sensor fusion (camera 0.35, radar 0.30, RF 0.25, acoustic 0.10) |
| Green/orange threshold at 0.70 | Detect threshold 0.70 / Engage threshold 0.95 |
| ID switching on occlusion | Solved in real C-UAS by radar maintaining track during camera loss |

## Training

- Dataset: Drone Computer Vision Model (project-986i8), 7,248 images, CC BY 4.0
- Augmented to 15,687 training images via Roboflow
- Model: YOLOv8n pretrained on COCO, fine-tuned for 40 epochs
- Final validation metrics: mAP50 0.978, Precision 0.956, Recall 0.955

## Results

- Consistent detection at 0.70-0.85 confidence on real-world footage
- ByteTrack maintains persistent IDs across frames
- ID switching observed on scene transitions — mitigated in production by radar track continuity
- Detection degrades on ground-level backgrounds (domain shift from aerial training data)

## Files

- `pipeline.py` — full inference pipeline with fusion layer
- `best.pt` — trained weights (download from Releases)

## Setup

```bash
pip install ultralytics supervision
python pipeline.py
```

## Key concepts demonstrated

- Transfer learning: YOLOv8n COCO weights fine-tuned on drone-specific data
- ByteTrack: IoU-based matching with Kalman filter position prediction during occlusion
- Sensor fusion: weighted confidence blending mirroring multi-modal C-UAS architecture
- Domain shift: model trained on aerial backgrounds fails on ground-level — real-world generalisation challenge
