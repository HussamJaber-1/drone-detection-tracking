/content/pipeline.py

# Drone Detection and Tracking Pipeline
# YOLOv8 + ByteTrack + Weighted Sensor Fusion
# Hussam Jaber — F.A.S.T. Manufacturing Skills Demo

from ultralytics import YOLO
import cv2

# ── 1. Load trained model ──────────────────────────────────────────
model = YOLO('best.pt')  # YOLOv8n fine-tuned on 15,687 drone images

# ── 2. Open input video ────────────────────────────────────────────
cap = cv2.VideoCapture('drone_clip.mp4')
frame_width = int(cap.get(cv2.CAP_PROP_FRAME_WIDTH))
frame_height = int(cap.get(cv2.CAP_PROP_FRAME_HEIGHT))
fps = int(cap.get(cv2.CAP_PROP_FPS))
frame_area = frame_width * frame_height

# ── 3. Set up output video ─────────────────────────────────────────
out = cv2.VideoWriter(
    'drone_fusion_output.avi',
    cv2.VideoWriter_fourcc(*'XVID'),
    fps,
    (frame_width, frame_height)
)

# ── 4. Process frame by frame ──────────────────────────────────────
while True:
    ret, frame = cap.read()
    if not ret:
        break

    # Run YOLOv8 detection + ByteTrack tracking on this frame
    results = model.track(
        frame,
        persist=True,       # ByteTrack remembers IDs across frames
        conf=0.5,           # Only report detections above 50% confidence
        tracker='bytetrack.yaml',
        verbose=False
    )

    if results[0].boxes is not None and results[0].boxes.id is not None:
        for box in results[0].boxes:
            # Bounding box coordinates
            x1, y1, x2, y2 = map(int, box.xyxy[0])

            # Signal 1: YOLO detection confidence (0.0 - 1.0)
            yolo_conf = float(box.conf[0])

            # Signal 2: Size signal — larger box means drone is closer
            box_area = (x2 - x1) * (y2 - y1)
            size_signal = min(box_area / frame_area * 10, 1.0)

            # Weighted fusion — mirrors C-UAS multi-sensor architecture
            # Camera 70% + Proximity 30% = fused threat confidence
            fused_score = (0.70 * yolo_conf) + (0.30 * size_signal)

            # Green = high confidence (>=0.7), Orange = uncertain (<0.7)
            # Mirrors C-UAS detect threshold (0.70) vs engage threshold (0.95)
            color = (0, 255, 0) if fused_score >= 0.7 else (0, 165, 255)

            # Draw box and label
            track_id = int(box.id[0])
            cv2.rectangle(frame, (x1, y1), (x2, y2), color, 2)
            label = f"id:{track_id} fused:{fused_score:.2f}"
            cv2.putText(frame, label, (x1, y1 - 10),
                       cv2.FONT_HERSHEY_SIMPLEX, 0.6, color, 2)

    out.write(frame)

cap.release()
out.release()
print("Done — output saved to drone_fusion_output.avi")
