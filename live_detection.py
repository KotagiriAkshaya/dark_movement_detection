
import cv2
import numpy as np
from ultralytics import YOLO
import threading
import math

# Load YOLO model
MODEL_PATH = "yolov8n.pt"
CONFIDENCE = 0.15

model = YOLO(MODEL_PATH)

# Previous human positions for movement detection
previous_centers = []
tracking_lock = threading.Lock()


def reset_tracking():
    """Reset previous human positions."""
    global previous_centers

    with tracking_lock:
        previous_centers = []


def detect_frame(image_data):

    global previous_centers

    # Decode browser image
    image_bytes = np.frombuffer(image_data, np.uint8)
    frame = cv2.imdecode(image_bytes, cv2.IMREAD_COLOR)

    if frame is None:
        raise ValueError("Invalid camera frame")

    height, width = frame.shape[:2]

    # Resize for faster processing
    target_width = 640
    scale = min(1.0, target_width / width)

    if scale < 1.0:
        frame = cv2.resize(
            frame,
            (int(width * scale), int(height * scale))
        )

    # YOLO detects humans (class 0)
    results = model.predict(
        frame,
        classes=[0],
        conf=CONFIDENCE,
        imgsz=640,
        verbose=False,
        device="cpu"
    )

    current_centers = []
    boxes = []

    for result in results:

        for box in result.boxes:

            x1, y1, x2, y2 = map(
                int, box.xyxy[0].tolist()
            )

            confidence = float(box.conf[0])

            center_x = (x1 + x2) // 2
            center_y = (y1 + y2) // 2

            current_centers.append(
                (center_x, center_y)
            )

            boxes.append({
                "x1": x1,
                "y1": y1,
                "x2": x2,
                "y2": y2,
                "confidence": round(confidence, 2)
            })

    # Compare current and previous human positions
    movement_detected = False

    with tracking_lock:

        if previous_centers and current_centers:

            for cx, cy in current_centers:

                distances = [
                    math.dist((cx, cy), old_center)
                    for old_center in previous_centers
                ]

                nearest_distance = min(distances)

                # Ignore small bounding-box fluctuations
                if 15 < nearest_distance < 120:
                    movement_detected = True

                # A person appearing at a new location
                elif nearest_distance >= 120:
                    movement_detected = True

        elif current_centers and not previous_centers:
            # First detection: person present,
            # but movement is not confirmed yet.
            movement_detected = False

        previous_centers = current_centers

    return {
        "boxes": boxes,
        "person_detected": len(boxes) > 0,
        "movement": movement_detected,
        "count": len(boxes),
        "width": frame.shape[1],
        "height": frame.shape[0]
    }
