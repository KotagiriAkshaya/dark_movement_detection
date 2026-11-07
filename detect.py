import os
import shutil
import subprocess
import cv2
import numpy as np
from ultralytics import YOLO
from enhance import enhance
from simulate import make_dark

model = YOLO("yolov8n.pt")   # small model, loaded once


def to_h264(src, dst):
    """Re-encode so browsers can play the video (needs ffmpeg)."""
    if shutil.which("ffmpeg"):
        subprocess.run(
            ["ffmpeg", "-y", "-i", src, "-vcodec", "libx264", "-pix_fmt", "yuv420p", dst],
            check=True, capture_output=True,
        )
        os.remove(src)
    else:
        os.replace(src, dst)


def process_video(in_path, out_path, min_area=1500, simulate_dark=False,
                  target_width=640, max_seconds=60, brightness=0.25, conf=0.15):
    cap = cv2.VideoCapture(in_path)
    fps = cap.get(cv2.CAP_PROP_FPS) or 25
    src_w = int(cap.get(cv2.CAP_PROP_FRAME_WIDTH))
    src_h = int(cap.get(cv2.CAP_PROP_FRAME_HEIGHT))
    scale = target_width / src_w if src_w > target_width else 1.0
    w, h = int(src_w * scale), int(src_h * scale)

    raw_path = out_path.replace(".mp4", "_raw.mp4")
    out = cv2.VideoWriter(raw_path, cv2.VideoWriter_fourcc(*"mp4v"), fps, (w, h))

    bg = cv2.createBackgroundSubtractorMOG2(history=300, varThreshold=25, detectShadows=False)
    kernel = np.ones((3, 3), np.uint8)

    seconds_with_humans = set()
    frame_no = 0
    human_frames = 0

    while True:
        ok, frame = cap.read()
        if not ok or frame_no / fps > max_seconds:
            break
        frame_no += 1

        if scale < 1:
            frame = cv2.resize(frame, (w, h))
        if simulate_dark:
            frame = make_dark(frame, brightness)

        view = enhance(frame)                      # brightened picture shown to the user
        gray = cv2.cvtColor(view, cv2.COLOR_BGR2GRAY)

        mask = bg.apply(gray)
        mask = cv2.morphologyEx(mask, cv2.MORPH_OPEN, kernel)
        mask = cv2.dilate(mask, None, iterations=2)
        contours, _ = cv2.findContours(mask, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)

        moving = any(cv2.contourArea(c) > min_area for c in contours)
        human_found = False

        if moving:
            result = model(view, classes=[0], conf=conf, verbose=False)   # class 0 = person
            for box in result[0].boxes:
                x1, y1, x2, y2 = map(int, box.xyxy[0])
                cv2.rectangle(view, (x1, y1), (x2, y2), (0, 255, 0), 2)
                human_found = True

        if human_found:
            human_frames += 1
            seconds_with_humans.add(int(frame_no / fps))
            cv2.putText(view, "HUMAN MOVEMENT", (10, 30),
                        cv2.FONT_HERSHEY_SIMPLEX, 0.8, (0, 0, 255), 2)

        out.write(view)

    cap.release()
    out.release()
    to_h264(raw_path, out_path)

    return {
        "events": sorted(seconds_with_humans),
        "frames": frame_no,
        "human_frames": human_frames,
        "duration": round(frame_no / fps, 1),
    }