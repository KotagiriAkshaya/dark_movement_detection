import cv2
import numpy as np
from ultralytics import YOLO
from enhance import enhance
from simulate import make_dark

model = YOLO("yolov8n.pt")   # small model, downloads on first run (needs internet)

cap = cv2.VideoCapture("sample.mp4")
frames = []
while True:
    ok, f = cap.read()
    if not ok:
        break
    h, w = f.shape[:2]
    frames.append(cv2.resize(f, (640, int(h * 640 / w))))
print("frames loaded:", len(frames), flush=True)

np.random.seed(0)
dark = [make_dark(f) for f in frames]


def temporal(seq, i, k=2):
    lo, hi = max(0, i - k), min(len(seq), i + k + 1)
    return np.mean(seq[lo:hi], axis=0).astype(np.uint8)


variants = {
    "Normal bright video (ceiling)": lambda i: frames[i],
    "Dark, no enhancement": lambda i: dark[i],
    "Dark, brighten only": lambda i: enhance(dark[i]),
    "Dark, average 5 frames + brighten": lambda i: enhance(temporal(dark, i)),
}

for name, fn in variants.items():
    found = checked = 0
    for i in range(0, len(frames), 3):
        r = model(fn(i), classes=[0], conf=0.25, verbose=False)   # class 0 = person
        checked += 1
        if len(r[0].boxes):
            found += 1
    print(f"{name}: people found in {found} of {checked} frames", flush=True)