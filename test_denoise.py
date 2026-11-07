import cv2
import numpy as np
from enhance import enhance
from simulate import make_dark

hog = cv2.HOGDescriptor()
hog.setSVMDetector(cv2.HOGDescriptor_getDefaultPeopleDetector())

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
mild = [make_dark(f, noise_level=6) for f in frames]


def temporal(seq, i, k=2):
    lo, hi = max(0, i - k), min(len(seq), i + k + 1)
    return np.mean(seq[lo:hi], axis=0).astype(np.uint8)


variants = {
    "A current (brighten only)": lambda i: enhance(dark[i]),
    "B blur 5x5 first": lambda i: enhance(cv2.GaussianBlur(dark[i], (5, 5), 0)),
    "C blur 7x7 first": lambda i: enhance(cv2.GaussianBlur(dark[i], (7, 7), 0)),
    "D average 5 frames": lambda i: enhance(temporal(dark, i)),
    "E milder noise (control)": lambda i: enhance(mild[i]),
}

for name, fn in variants.items():
    found = checked = 0
    for i in range(0, len(frames), 3):
        img = fn(i)
        rects, _ = hog.detectMultiScale(img, winStride=(8, 8), padding=(8, 8), scale=1.05)
        checked += 1
        if len(rects):
            found += 1
    print(f"{name}: people found in {found} of {checked} frames", flush=True)