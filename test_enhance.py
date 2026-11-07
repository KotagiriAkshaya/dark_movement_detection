import cv2
import numpy as np
from enhance import enhance

def boost(frame):
    """Stronger brightening for videos that are already dark."""
    gray = cv2.cvtColor(frame, cv2.COLOR_BGR2GRAY)
    p = np.percentile(gray, 99.5)
    gain = min(200.0 / max(p, 5), 40)
    out = np.clip(frame.astype(np.float32) * gain, 0, 255).astype(np.uint8)
    return enhance(out), gain

cap = cv2.VideoCapture("sample.mp4")
total = int(cap.get(cv2.CAP_PROP_FRAME_COUNT))
print("Total frames:", total)

for i, pos in enumerate([0.1, 0.3, 0.5, 0.7, 0.9], start=1):
    cap.set(cv2.CAP_PROP_POS_FRAMES, int(total * pos))
    ok, frame = cap.read()
    if not ok:
        continue
    boosted, gain = boost(frame)
    small_a = cv2.resize(frame, (640, 360))
    small_b = cv2.resize(boosted, (640, 360))
    cv2.imwrite(f"compare_{i}.jpg", np.hstack([small_a, small_b]))
    print(f"frame at {int(pos*100)}%: average brightness {frame.mean():.1f}, gain used {gain:.1f}")

print("Done. Open compare_1.jpg to compare_5.jpg (left = original, right = brightened)")