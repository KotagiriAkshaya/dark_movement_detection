import cv2
from enhance import enhance
from simulate import make_dark

hog = cv2.HOGDescriptor()
hog.setSVMDetector(cv2.HOGDescriptor_getDefaultPeopleDetector())


def count(width, dark, every=3):
    cap = cv2.VideoCapture("sample.mp4")
    n = found = checked = 0
    while True:
        ok, frame = cap.read()
        if not ok:
            break
        n += 1
        if n % every:
            continue
        h, w = frame.shape[:2]
        frame = cv2.resize(frame, (width, int(h * width / w)))
        if dark:
            frame = enhance(make_dark(frame))
        rects, _ = hog.detectMultiScale(frame, winStride=(8, 8), padding=(8, 8), scale=1.05)
        checked += 1
        if len(rects):
            found += 1
    return found, checked


for width in (320, 480, 640):
    for dark in (False, True):
        f, c = count(width, dark)
        print(f"width={width} dark={dark}: people found in {f} of {c} frames checked", flush=True)