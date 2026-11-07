import cv2

src = cv2.VideoCapture("sample_full.mp4")
fps = src.get(cv2.CAP_PROP_FPS)
w = int(src.get(cv2.CAP_PROP_FRAME_WIDTH))
h = int(src.get(cv2.CAP_PROP_FRAME_HEIGHT))
out = cv2.VideoWriter("sample.mp4", cv2.VideoWriter_fourcc(*"mp4v"), fps, (w, h))

start = int(3.0 * fps)    # start of the steady part
end = int(11.5 * fps)     # end of the steady part
src.set(cv2.CAP_PROP_POS_FRAMES, start)
for _ in range(end - start):
    ok, frame = src.read()
    if not ok:
        break
    out.write(frame)

src.release()
out.release()
print("Saved sample.mp4 (about 8.5 seconds)")