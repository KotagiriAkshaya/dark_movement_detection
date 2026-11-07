import cv2
import numpy as np

clahe = cv2.createCLAHE(clipLimit=3.0, tileGridSize=(8, 8))
GAMMA_TABLE = np.array([(i / 255.0) ** 0.5 * 255 for i in range(256)]).astype("uint8")

def enhance(frame, denoise=False):
    brightened = cv2.LUT(frame, GAMMA_TABLE)
    lab = cv2.cvtColor(brightened, cv2.COLOR_BGR2LAB)
    l, a, b = cv2.split(lab)
    l = clahe.apply(l)
    result = cv2.cvtColor(cv2.merge((l, a, b)), cv2.COLOR_LAB2BGR)
    if denoise:  # slow, so off by default
        result = cv2.fastNlMeansDenoisingColored(result, None, 5, 5, 7, 21)
    return cv2.GaussianBlur(result, (3, 3), 0)