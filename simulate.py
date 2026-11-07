import numpy as np

def make_dark(frame, brightness=0.25, noise_level=12):
    dark = (frame * brightness).astype(np.float32)
    noise = np.random.normal(0, noise_level, frame.shape).astype(np.float32)
    return np.clip(dark + noise, 0, 255).astype(np.uint8)