import cv2
import numpy as np

FEATURE_NAMES = ["r_n","g_n","b_n","h_n","s_n","v_n"]

def extract_color_features(frame_bgr, mask):
    pixels_bgr = frame_bgr[mask > 0]
    if len(pixels_bgr) == 0:
        raise ValueError("No foreground pixels for colour extraction")
    pixels_rgb = pixels_bgr[:, ::-1].reshape(-1, 1, 3)
    rgb_mean = pixels_rgb.reshape(-1, 3).mean(axis=0) / 255.0
    hsv = cv2.cvtColor(pixels_rgb.astype(np.uint8), cv2.COLOR_RGB2HSV)
    hsv_mean = hsv.reshape(-1, 3).mean(axis=0)
    return np.array([
        rgb_mean[0], rgb_mean[1], rgb_mean[2],
        hsv_mean[0]/179.0, hsv_mean[1]/255.0, hsv_mean[2]/255.0
    ], dtype=float)
