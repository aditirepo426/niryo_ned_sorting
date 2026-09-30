import cv2
import numpy as np

def crop_roi(frame, roi):
    x1,y1,x2,y2 = roi
    return frame[y1:y2, x1:x2]

def clean_mask(mask):
    kernel = np.ones((5,5), np.uint8)
    mask = cv2.morphologyEx(mask, cv2.MORPH_OPEN, kernel)
    return cv2.morphologyEx(mask, cv2.MORPH_CLOSE, kernel)
