import cv2
from .preprocessing import crop_roi, clean_mask
from .shape_features import classify_shape

def detect_object(frame, roi, min_area=500):
    cropped = crop_roi(frame, roi)
    gray = cv2.cvtColor(cropped, cv2.COLOR_BGR2GRAY)
    blur = cv2.GaussianBlur(gray, (5,5), 0)
    _, mask = cv2.threshold(blur, 40, 255, cv2.THRESH_BINARY)
    mask = clean_mask(mask)
    contours, _ = cv2.findContours(mask, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)
    contours = [c for c in contours if cv2.contourArea(c) >= min_area]
    if not contours:
        return None
    contour = max(contours, key=cv2.contourArea)
    shape, shape_data = classify_shape(contour)
    object_mask = mask
    return {
        "roi_frame": cropped,
        "mask": object_mask,
        "contour": contour,
        "shape": shape,
        "shape_features": shape_data,
    }
