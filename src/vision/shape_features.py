import cv2
import numpy as np

def contour_features(contour):
    area = cv2.contourArea(contour)
    perimeter = cv2.arcLength(contour, True)
    x,y,w,h = cv2.boundingRect(contour)
    hull = cv2.convexHull(contour)
    hull_area = max(cv2.contourArea(hull), 1.0)
    circularity = (4*np.pi*area)/(perimeter*perimeter) if perimeter else 0.0
    extent = area / max(w*h, 1)
    solidity = area / hull_area
    aspect_ratio = w / max(h, 1)
    return {
        "area": float(area),
        "perimeter": float(perimeter),
        "circularity": float(circularity),
        "extent": float(extent),
        "solidity": float(solidity),
        "aspect_ratio": float(aspect_ratio),
    }

def classify_shape(contour):
    f = contour_features(contour)
    approx = cv2.approxPolyDP(contour, 0.04 * cv2.arcLength(contour, True), True)
    if len(approx) == 4 and 0.75 <= f["aspect_ratio"] <= 1.33 and f["extent"] > 0.65:
        return "SQUARE", f
    if f["circularity"] > 0.72:
        return "CIRCLE", f
    return "UNKNOWN", f
