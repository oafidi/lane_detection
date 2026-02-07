import numpy as np
import cv2

def gray_scale(image: np.ndarray) -> np.ndarray:
    result = np.zeros_like(image[:, :, 0], dtype=np.uint16)
    for i in range(3):
        result += image[:, :, i]
    return (result // 3).astype(np.uint8)

def region_of_interst(image: np.ndarray) -> np.ndarray :
    mask = np.zeros_like(image)
    rows, cols = image.shape
    new_img = image.copy()

    bottom_left  = [cols * 0.1, rows * 0.95]
    top_left     = [cols * 0.4, rows * 0.6]
    bottom_right = [cols * 0.9, rows * 0.95]
    top_right    = [cols * 0.6, rows * 0.6]

    points = np.array([[bottom_left, top_left, top_right, bottom_right]], dtype=np.int32)
    cv2.fillPoly(mask, points, 1)
    return mask * new_img, mask