import numpy as np
import math

def create_guassian_kernel(k: int, sigma: float) -> np.ndarray:
    mid = k // 2
    kernel = np.zeros((k, k), dtype=np.float32)

    for i in range(k):
        for j in range(k):
            x = i - mid
            y = j - mid
            kernel[i][j] = math.exp(-(x*x + y*y) / (2 * sigma * sigma)) / (2 * math.pi * sigma * sigma)

    return kernel / kernel.sum()

def guassian_blur(image: np.ndarray, kernel: np.ndarray) -> np.ndarray:
    k = len(kernel)
    H, W = image.shape
    blurred_img = np.zeros((H, W), dtype=np.float32)
    padded_img = np.pad(image, pad_width=(k // 2), mode="edge")

    for i in range(k):
        for j in range(k):
            blurred_img += kernel[i][j] * padded_img[i:i+H, j:j+W]
    return blurred_img.astype(np.uint8)
