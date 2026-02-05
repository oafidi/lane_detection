import numpy as np

def gray_scale(image: np.ndarray) -> np.ndarray:
    result = np.zeros_like(image[:, :, 0], dtype=np.uint16)
    for i in range(3):
        result += image[:, :, i]
    return (result // 3).astype(np.uint8)