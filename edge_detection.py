import numpy as np

def first_derivative_x(image):
    kernel = np.array([[-1,0,1],[-2,0,2],[-1,0,1]], dtype=np.float32)

    H,W = image.shape
    padded = np.pad(image, 1, mode="edge")
    dx = np.zeros((H,W),dtype=np.float32)

    for i in range(3):
        for j in range(3):
            dx += kernel[i,j] * padded[i:i+H,j:j+W]

    return dx

def first_derivative_y(image):
    kernel = np.array([[1,2,1],[0,0,0],[-1,-2,-1]], dtype=np.float32)

    H,W = image.shape
    padded = np.pad(image, 1, mode="edge")
    dy = np.zeros((H,W),dtype=np.float32)

    for i in range(3):
        for j in range(3):
            dy += kernel[i,j] * padded[i:i+H,j:j+W]

    return dy

def gradient_magnitude(dx,dy):
    magnitude = np.abs(dx) + np.abs(dy)
    return magnitude

def gradient_orientation(dx: np.ndarray, dy: np.ndarray) -> np.ndarray:
    return np.arctan2(dy, dx)

def non_max_suppression(mag, ang):
    H, W = mag.shape
    Z = np.zeros((H, W), dtype=np.float32)
    angle = ang * 180.0 / np.pi
    angle[angle < 0] += 180

    M = np.pad(mag, 1, mode="constant", constant_values=0)
    center = M[1:-1, 1:-1]
    left  = M[1:-1, :-2]
    right = M[1:-1, 2:]
    up    = M[:-2, 1:-1]
    down  = M[2:, 1:-1]
    upleft  = M[:-2, :-2]
    upright = M[:-2, 2:]
    downleft  = M[2:, :-2]
    downright = M[2:, 2:]

    Z0   = (center > left) & (center >= right)
    Z90  = (center > up) & (center >= down)
    Z45  = (center > downleft) & (center >= upright)
    Z135 = (center > upleft) & (center >= downright)

    mask0   = ((angle < 22.5) | (angle >= 157.5)) & Z0
    mask45  = ((angle >= 22.5) & (angle < 67.5)) & Z45
    mask90  = ((angle >= 67.5) & (angle < 112.5)) & Z90
    mask135 = ((angle >= 112.5) & (angle < 157.5)) & Z135

    mask = mask0 | mask45 | mask90 | mask135
    Z[mask] = center[mask]
    return Z

def threshold(nms: np.ndarray, high_th: int, low_th: int) -> np.ndarray:
    thresholded = np.zeros(nms.shape, dtype=np.uint8)

    strong_mask = nms >= high_th
    weak_mask = (nms >= low_th) & (nms < high_th)
    
    thresholded[weak_mask] = 25
    thresholded[strong_mask] = 255
    return thresholded

def hysteresis_vectorized(img):
    img = img.copy().astype(np.uint8)
    M, N = img.shape

    while True:
        padded = np.pad(img, pad_width=1, mode='constant', constant_values=0)
        neighbors = np.stack([
            padded[:-2, :-2], padded[:-2, 1:-1], padded[:-2, 2:],
            padded[1:-1, :-2],                   padded[1:-1, 2:],
            padded[2:, :-2],  padded[2:, 1:-1],  padded[2:, 2:]
        ], axis=0)
        any_strong_neighbor = np.any(neighbors == 255, axis=0)
        weak_mask = (img == 25)
        to_promote = weak_mask & any_strong_neighbor
        if not np.any(to_promote):
            break
        img[to_promote] = 255

    img[img == 25] = 0
    return img

def edge_thresholding(image: np.ndarray, high_th: int, low_th: int) -> np.ndarray:
    dx = first_derivative_x(image)
    dy = first_derivative_y(image)
    magnitude = gradient_magnitude(dx, dy)
    orientation = gradient_orientation(dx, dy)
    nms = non_max_suppression(mag=magnitude, ang=orientation)
    thresholded = threshold(nms, high_th, low_th)

    return hysteresis_vectorized(thresholded)

