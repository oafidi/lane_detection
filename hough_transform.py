import numpy as np
import cv2

def polar2cartesian(rho: float, theta: float) -> np.ndarray:
    return rho * np.array([np.sin(theta), np.cos(theta)])

def hough_transform(edges: np.ndarray, threshold: int):
    H, W = edges.shape
    thetas = np.arange(-np.pi / 2, np.pi / 2, 0.261)
    rhos = np.arange(-np.sqrt(H**2 + W**2), np.sqrt(H**2 + W**2), 9)

    len_thetas = len(thetas)
    len_rhos = len(rhos)

    accumulator  = np.zeros((len_rhos, len_thetas))
    cos_thetas = np.cos(thetas)
    sin_thetas = np.sin(thetas)

    xs, ys = np.where(edges > 0)

    for x, y in zip(xs, ys):
        for t in range(len_thetas):
            rho = x * cos_thetas[t] + y * sin_thetas[t]
            rho_pos = np.where(rho > rhos)[0][-1]
            accumulator[rho_pos][t] += 1
    rhos_indexes, thetas_indexes = np.where(accumulator > threshold)
    return np.vstack([rhos[rhos_indexes], thetas[thetas_indexes]]).T

def draw_lines(img: np.ndarray, lines: np.ndarray, mask:np.ndarray, color: list[int], thickness: int):
    empty_image = np.zeros(img.shape[:2])
    new_img = img.copy()

    for rho, theta in lines:
        normal = polar2cartesian(rho, theta)
        direction = np.array([normal[1], -normal[0]])
        p1 = np.round(normal + 1000 * direction).astype(int)
        p2 = np.round(normal - 1000 * direction).astype(int)
        empty_image = cv2.line(img=empty_image, pt1=p1, pt2=p2, color=255, thickness=thickness)
    
    # min_diff = np.inf
    # vanishing_point_line = 0
    mask_lines = empty_image > 0
    # for i in range(mask_lines.shape[0]):
    #     line = mask_lines[i]
    #     indexes = np.argwhere(line)
    #     if (len(indexes) and indexes[-1] - indexes[0] < min_diff):
    #         min_diff = indexes[-1] - indexes[0]
    #         vanishing_point_line = i
    # mask_boundaries = np.zeros_like(empty_image)
    # mask_boundaries[vanishing_point_line:] = 1
    mask = (mask * mask_lines).astype(bool)
    new_img[mask] = np.array(color)

    return new_img

