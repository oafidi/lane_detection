import numpy as np
import cv2

image = cv2.imread("dataset/edges.png")
image_gray = cv2.cvtColor(image, cv2.COLOR_BGR2GRAY)
_, edges = cv2.threshold(image_gray, 127, 255, cv2.THRESH_BINARY)
cv2.imshow("Edges", edges)
cv2.waitKey(0)

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

def draw_lines(img: np.ndarray, lines: np.ndarray, thickness: int):
    empty_image = np.zeros(img.shape[:2])

    for rho, theta in lines:
        normal = polar2cartesian(rho, theta)
        direction = np.array([normal[1], -normal[0]])
        p1 = normal + 1000 * direction
        p2 = normal - 1000 * direction
        empty_image = cv2.line(img=empty_image, pt1=p1, pt2=p2, color=255, thickness=thickness)
    
    
print(hough_transform(edges, 360))
