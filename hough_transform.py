import numpy as np
import cv2

previous_lines = None

def polar2cartesian(rho: float, theta: float) -> np.ndarray:
    return rho * np.array([np.sin(theta), np.cos(theta)])

def score_line(rho: float, theta: float, edges: np.ndarray) -> float:
    mask = np.zeros(edges.shape, dtype=edges.dtype)
    
    normal = polar2cartesian(rho, theta)
    direction = np.array([normal[1], -normal[0]])
    
    p1 = np.round(normal + 1000 * direction).astype(int)
    p2 = np.round(normal - 1000 * direction).astype(int)
    
    cv2.line(mask, tuple(p1), tuple(p2), 255, 12)
    
    return cv2.countNonZero(cv2.bitwise_and(mask, edges))


def score_all_lines(edges: np.ndarray, lines: np.ndarray) -> np.ndarray:
    scored = []
    for rho, theta in lines:
        score = score_line(rho, theta, edges)
        scored.append([rho, theta, score])
    return np.array(scored)


def filter_strong_lines(scored: np.ndarray, threshold: float = 0.3) -> np.ndarray:
    max_score = scored[:, 2].max()
    if max_score == 0:
        return np.empty((0, 3))
    
    strong = scored[scored[:, 2] > threshold * max_score]
    return strong[strong[:, 2].argsort()[::-1]]


def remove_duplicate_lines(lines: np.ndarray, 
                          delta_rho: float = 230, 
                          delta_theta: float = np.deg2rad(5),
                          max_lines: int = 10) -> np.ndarray:
    filtered = []
    
    for rho, theta, score in lines:
        keep = True
        for r2, t2, _ in filtered:
            if abs(rho - r2) < delta_rho and abs(theta - t2) < delta_theta:
                keep = False
                break
        if keep:
            filtered.append([rho, theta, score])
        if len(filtered) >= max_lines:
            break
    
    return np.array(filtered) if filtered else np.empty((0, 3))


def find_most_separated_pair(lines: np.ndarray) -> tuple:
    if len(lines) < 2:
        return 0, None
    
    max_dist = 0
    best_pair = None
    
    for i in range(len(lines)):
        for j in range(i + 1, len(lines)):
            dist = abs(lines[i][0] - lines[j][0])
            if dist > max_dist:
                max_dist = dist
                best_pair = (lines[i], lines[j])
    
    return max_dist, best_pair


def filter_lines(edges: np.ndarray, lines: np.ndarray, min_separation: int = 210):
    if lines is None or len(lines) == 0:
        return np.empty((0, 2))
    
    scored = score_all_lines(edges, lines)
    
    strong = filter_strong_lines(scored, threshold=0.3)
    if len(strong) == 0:
        return np.empty((0, 2))
    
    filtered = remove_duplicate_lines(strong, delta_rho=230, delta_theta=np.deg2rad(5))
    if len(filtered) == 0:
        return np.empty((0, 2))
    
    if len(filtered) == 1:
        return filtered[:1, :2]
    
    max_dist, best_pair = find_most_separated_pair(filtered)
    
    if max_dist < min_separation:
        return filtered[:1, :2]
    
    return np.array([
        best_pair[0][:2],
        best_pair[1][:2]
    ])

def rotate_line(lines: np.ndarray, edges: np.ndarray, 
                angle_range: float = np.deg2rad(15),
                angle_step: float = np.deg2rad(0.5),
                rho_range: float = 50,
                rho_step: float = 5) -> np.ndarray:
    if len(lines) == 0:
        return lines
    
    optimized = []
    
    for rho, theta in lines:
        best_rho = rho
        best_theta = theta
        best_score = score_line(rho, theta, edges)
        
        rho_min = rho - rho_range
        rho_max = rho + rho_range
        theta_min = theta - angle_range
        theta_max = theta + angle_range
        
        test_rho = rho_min
        while test_rho <= rho_max:
            test_theta = theta_min
            while test_theta <= theta_max:
                score = score_line(test_rho, test_theta, edges)
                
                if score > best_score:
                    best_score = score
                    best_rho = test_rho
                    best_theta = test_theta
                
                test_theta += angle_step
            test_rho += rho_step
        
        optimized.append([best_rho, best_theta])
    
    return np.array(optimized)

def hough_transform(edges: np.ndarray, threshold: int):
    global previous_lines
    
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
    lines = filter_lines(edges, np.vstack([rhos[rhos_indexes], thetas[thetas_indexes]]).T)
    lines = rotate_line(lines, edges)

    if len(lines) == 1 and previous_lines is not None and len(previous_lines) == 2:
        current_rho = lines[0][0]
        prev_rho_0 = previous_lines[0][0]
        prev_rho_1 = previous_lines[1][0]
        if abs(current_rho - prev_rho_0) < abs(current_rho - prev_rho_1):
            lines = np.vstack([lines, previous_lines[1:2]])
        else:
            lines = np.vstack([lines, previous_lines[0:1]])
    if len(lines) == 2:
        previous_lines = lines.copy()
    return lines

def draw_lines(img: np.ndarray, lines: np.ndarray, mask:np.ndarray, color: list[int], thickness: int):
    empty_image = np.zeros(img.shape[:2])
    new_img = img.copy()

    for rho, theta in lines:
        normal = polar2cartesian(rho, theta)
        direction = np.array([normal[1], -normal[0]])
        p1 = np.round(normal + 1000 * direction).astype(int)
        p2 = np.round(normal - 1000 * direction).astype(int)
        empty_image = cv2.line(img=empty_image, pt1=p1, pt2=p2, color=255, thickness=thickness)

    mask_lines = empty_image > 0
    mask = (mask * mask_lines).astype(bool)
    new_img[mask] = np.array(color)

    return new_img

