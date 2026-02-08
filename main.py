from gray_scale import gray_scale, region_of_interst
from guassian_blur import guassian_blur, create_guassian_kernel
from edge_detection import edge_thresholding
from hough_transform import hough_transform, draw_lines
import cv2
import sys

cap = cv2.VideoCapture(filename="dataset/the_road.mp4")

if not cap.isOpened():
    print("Error: Could not open video file.", file=sys.stderr)
    sys.exit(1)
else:
    print("Video file opened successfully!")

frame_count = int(cap.get(cv2.CAP_PROP_FRAME_COUNT))
fps = int(cap.get(cv2.CAP_PROP_FPS))

print(f"frame count: {frame_count}  fps: {fps}")

kernel = create_guassian_kernel(k=5, sigma=1.4)
while True:
    ret, frame = cap.read()
    if not ret:
        print("End of video or error occurred.")
        break
    frame = cv2.resize(frame, (700, 500))
    gray_frame = gray_scale(frame)
    blurred_frame = guassian_blur(gray_frame, kernel=kernel)

    edges = edge_thresholding(blurred_frame, high_th=100, low_th=75)
    region, mask = region_of_interst(edges)
    lines = hough_transform(region, 180)
    frame = draw_lines(region, lines, mask, 255, 3)
    cv2.imshow("Lines", frame)
    cv2.imshow("Edge", region)
    if (cv2.waitKey(1) == ord('q')):
        break

cap.release()
cv2.destroyAllWindows()