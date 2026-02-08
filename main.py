from gray_scale import gray_scale, region_of_interst
from guassian_blur import guassian_blur, create_guassian_kernel
from edge_detection import edge_thresholding
from hough_transform import hough_transform, draw_lines
import cv2
import sys
import os

os.makedirs("output", exist_ok=True)
filename = "dataset/the_road.mp4"
cap = cv2.VideoCapture(filename=filename)

if not cap.isOpened():
    print("Error: Could not open video file.", file=sys.stderr)
    sys.exit(1)
else:
    print("Video file opened successfully!")

frame_count = int(cap.get(cv2.CAP_PROP_FRAME_COUNT))
fps = int(cap.get(cv2.CAP_PROP_FPS))
frame_width = int(cap.get(cv2.CAP_PROP_FRAME_WIDTH))
frame_height = int(cap.get(cv2.CAP_PROP_FRAME_HEIGHT))

fourcc = cv2.VideoWriter_fourcc(*"mp4v")
out = cv2.VideoWriter("output/"+os.path.basename(filname), fourcc, 30.0, (700, 500))

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
    frame = draw_lines(frame, lines, mask, [0, 255, 0], 7)
    cv2.imshow("Lines", frame)
    out.write(frame)
    if (cv2.waitKey(1) == ord('q')):
        break

cap.release()
out.release()
cv2.destroyAllWindows()