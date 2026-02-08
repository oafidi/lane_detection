# Road Lane Detection

A computer vision project that detects road lanes in video footage using classical image processing techniques. This project implements the complete pipeline from scratch using NumPy and OpenCV, without relying on high-level detection libraries.

## Features

- Real-time lane detection on video streams
- Custom implementation of core algorithms (no reliance on OpenCV's built-in detection)
- Temporal smoothing using previous frame data for robust detection
- Output video generation with detected lanes overlay

## Pipeline

The lane detection pipeline consists of the following stages:

```
Input Video → Grayscale → Gaussian Blur → Edge Detection → ROI Mask → Hough Transform → Lane Lines
```

### 1. Grayscale Conversion
Converts the RGB image to grayscale by averaging the three color channels.

### 2. Gaussian Blur
Applies a custom 5x5 Gaussian kernel (σ=1.4) to reduce noise and smooth the image, which helps in edge detection.

### 3. Edge Detection (Canny-like)
Custom implementation including:
- **Sobel operators** for computing gradients (dx, dy)
- **Gradient magnitude** calculation
- **Non-maximum suppression** to thin edges
- **Double thresholding** (high=100, low=75)
- **Hysteresis** for edge tracking

### 4. Region of Interest (ROI)
Applies a trapezoidal mask to focus on the road area, filtering out irrelevant parts of the image.

### 5. Hough Transform
Custom implementation of the Hough Transform algorithm to detect lines in polar coordinates (ρ, θ). Includes:
- Line scoring based on edge support
- Duplicate line removal
- Selection of the most separated line pair
- Line rotation optimization for fine-tuning

### 6. Temporal Smoothing
If only one lane is detected in a frame, the algorithm uses the previous frame's lane data to maintain continuity.

## Project Structure

```
road_lane_detection/
├── main.py              # Main entry point
├── gray_scale.py        # Grayscale conversion & ROI masking
├── guassian_blur.py     # Gaussian blur implementation
├── edge_detection.py    # Canny edge detection implementation
├── hough_transform.py   # Hough transform & line detection
├── requirements.txt     # Python dependencies
├── Makefile             # Build commands
├── dataset/             # Input video files
│   ├── the_road.mp4
│   └── the_road_2.mp4
└── output/              # Generated output videos
```

## Requirements

- Python 3.8+
- OpenCV 4.13+
- NumPy 2.4+
- Matplotlib 3.10+ (optional, for visualization)

## Installation

1. Clone the repository:
```bash
git clone https://github.com/oafidi/lane_detection.git
cd lane_detection
```

2. Install dependencies:
```bash
make install
```

Or manually:
```bash
pip install -r requirements.txt
```

## Usage

### Run the lane detection:
```bash
make start
```

Or:
```bash
python main.py
```

### Controls
- Press `q` to quit the video playback

### Change input video
Edit `main.py` and modify the `filename` variable:
```python
filename = "dataset/the_road.mp4"  # or "dataset/the_road_2.mp4"
```

### Output
The processed video with detected lanes (green overlay) is saved to `output/<video_name>.mp4`.

## How It Works

1. **Frame Processing**: Each frame is resized to 700x500 pixels for consistent processing.

2. **Edge Detection**: The Canny-like edge detector identifies potential lane markings by detecting strong gradients in the image.

3. **ROI Filtering**: A trapezoidal region focuses the detection on the road area ahead of the vehicle.

4. **Hough Transform**: Converts edge points to Hough space to find line candidates. Lines are scored based on how many edge pixels they pass through.

5. **Line Selection**: The algorithm selects the two most prominent, well-separated lines representing the left and right lane boundaries.

6. **Temporal Consistency**: When detection fails for one lane, the previous frame's data is used to maintain stable output.

## Makefile Commands

| Command | Description |
|---------|-------------|
| `make install` | Install Python dependencies |
| `make start` | Run the lane detection |
| `make clean` | Remove Python cache files |

## Limitations

- Works best on clear road conditions with visible lane markings
- ROI parameters are tuned for specific camera angles
- May struggle with curved roads or complex intersections

## License

This project is for educational purposes.
