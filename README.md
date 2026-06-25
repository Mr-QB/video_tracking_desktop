# Video Enhancement for Tracking Evaluation

A native Python desktop application built with PySide6 for visualizing and comparing the impact of video enhancement on multi-object tracking.

## Features
- Synchronized frame-by-frame comparison of compressed vs. enhanced video.
- Interactive timeline chart mapping tracking metrics (IDF1) over time.
- Overlay controls for bounding boxes, tracking IDs, and confidence scores.
- Mock data engine to function without real video/tracking files initially.

## Installation

1. Make sure you have Python 3.10+ installed.
2. Install the required dependencies:

```bash
pip install -r requirements.txt
```

## Running the Application

```bash
python main.py
```

## Structure
- `core/`: Application logic, video playback control, and mock data generation.
- `ui/`: UI components (Main Window, Metric Cards, Timeline Chart, Video Canvas).
- `styles/`: QSS stylesheet for the application's look and feel.
