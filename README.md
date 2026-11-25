# Parking Lot Occupancy Detection

![Python](https://img.shields.io/badge/Python-3.9%2B-3776AB?logo=python&logoColor=white)
![OpenCV](https://img.shields.io/badge/OpenCV-computer%20vision-5C3EE8?logo=opencv&logoColor=white)
![YOLOv8](https://img.shields.io/badge/YOLOv8-Ultralytics-111F68)
![License](https://img.shields.io/badge/license-MIT-blue)

Real-time parking space monitoring from a camera feed. Each parking space is drawn once as a polygon. The system then decides, frame by frame, whether each space is **occupied (DOLU)** or **free (BOŞ)**, and mirrors the result onto a live parking map.

![Demo](docs/demo.gif)

## Features

- **Two detection modes**, switchable at runtime with `m`:
  - **YOLO mode:** YOLOv8 detects vehicles (car, motorcycle, bus, truck), and a space counts as occupied when a vehicle overlaps it.
  - **CV mode:** classic image processing (adaptive thresholding) measures texture density inside each space. It needs no neural network and suits top-down / drone cameras, where pretrained detectors struggle.
- **Visual space editor** (`editor.py`): click four points on the video to define a space, then arrange the spaces on a schematic map with drag, resize, snap-to-grid and alignment guides.
- **Live parking map** showing total, free and occupied counts next to the camera view.
- **Any video source:** a video file, a USB/HDMI camera or an RTSP IP camera.
- No database needed: spaces are stored in `spots.json`.

![YOLO mode](docs/demo-yolo.png)

## How it works

```mermaid
flowchart LR
    V[Video / camera frame] --> R[Resize 1280x720]
    R --> Y{Mode}
    Y -->|YOLO| D[YOLOv8 vehicle detection<br/>every 2nd frame]
    D --> O1[Box center inside polygon<br/>OR overlap >= 30%]
    Y -->|CV| P[Gray → Gaussian blur → adaptive threshold<br/>→ median blur → dilate]
    P --> O2[Pixel density inside polygon >= 18%]
    O1 --> S[Occupied / free per space]
    O2 --> S
    S --> UI[Overlay on video + live parking map]
```

### YOLO mode
A space is **occupied** if, for any detected vehicle:
1. the center of its bounding box lies inside the space polygon (`cv2.pointPolygonTest`), **or**
2. the box covers at least **30%** of the space area, measured on a polygon mask.

To keep the frame rate high, detection runs on every second frame.

### CV mode
Each frame goes through grayscale conversion, Gaussian blur, **adaptive Gaussian thresholding**, median blur and dilation. Empty asphalt produces few edge pixels, while a car produces many. A space is occupied when the edge-pixel density inside its polygon is at least **18%**.

> **When to use which:** with angled street-level cameras (like the sample video) YOLO is clearly more accurate. COCO-trained detectors were not trained on straight top-down imagery, which is why CV mode exists. Accuracy in both modes depends heavily on how carefully the spaces are drawn. Marking the central part of each space, rather than its full outline, often works better at steep camera angles.

## Getting started

```bash
git clone https://github.com/Mastena12/parking-lot-occupancy-detection.git
cd parking-lot-occupancy-detection
pip install -r requirements.txt
```

### Run the monitor

```bash
python main.py
```

The YOLOv8n weights (`yolov8n.pt`, ~6 MB) are downloaded automatically on the first run. A sample video and pre-drawn spaces are included, so it works out of the box.

| Key | Action |
|---|---|
| `m` | Switch between YOLO and CV mode |
| `t` | Show the thresholded image (CV mode) |
| `r` | Reload `spots.json` |
| `q` / `Esc` | Quit |

### Define your own parking spaces

```bash
python editor.py
```

| Action | Effect |
|---|---|
| Left-click 4 points on the video | Create a space |
| Right-click a space | Delete it |
| Drag a box on the map (right panel) | Move it; snaps to neighbours |
| Drag a box's corner handle | Resize all boxes |
| `s` | Save to `spots.json` |
| `d` | Delete last space |
| `r` | Clear all |
| `a` | Toggle snapping, `+` / `-` changes the snap tolerance |
| `q` / `Esc` | Save and quit |

## Using a live camera

The video source is defined in `prepare_stream.py`. Replace the file path with your camera:

```python
# IP camera over RTSP
cv2.VideoCapture("rtsp://user:password@192.168.1.100:554/stream")

# USB / HDMI capture device (shown to Windows as a webcam)
cv2.VideoCapture(0)
```

For a camera plugged directly into the computer's Ethernet port, give the camera and the PC static IPs on the same subnet and use the RTSP address.

## Project structure

```
├── main.py             # Live monitor: detection, overlay, parking map
├── editor.py           # Interactive parking space editor
├── prepare_stream.py   # Video source (file / USB / RTSP)
├── spots.json          # Saved parking spaces (polygons + map layout)
├── input_video/        # Sample video
└── docs/               # README media
```

## Tuning

| Parameter | Location | Default | Meaning |
|---|---|---|---|
| `DENSITY_THRESHOLD` | `main.py` | `0.18` | CV mode: edge-pixel ratio that marks a space as occupied |
| `min_overlap` | `check_occupancy_yolo` | `0.30` | YOLO mode: minimum share of the space covered by a vehicle box |
| `conf` | YOLO call in `main.py` | `0.15` | Detection confidence threshold |
| `VEHICLES` | `main.py` | `[2, 3, 5, 7]` | COCO classes: car, motorcycle, bus, truck |

## License

MIT. The UI labels are in Turkish: *DOLU* = occupied, *BOŞ* = free.
