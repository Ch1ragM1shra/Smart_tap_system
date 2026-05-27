<<<<<<< HEAD
# Smart Tap — Bucket Water Level Classifier

## Project Structure
```
smart_tap/
├── smart_tap.py          # Main pipeline (hybrid CV + CNN)
├── classical_fallback.py # Pure OpenCV, no ML needed
├── color_tuner.py        # Interactive HSV calibration tool
├── deploy_pi.py          # Raspberry Pi TFLite deployment
├── requirements.txt
└── dataset/              # Your images go here
    ├── empty/
    ├── half/
    └── full/
```

## Dataset Setup
Organize your labeled images like this:
```
dataset/
    empty/      ← bucket empty or nearly empty (tag clearly visible)
    half/       ← bucket half filled (tag partially visible)
    full/       ← bucket full (tag invisible, water surface covers it)
```
Aim for at least 50 images per class. 100+ is better.

## Quick Start

### Step 1: Calibrate HSV ranges (do this first!)
```bash
python color_tuner.py --image dataset/empty/sample.jpg
# Drag sliders until the rainbow tag is white in the mask
# Press S to print the values, then update TAG_COLORS_HSV in smart_tap.py
```

### Step 2: Train
```bash
python smart_tap.py train
# Saves: smart_tap_model.h5, training_curves.png, confusion_matrix.png
```

### Step 3: Test on an image
```bash
python smart_tap.py predict --image test.jpg
```

### Step 4: Live camera
```bash
python smart_tap.py live --camera 0
```

## Classical-only mode (no ML, great starting point)
```bash
# Test on image
python classical_fallback.py path/to/image.jpg

# Live camera
python classical_fallback.py
```

## Raspberry Pi Deployment

### On your laptop — convert model:
```bash
python deploy_pi.py convert
# Creates smart_tap_int8.tflite (~1-2 MB)
```

### Copy to Pi:
```bash
scp smart_tap_int8.tflite smart_tap.py deploy_pi.py pi@raspberrypi.local:~/smart_tap/
```

### On Pi — install:
```bash
pip install tflite-runtime opencv-python-headless numpy
```

### On Pi — run:
```bash
python deploy_pi.py
```

## Tuning Tips

| Problem | Fix |
|---|---|
| Wrong predictions in bright light | Re-run color_tuner.py, update HSV ranges |
| Tag detected in full bucket | Lower `S_low` threshold for tag colors |
| Too slow on Pi | Increase frame skip interval in deploy_pi.py |
| Half vs full confused | Collect more half-filled images |
| Water reflections causing issues | Add a circular polarizer to the camera lens |

## How It Works

```
Camera frame
    │
    ▼
TagDetector (HSV color masking)
    │
    ├─ confidence ≥ 0.75 → return result  (fast path, ~2ms)
    │
    └─ confidence < 0.75 → MobileNetV2 CNN  (slow path, ~30ms)
                               │
                               └─ return result
```

Variance is measured in the ROI (center third of frame) to avoid
bucket walls and background clutter.
=======
# Smart_tap_system
AI-powered Smart Tap system using ESP32, Computer Vision, OpenCV, TensorFlow, and MobileNetV2 for real-time water level detection and automatic tap control with edge AI optimization and Raspberry Pi deployment support.
>>>>>>> 21a6e39cf93663e772d7a82bba4f26d13833869e
