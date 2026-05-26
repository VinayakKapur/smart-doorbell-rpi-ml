# Smart Doorbell — Raspberry Pi + TensorFlow Lite

A lightweight smart doorbell system that runs real-time facial recognition and delivery person detection on a Raspberry Pi. When triggered by motion or a button press, the system captures a frame, classifies the visitor, and sends an SMS alert to the homeowner.

## What It Does

- **Facial recognition** — identifies registered household members using a MobileNet-based classifier trained on custom face data
- **Stranger detection** — flags unrecognized visitors
- **Delivery person detection** — a second MobileNet model trained on a Roboflow delivery-worker dataset identifies couriers separately
- **SMS alerts** — sends a notification via ClickSend with the classification result

## Architecture

```
Camera (PiCamera)
    │
    ▼
Frame capture (capture.py)
    │
    ▼
MobileNet classifier (TFLite, faces.h5)
    │
    ├── Known person  →  "Welcome, [Name]"
    ├── Delivery      →  "Delivery person at door"
    └── Stranger      →  SMS alert + optional snapshot
```

## Tech Stack

- Python 3, TensorFlow Lite, Keras
- OpenCV for image preprocessing
- MobileNet V2 (transfer learning, fine-tuned on custom face dataset)
- ClickSend API for SMS
- Raspberry Pi 4 (tested), PiCamera module

## Files

| File | Description |
|---|---|
| `main.py` | Entry point — starts camera loop and dispatches classification |
| `capture.py` | Camera capture and frame preprocessing |
| `classify.py` | TFLite inference wrapper |
| `train.py` | Model training script (MobileNet fine-tuning) |
| `sms.py` | ClickSend SMS dispatch |
| [Project Report (PDF)](../../releases/latest) | Full project report with accuracy benchmarks and design rationale |

## Model Download

The pre-trained `faces.h5` model is attached as a [GitHub Release asset](../../releases) (33 MB). Download it and place it in the project root before running.

To retrain on your own face dataset:
```bash
python train.py --data_dir ./your_faces/ --epochs 20
```

## Datasets

Training data is **not** included in this repo. Sources used:

- Face recognition: [Celebrity Faces Dataset on Kaggle](https://www.kaggle.com/datasets/vishesh1412/celebrity-face-image-dataset) (used as base, augmented with custom household photos)
- Delivery person detection: [Roboflow delivery worker dataset](https://universe.roboflow.com) — see `delivery man detection/README.roboflow.txt` for full attribution

## Setup

```bash
# Install dependencies (on Raspberry Pi or any Linux system)
pip install tensorflow opencv-python picamera numpy

# Set your ClickSend credentials in sms.py
# (replace YOUR_CLICKSEND_USERNAME and YOUR_CLICKSEND_API_KEY)

# Run
python main.py
```

## Results

From the project report (`docs/report.pdf`):

- Face classification accuracy: **~92%** on held-out test set
- Delivery person detection mAP: **~87%**
- Inference latency on Raspberry Pi 4: **~1.2s per frame**

## Contributors

Built as ECE 535 senior capstone at UMass Amherst:

- Vinayak Kapur
- [See full team in project report]

## License

MIT
