# RF Spectrogram YOLO Detection

This folder contains the setup and training scripts for YOLO-based detection of RF frames in spectrograms.

## Dataset

The dataset consists of 20,000 labeled spectrograms from the MDPI paper:
- **Training**: 16,000 images
- **Validation**: 4,000 images
- **Classes**: WLAN, collision, bluetooth

## Setup

1. **Install dependencies**:
   ```bash
   pip install -r requirements.txt
   ```

2. **Organize dataset**:
   ```bash
   python setup_yolo_dataset.py
   ```

3. **Train YOLO model**:
   ```bash
   python train_yolo_model.py --train
   ```

4. **Test trained model**:
   ```bash
   python train_yolo_model.py --test
   ```

## Files

- `setup_yolo_dataset.py`: Organizes the spectrogram dataset for YOLO training
- `train_yolo_model.py`: Trains and tests the YOLO model
- `requirements.txt`: Python dependencies
- `datasets/`: Organized dataset with images and labels
- `yolo_training/`: Training outputs and model weights

## Dataset Structure

```
datasets/
├── dataset.yaml          # Dataset configuration
├── images/
│   ├── train/            # Training images
│   └── val/              # Validation images
└── labels/
    ├── train/            # Training labels (YOLO format)
    └── val/              # Validation labels (YOLO format)
```

## Classes

- **0: WLAN** - Wi-Fi frames
- **1: collision** - Overlapping frames
- **2: bluetooth** - Bluetooth/BLE frames

## Training Parameters

- **Model**: YOLOv8n (nano)
- **Epochs**: 100
- **Image size**: 640x640
- **Batch size**: 16
- **Device**: CUDA (GPU)

## Usage

The trained model can detect and classify RF frames in spectrograms, which is useful for:
- Automated spectrum analysis
- RF interference detection
- Wireless network monitoring
- Drone communication analysis
