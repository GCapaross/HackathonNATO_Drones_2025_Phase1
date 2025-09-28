# YOLO RF Signal Detection

Simple YOLO implementation for detecting RF signals in spectrograms using Ultralytics YOLOv8.

## Quick Start

### 1. Train the Model
```bash
python3 train_model.py
```

### 2. Test the Model
```bash
# Test multiple images
python3 test_model.py

# Test single image
python3 test_model.py --image_path path/to/image.png

# Test with custom confidence threshold
python3 test_model.py --conf_threshold 0.3
```

## Files

- `train_model.py` - Train YOLO model using Ultralytics YOLOv8
- `test_model.py` - Test trained model on spectrograms
- `dataset.yaml` - YOLO dataset configuration
- `datasets/` - Training and validation data
- `yolo_training/` - Training results and model weights
- `test_results/` - Test visualizations

## Model Output

The model detects 3 classes of RF signals:
- **Background** (red boxes)
- **WLAN** (blue boxes) 
- **Bluetooth** (green boxes)

## Requirements

- ultralytics
- torch
- torchvision
- matplotlib
- pillow
- pyyaml

## Training Process

1. Automatically splits dataset 80/20 train/validation
2. Uses YOLOv8n (nano) for fast training
3. Saves best model to `yolo_training/rf_detection/weights/best.pt`
4. Creates training plots and metrics

## Testing Process

1. Loads trained model
2. Tests on spectrogram images
3. Creates 3-panel visualizations:
   - Original spectrogram
   - Model predictions with bounding boxes
   - Human-marked ground truth (if available)
4. Saves results to `test_results/` folder
