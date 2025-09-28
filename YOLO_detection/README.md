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
3. **Optimized for speed**: Confusion matrix generation disabled
4. **CUDA acceleration**: Automatically uses GPU if available
5. Saves best model to `yolo_training/rf_detection/weights/best.pt`
6. Creates training plots and metrics

## Understanding Training Losses

YOLO training uses three main loss components:

### Box Loss (Bounding Box Loss)
- **What it measures**: How well the model predicts the exact position and size of bounding boxes
- **What it optimizes**: The coordinates (x, y, width, height) of detected objects
- **Good values**: Lower is better (typically 0.1-0.5)
- **What it means**: If box loss is high, the model is struggling to place bounding boxes accurately

### Cls Loss (Classification Loss)
- **What it measures**: How well the model predicts the correct class (Background, WLAN, Bluetooth, BLE)
- **What it optimizes**: The class probabilities for each detected object
- **Good values**: Lower is better (typically 0.1-0.3)
- **What it means**: If cls loss is high, the model is struggling to identify what type of signal it is

### DFL Loss (Distribution Focal Loss)
- **What it measures**: How confident the model is in its predictions
- **What it optimizes**: The confidence scores and objectness predictions
- **Good values**: Lower is better (typically 0.1-0.4)
- **What it means**: If DFL loss is high, the model is uncertain about its predictions

### What to Look For
**Healthy training:**
- All three losses should decrease over time
- Box loss: 0.2-0.4 (good bounding box placement)
- Cls loss: 0.1-0.3 (good class identification)
- DFL loss: 0.1-0.3 (confident predictions)

**Problem signs:**
- **Box loss high**: Model can't find objects accurately
- **Cls loss high**: Model can't identify signal types
- **DFL loss high**: Model is uncertain about everything
- **All losses stuck**: Model stopped learning (overfitting)

**For RF signal detection:**
- **Box loss**: How well it finds the signal regions
- **Cls loss**: How well it identifies WLAN vs Bluetooth vs Background
- **DFL loss**: How confident it is about detections

## Training Messages Explained

### "Duplicate labels removed" Messages
- **Normal behavior**: YOLO automatically detects and removes duplicate bounding boxes
- **Data cleaning**: Prevents the model from learning redundant information
- **Quality control**: Ensures clean training data

### "optimizer: 'optimizer=auto' found, ignoring 'lr0=0.01' and 'mo'"
- **Auto-optimizer**: YOLO uses automatic optimizer selection (usually AdamW)
- **Ignoring manual settings**: Uses proven optimization strategies instead of manual parameters
- **This is normal**: YOLO's auto-optimizer often works better than manual settings

### CUDA Usage
- **Automatic detection**: Script automatically detects and uses CUDA if available
- **Fallback to CPU**: Uses CPU if CUDA is not available
- **Device info**: Shows which device is being used during training

## Testing Process

1. Loads trained model
2. Tests on spectrogram images
3. Creates 3-panel visualizations:
   - Original spectrogram
   - Model predictions with bounding boxes
   - Human-marked ground truth (if available)
4. Saves results to `test_results/` folder

## Troubleshooting

### CUDA Not Working
If training is not using CUDA despite having a GPU:

1. **Check CUDA installation**:
   ```bash
   nvidia-smi
   python3 -c "import torch; print(torch.cuda.is_available())"
   ```

2. **Check PyTorch CUDA version**:
   ```bash
   python3 -c "import torch; print(torch.version.cuda)"
   ```

3. **Reinstall PyTorch with CUDA**:
   ```bash
   pip install torch torchvision --index-url https://download.pytorch.org/whl/cu118
   ```

### Training Issues
- **Slow training**: Normal for first epoch, speeds up after
- **Memory errors**: Reduce batch size in `train_model.py`
- **Duplicate labels**: Normal YOLO behavior, not an error
