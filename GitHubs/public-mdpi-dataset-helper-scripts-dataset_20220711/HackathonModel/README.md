# RF Spectrogram YOLO Detection

This folder contains the setup and training scripts for YOLO-based detection of RF frames in spectrograms.

## Quick Start

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
   python test_model_visualization.py --compare
   ```

## Dataset Analysis Results

- **Total Images**: 20,000 spectrograms
- **Training**: 16,000 images (369,843 objects)
- **Validation**: 4,000 images (92,837 objects)
- **Class Distribution**:
  - WLAN: 71.3% (263,586 objects)
  - Collision: 16.0% (59,115 objects)
  - Bluetooth: 12.7% (47,142 objects)

## Files

### Core Scripts
- `setup_yolo_dataset.py`: Organizes the spectrogram dataset for YOLO training
- `train_yolo_model.py`: Basic YOLO training
- `test_model_visualization.py`: Test and visualize model predictions

### Analysis & Improvement Scripts
- `analyze_dataset.py`: Analyze dataset class distribution
- `train_yolo_improved.py`: Enhanced training with better parameters
- `test_bluetooth_detection.py`: Bluetooth-specific testing and analysis

### Documentation
- `README_IMPROVEMENTS.md`: Detailed improvement strategies and analysis

## Classes

- **0: WLAN** - Wi-Fi frames (71.3% of data)
- **1: collision** - Overlapping frames (16.0% of data)
- **2: bluetooth** - Bluetooth/BLE frames (12.7% of data)

## Training Parameters

### Basic Training
- **Model**: YOLOv8n (nano)
- **Epochs**: 100
- **Image size**: 640x640
- **Batch size**: 16
- **Device**: CUDA (GPU)

### Improved Training (Recommended)
- **Model**: YOLOv8s (small)
- **Epochs**: 200
- **Learning rate**: 0.005
- **Loss function**: Focal loss (handles class imbalance)
- **Augmentation**: Enhanced (mixup, copy-paste)

## Usage Examples

### Basic Testing
```bash
# Test with side-by-side comparison
python test_model_visualization.py --image_dir datasets/images/val --num_images 5 --compare

# Test with low confidence for Bluetooth
python test_model_visualization.py --confidence 0.1 --compare
```

### Advanced Analysis
```bash
# Analyze dataset balance
python analyze_dataset.py

# Test Bluetooth detection specifically
python test_bluetooth_detection.py --analysis

# Train with improved parameters
python train_yolo_improved.py
```

## Known Issues & Solutions

### Bluetooth Detection Issues
- **Problem**: Bluetooth signals are underrepresented (12.7% vs 71.3% WLAN)
- **Solution**: Use `train_yolo_improved.py` with focal loss and class weighting
- **Testing**: Use confidence threshold 0.1-0.2 for Bluetooth detection

### Class Imbalance
- **Problem**: Model focuses on majority class (WLAN)
- **Solution**: Enhanced training with focal loss and more epochs
- **Analysis**: Run `analyze_dataset.py` to understand distribution

## Output Folders

- `datasets/`: Organized training data
- `yolo_training/`: Basic training outputs
- `yolo_training_improved/`: Enhanced training outputs
- `model_comparisons/`: Side-by-side comparison images
- `bluetooth_analysis/`: Bluetooth-specific analysis results

## Applications

The trained model can detect and classify RF frames in spectrograms for:
- Automated spectrum analysis
- RF interference detection
- Wireless network monitoring
- Drone communication analysis
- IoT device detection
