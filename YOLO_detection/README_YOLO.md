# YOLO RF Signal Detection System

## Overview

This system implements YOLO (You Only Look Once) object detection for RF signal detection in spectrograms. Unlike the CNN classification approach, YOLO can detect multiple signals within a single spectrogram and provide their precise locations.

## Key Advantages of YOLO Approach

### **1. Multiple Signal Detection**
- **CNN**: Predicts one class per entire image
- **YOLO**: Detects multiple signals with bounding boxes
- **Example**: One spectrogram can contain both WLAN and Bluetooth signals

### **2. Preserved Resolution**
- **CNN**: Resizes 1024×192 → 224×224 (loses detail)
- **YOLO**: Keeps original resolution 1024×192 (preserves fine details)

### **3. Spatial Information**
- **CNN**: No location information
- **YOLO**: Provides exact coordinates of each signal
- **Useful for**: Frequency analysis, signal tracking, interference detection

## Architecture

### **YOLOv8 Custom Implementation**
```
Input: Spectrogram (1024×192×3)
  ↓
Backbone: Feature extraction
  ↓
Neck: FPN + PAN (Feature Pyramid Network)
  ↓
Head: Multi-scale detection
  ↓
Output: Bounding boxes + classes
```

### **Model Components**

1. **Backbone**: Custom YOLOv8 backbone optimized for spectrograms
2. **Neck**: FPN + PAN for multi-scale feature fusion
3. **Head**: Detection heads for different scales
4. **Loss**: YOLO loss function for bounding box regression

## Dataset Format

### **Input Images**
- **Format**: PNG spectrograms (1024×192 pixels)
- **Content**: RF signals in frequency-time representation
- **Channels**: RGB (3 channels)

### **Labels (YOLO Format)**
```
class_id x_center y_center width height
0 0.5 0.5 0.1 0.2
1 0.3 0.7 0.05 0.1
2 0.8 0.4 0.08 0.15
```

**Class Mapping**:
- `0`: Background/Noise
- `1`: WLAN (WiFi)
- `2`: Bluetooth Classic
- `3`: BLE (Bluetooth Low Energy)

## Usage

### **1. Setup Environment**
```bash
cd YOLO_detection
pip install -r requirements.txt
```

### **2. Test Setup**
```bash
python test_yolo_setup.py
```

### **3. Train Model**
```bash
python train_yolo.py --data_dir ../spectrogram_training_data_20220711 --epochs 50
```

### **4. Evaluate Model**
```bash
python evaluate_yolo.py --model_path checkpoints/best_model.pth
```

## Training Process

### **Data Loading**
- Loads spectrogram images and YOLO labels
- Handles variable number of labels per image
- Preserves original resolution

### **Model Training**
- Multi-scale detection (3 scales)
- YOLO loss function
- AdamW optimizer with cosine annealing
- Batch size: 16 (adjustable)

### **Validation**
- Computes validation loss
- Saves best model based on validation loss
- Generates training curves

## Output Format

### **Detection Results**
```python
{
    'image_path': 'path/to/spectrogram.png',
    'detections': [
        {
            'class': 'WLAN',
            'confidence': 0.95,
            'bbox': [x1, y1, x2, y2],
            'frequency_range': [2.4, 2.5],  # GHz
            'time_range': [0.1, 0.3]        # seconds
        },
        {
            'class': 'Bluetooth',
            'confidence': 0.87,
            'bbox': [x1, y1, x2, y2],
            'frequency_range': [2.4, 2.48],  # GHz
            'time_range': [0.2, 0.4]         # seconds
        }
    ]
}
```

## Advantages for RF Signal Detection

### **1. Real-World Applicability**
- **Multiple signals**: Real RF environments have multiple simultaneous signals
- **Precise localization**: Know exactly where each signal is
- **Frequency analysis**: Can analyze specific frequency bands

### **2. Drone Detection**
- **Signal tracking**: Track drone signals over time
- **Interference detection**: Identify conflicting signals
- **Pattern recognition**: Detect drone communication patterns

### **3. Scalability**
- **Real-time processing**: YOLO is fast for inference
- **Batch processing**: Handle multiple spectrograms
- **Deployment ready**: Can be integrated into RF monitoring systems

## Comparison with CNN Approach

| Aspect | CNN Classification | YOLO Detection |
|--------|-------------------|----------------|
| **Input** | 224×224 (downsampled) | 1024×192 (original) |
| **Output** | Single class per image | Multiple detections per image |
| **Resolution** | Low (lost details) | High (preserved details) |
| **Signals** | One dominant signal | All signals detected |
| **Location** | No spatial info | Precise coordinates |
| **Use case** | Quick classification | Detailed analysis |

## Next Steps

1. **Train the model** on the spectrogram dataset
2. **Evaluate performance** on validation set
3. **Create visualizations** showing detected signals
4. **Optimize for real-time** inference
5. **Deploy for drone detection** applications

## Files Structure

```
YOLO_detection/
├── requirements.txt          # Dependencies
├── yolo_data_loader.py       # Data loading and preprocessing
├── yolo_model.py            # YOLO model architecture
├── train_yolo.py            # Training script
├── test_yolo_setup.py       # Setup verification
├── evaluate_yolo.py         # Model evaluation
├── visualize_detections.py   # Detection visualization
└── README_YOLO.md           # This file
```

## Technical Details

### **Model Architecture**
- **Backbone**: Custom YOLOv8 with C2f blocks
- **Neck**: FPN + PAN for multi-scale features
- **Head**: 3 detection scales (80×80, 40×40, 20×20)
- **Parameters**: ~25M (efficient for real-time)

### **Training Configuration**
- **Optimizer**: AdamW (lr=0.001, weight_decay=0.0005)
- **Scheduler**: CosineAnnealingLR
- **Batch size**: 16
- **Epochs**: 50
- **Image size**: 640×640 (resized from 1024×192)

### **Loss Function**
- **YOLO Loss**: Combines classification, localization, and confidence
- **Multi-scale**: Loss computed at 3 different scales
- **Anchor-free**: Modern YOLO approach without anchor boxes

This YOLO-based approach provides a much more suitable solution for RF signal detection, especially for drone detection applications where multiple signals need to be identified and localized simultaneously.
