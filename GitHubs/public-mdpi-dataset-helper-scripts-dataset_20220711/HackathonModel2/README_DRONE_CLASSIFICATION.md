# Drone Classification System

Based on the paper: **"Combined RF-Based Drone Detection and Classification"**  
*IEEE TRANSACTIONS ON COGNITIVE COMMUNICATIONS AND NETWORKING, VOL. 8, NO. 1, MARCH 2022*

## Overview

This system implements a **two-stage approach** for drone detection and classification:

1. **Stage 1**: RF Signal Detection using YOLO (your existing system)
2. **Stage 2**: Drone Classification using YOLO-Lite (new system)

## System Architecture

```
Raw RF Data → Spectrogram → YOLO Detection → YOLO-Lite Classification → Drone Type
```

### Components

- **`drone_rf_spectrogram_generator.py`**: Generates spectrograms from DroneRF dataset
- **`drone_classification_system.py`**: YOLO-Lite model for drone classification
- **`integrated_drone_detection_system.py`**: Combines both systems
- **`train_drone_classification_pipeline.py`**: Complete training pipeline

## Dataset Structure

The system uses the **DroneRF dataset** with the following structure:

```
DroneRF/
├── Bepop drone/
│   ├── RF Data_10000_H/
│   ├── RF Data_10000_L/
│   └── ...
├── AR drone/
│   ├── RF Data_10100_H/
│   ├── RF Data_10100_L/
│   └── ...
├── Phantom drone/
│   ├── RF Data_11000_H/
│   └── ...
└── Background RF activites/
    ├── RF Data_00000_H1/
    └── ...
```

## Installation

1. **Install dependencies**:
```bash
pip install -r requirements_drone_classification.txt
```

2. **Verify DroneRF dataset**:
```bash
ls /home/gabriel/Desktop/HackathonNATO_Drones_2025/DroneRF/
```

## Quick Start

### 1. Complete Training Pipeline

Run the complete training pipeline:

```bash
python3 train_drone_classification_pipeline.py \
    --drone_rf_path /home/gabriel/Desktop/HackathonNATO_Drones_2025/DroneRF \
    --output_dir drone_classification_results \
    --max_files 100 \
    --epochs 50 \
    --batch_size 16
```

### 2. Generate Spectrograms Only

```bash
python3 drone_rf_spectrogram_generator.py \
    --drone_rf_path /home/gabriel/Desktop/HackathonNATO_Drones_2025/DroneRF \
    --output_dir drone_spectrograms \
    --max_files 50
```

### 3. Train Classification Model

```bash
python3 drone_classification_system.py \
    --data_dir drone_spectrograms \
    --epochs 100 \
    --batch_size 32 \
    --model_path drone_classifier.pth
```

### 4. Test Integrated System

```bash
python3 integrated_drone_detection_system.py \
    --rf_model yolo_training_improved/rf_spectrogram_detection_improved4/weights/best.pt \
    --drone_classifier drone_classifier.pth \
    --spectrogram path/to/spectrogram.png \
    --output_dir test_results
```

## System Features

### Stage 1: RF Signal Detection
- **Input**: Spectrogram image
- **Output**: RF signal detections (WLAN, Bluetooth, Background)
- **Model**: Your existing YOLO model
- **Classes**: Background, WLAN, Bluetooth

### Stage 2: Drone Classification
- **Input**: Spectrogram image
- **Output**: Drone type classification
- **Model**: YOLO-Lite (new)
- **Classes**: Bebop, AR Drone, Phantom, Background

### Combined Analysis
- **Threat Assessment**: Low/Medium/High
- **Communication Analysis**: WLAN/Bluetooth detection
- **Recommendations**: Actionable insights

## Technical Details

### YOLO-Lite Architecture
Based on the paper's Table II:

```
Input: 256x256x3 spectrogram
├── Conv1: 3→16, 3x3, LeakyReLU
├── MaxPool: 2x2
├── Conv2: 16→32, 3x3, LeakyReLU
├── MaxPool: 2x2
├── Conv3: 32→64, 3x3, LeakyReLU
├── MaxPool: 2x2
├── Conv4: 64→128, 3x3, LeakyReLU
├── MaxPool: 2x2
├── Conv5: 128→256, 3x3, LeakyReLU
├── MaxPool: 2x2
├── Conv6: 256→512, 3x3, LeakyReLU
├── MaxPool: 2x2
├── Conv7: 512→1024, 3x3, LeakyReLU
├── MaxPool: 2x2
└── FC: 1024*4*4 → Grid*Grid*(Boxes*5 + Classes)
```

### Spectrogram Generation
- **FFT Size**: 2048
- **Overlap**: 50%
- **Window**: Hann
- **Frequency Range**: 2.4-2.48 GHz
- **Output**: 256x256 spectrogram images

## Performance Metrics

Based on the paper:

| Metric | Single Signal | Multi-Signal |
|--------|---------------|--------------|
| Detection Accuracy | 99.7% | 96% |
| Classification F1-Score | 97% | 97% |
| Processing Time | ~3.4x faster than DRNN | - |

## Output Examples

### Detection Results
```json
{
  "stage_1_rf_detection": [
    {
      "type": "WLAN",
      "confidence": 0.85,
      "bbox": [100, 150, 200, 250]
    }
  ],
  "stage_2_drone_classification": {
    "predicted_class": "bebop",
    "confidence": 0.92,
    "all_probabilities": {
      "bebop": 0.92,
      "ar_drone": 0.05,
      "phantom": 0.02,
      "background": 0.01
    }
  },
  "combined_analysis": {
    "threat_level": "high",
    "drone_present": true,
    "drone_type": "bebop",
    "communication_types": ["WLAN"],
    "recommendations": [
      "Drone detected: bebop",
      "Threat level: high",
      "WLAN communication detected - possible video transmission"
    ]
  }
}
```

## Integration with Existing System

The new system integrates seamlessly with your existing RF detection:

1. **Keep your existing YOLO model** for RF signal detection
2. **Add the new YOLO-Lite model** for drone classification
3. **Combine results** for comprehensive analysis

### Integration Code Example

```python
from integrated_drone_detection_system import IntegratedDroneSystem

# Initialize system
system = IntegratedDroneSystem(
    rf_model_path="your_rf_model.pt",
    drone_classifier_path="drone_classifier.pth"
)

# Process spectrogram
results = system.process_spectrogram(spectrogram_image)

# Get combined analysis
analysis = results['combined_analysis']
print(f"Drone: {analysis['drone_present']}")
print(f"Type: {analysis['drone_type']}")
print(f"Threat: {analysis['threat_level']}")
```

## Troubleshooting

### Common Issues

1. **"DroneRF dataset not found"**
   - Check path: `/home/gabriel/Desktop/HackathonNATO_Drones_2025/DroneRF`
   - Ensure folder structure matches expected format

2. **"CUDA out of memory"**
   - Reduce batch size: `--batch_size 8`
   - Use CPU: `--device cpu`

3. **"No spectrograms generated"**
   - Check CSV file format
   - Verify DroneRF dataset structure
   - Check file permissions

### Performance Optimization

1. **For faster training**:
   - Reduce `--max_files` parameter
   - Use smaller `--batch_size`
   - Reduce `--epochs`

2. **For better accuracy**:
   - Increase `--max_files` parameter
   - Train for more epochs
   - Use data augmentation

## Paper Reference

**Title**: Combined RF-Based Drone Detection and Classification  
**Authors**: Sanjoy Basak, Sreeraj Rajendran, Sofie Pollin, Bart Scheers  
**Journal**: IEEE TRANSACTIONS ON COGNITIVE COMMUNICATIONS AND NETWORKING, VOL. 8, NO. 1, MARCH 2022  
**DOI**: 10.1109/TCCN.2021.3099114

## License

This implementation is for research and educational purposes. Please cite the original paper if you use this code.

## Support

For issues and questions:
1. Check the troubleshooting section
2. Verify your DroneRF dataset structure
3. Ensure all dependencies are installed
4. Check file permissions and paths
