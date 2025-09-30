# Real-time RF Detection Pipeline

Complete pipeline for real-time RF frame detection in spectrograms.

## Components

### 1. Real-time Spectrogram Generator (`realtime_spectrogram_generator.py`)
- Generates spectrograms from .packet files
- Merges multiple RF frames with random positioning
- Adds noise and frequency offset
- Compatible with training data format

### 2. Real-time YOLO Detector (`realtime_yolo_detector.py`)
- Detects RF frames in spectrograms using trained YOLO model
- Supports confidence thresholding and NMS
- Real-time processing with queues
- Draws detection boxes and labels

### 3. Live GUI Interface (`realtime_gui.py`)
- Live monitoring of detection results
- Manual labeling capabilities
- Model selection from available models
- Statistics and result saving

### 4. Complete Pipeline (`realtime_pipeline.py`)
- Integrates all components
- Command-line and GUI modes
- Automatic model detection
- Statistics reporting

## Usage

### GUI Mode (Recommended)
```bash
python3 realtime_pipeline.py --gui
```

### Command Line Mode
```bash
# Auto-detect model and use default packet source
python3 realtime_pipeline.py

# Specify model and packet source
python3 realtime_pipeline.py --model path/to/model.pt --packets ../spectrogram_training_data_20220711/single_packet_samples/

# Custom output directory and processing interval
python3 realtime_pipeline.py --output my_output --interval 1.5
```

### Individual Components

#### Generate Spectrograms
```python
from realtime_spectrogram_generator import RealtimeSpectrogramGenerator

generator = RealtimeSpectrogramGenerator()
image, path = generator.generate_from_packets(packet_files, "output_dir")
```

#### Run Detection
```python
from realtime_yolo_detector import RealtimeYOLODetector

detector = RealtimeYOLODetector("model.pt")
result_image, detections, output_path = detector.process_spectrogram(image)
```

#### Launch GUI
```python
from realtime_gui import RealtimeRFDetectionGUI

app = RealtimeRFDetectionGUI()
app.run()
```

## Features

### Real-time Processing
- Continuous spectrogram generation from packet files
- Live YOLO detection with configurable confidence
- Queue-based processing for smooth operation
- Automatic model selection from available models

### GUI Interface
- Live spectrogram display with detection overlays
- Manual labeling with click-to-label interface
- Real-time statistics (detection counts, processing rate)
- Save results with detection data and manual labels

### Detection Capabilities
- WLAN frame detection
- Bluetooth frame detection  
- Collision detection
- Configurable confidence thresholds
- Non-maximum suppression

### Data Management
- Automatic output organization
- JSON export of detection data
- Manual label storage
- Statistics tracking

## Requirements

- Python 3.7+
- ultralytics (YOLO)
- opencv-python
- numpy
- matplotlib
- scipy
- tkinter (for GUI)

## File Structure

```
HackathonModel2/
├── realtime_spectrogram_generator.py  # Spectrogram generation
├── realtime_yolo_detector.py         # YOLO detection
├── realtime_gui.py                   # GUI interface
├── realtime_pipeline.py              # Complete pipeline
├── README_REALTIME.md               # This file
└── realtime_output/                 # Output directory (created automatically)
    ├── spectrogram_*.png            # Generated spectrograms
    └── detection_*.png              # Detection results
```

## Configuration

### Model Selection
The GUI automatically detects available models in:
- `yolo_training_improved/`
- `yolo_training/`
- `checkpoints/`
- `models/`
- `weights/`

### Processing Parameters
- **Processing Interval**: Time between spectrogram generation (default: 2.0s)
- **Confidence Threshold**: Minimum detection confidence (default: 0.3)
- **NMS Threshold**: Non-maximum suppression threshold (default: 0.5)
- **Packet Selection**: Number of packets to merge (default: 5)

### Output Format
- **Spectrograms**: PNG images with viridis colormap
- **Detections**: Bounding boxes with class labels and confidence
- **Data Export**: JSON format with detection coordinates and metadata

## Troubleshooting

### Common Issues

1. **No models found**: Train a model first or specify path with `--model`
2. **No packet files**: Ensure packet source directory contains .packet files
3. **GUI not working**: Install tkinter: `sudo apt-get install python3-tk`
4. **CUDA errors**: Ensure PyTorch CUDA installation matches your GPU

### Performance Tips

1. **Reduce processing interval** for faster detection
2. **Lower confidence threshold** for more detections
3. **Use smaller models** (YOLOv8n) for faster inference
4. **Reduce packet count** for faster spectrogram generation

## Integration with Training

This pipeline uses the same spectrogram format as the training data:
- Same frequency range and time resolution
- Same noise characteristics
- Same class labels (WLAN, bluetooth, collision)
- Compatible with trained YOLO models

The real-time system can be used to:
1. **Test trained models** on live data
2. **Collect new training data** with manual labels
3. **Validate model performance** in real-world scenarios
4. **Fine-tune detection parameters** for optimal performance
