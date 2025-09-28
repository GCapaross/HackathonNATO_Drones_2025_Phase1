# Image Analysis: result_frame_138769260662662470_bw_125E+6

## Files in this folder:

### 📁 Original Data
- **`original_spectrogram.png`**: The raw spectrogram image (1024×192 pixels)
- **`raw_data`**: Raw RF signal data (OpenPGP encrypted, ~900KB)

### 📁 Ground Truth
- **`ground_truth_marked.png`**: Human-labeled version with bounding boxes
- **`labels.txt`**: YOLO format labels (bounding box coordinates)

### 📁 Analysis
- **`labeling_comparison.png`**: 3-panel comparison showing:
  - Panel 1: Original spectrogram (clean)
  - Panel 2: Ground truth (marked image)
  - Panel 3: Our generated labels (from .txt file)

## Label Classes:
- **0**: Background/Noise (Gray)
- **1**: WLAN/WiFi (Red)
- **2**: Bluetooth Classic (Blue)
- **3**: BLE - Bluetooth Low Energy (Green)

## Purpose:
This folder contains all files related to one spectrogram sample, allowing you to:
1. Compare the original image with ground truth
2. Verify that our .txt coordinate conversion is accurate
3. Check if all signals are properly labeled
4. Analyze the quality of the labeling system

## Usage:
- Open `labeling_comparison.png` to see the 3-panel analysis
- Compare `original_spectrogram.png` with `ground_truth_marked.png`
- Check `labels.txt` for the exact YOLO coordinates
