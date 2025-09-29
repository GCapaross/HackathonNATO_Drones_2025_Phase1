# Enhanced Spectrogram Generation with Frequency Labels and YOLO Bounding Boxes

This directory contains enhanced versions of the spectrogram generation scripts that create spectrograms with:

1. **Frequency axis labels** (MHz) for better signal analysis
2. **YOLO bounding box labels** overlaid for object detection training
3. **Class labels** on bounding boxes with color coding
4. **Proper coordinate system handling** to fix white image issues

## Overview

The enhanced scripts (`generator_with_labels.py` and `main_with_labels.py`) are exact copies of the original `generator.py` and `main.py` but with added functionality to:

1. **Read YOLO format labels** from `.txt` files
2. **Draw bounding boxes** on spectrograms with different colors for each class
3. **Add class labels** as text on the bounding boxes
4. **Generate training data** ready for YOLO model training

## Class Mapping

The dataset uses the following class IDs:
- **0**: WLAN (Red bounding boxes)
- **1**: collision (Yellow bounding boxes) 
- **2**: BT_classic (Blue bounding boxes)
- **3**: BLE_1MHz (Green bounding boxes)
- **4**: BLE_2MHz (Purple bounding boxes)

## Usage

### Enhanced Frequency Labeling (Recommended)

```bash
# Generate spectrograms with frequency axis labels and YOLO bounding boxes
python3 spectrogram_images/main_with_frequency_labels.py \
    -p /path/to/spectrogram_training_data_20220711 \
    -r 1024 192 \
    -o frequency_spectrograms \
    --center-freq 2.412e9
```

### YOLO Training Format (No Frequency Axis)

```bash
# Generate spectrograms for YOLO training (no frequency axis)
python3 spectrogram_images/main_with_frequency_labels.py \
    -p /path/to/spectrogram_training_data_20220711 \
    -r 1024 192 \
    -o yolo_training_data \
    --no-frequency-axis
```

### Basic YOLO Labels (Original)

```bash
# Generate spectrograms with YOLO labels (original version)
python3 spectrogram_images/main_with_labels.py \
    -p /path/to/spectrogram_training_data_20220711 \
    -r 1024 192 \
    -o yolo_training_data
```

### Advanced Usage

```bash
# Generate with custom settings
python3 spectrogram_images/main_with_labels.py \
    -p /path/to/dataset \
    -c viridis \
    -r 1024 192 \
    -o output_directory \
    --show-labels \
    --max-samples 100
```

### Parameters

- `-p, --path`: Root path of the spectrogram dataset
- `-c, --colormap`: Matplotlib colormap (default: viridis)
- `-r, --resolution`: Image resolution as "width height" (default: 1024 192)
- `-o, --output-dir`: Output directory for marked spectrograms (default: results2)
- `--show-labels`: Show class labels on bounding boxes
- `--max-samples`: Maximum number of samples to process (for testing)

## File Structure

The scripts expect the following structure:
```
dataset_root/
├── results/                    # Signal sample files (.bin, .dat, etc.)
│   ├── result_frame_*.txt      # YOLO label files
│   └── result_frame_*.bin     # Signal data files
├── merged_packets/            # Additional label data
└── config_training_data.toml  # Configuration file
```

## YOLO Label Format

The `.txt` label files contain YOLO format annotations:
```
class_id x_center y_center width height
```

Where all coordinates are normalized (0.0 to 1.0).

## Output

The enhanced scripts generate:
- **Spectrograms with bounding boxes** overlaid in different colors
- **Class labels** displayed on each bounding box
- **Training-ready images** for YOLO model training

## Testing

Run the test script to verify everything works:

```bash
python3 test_yolo_generation.py
```

This will:
1. Generate a small sample of spectrograms with labels
2. Verify the output format
3. Show example usage

## Integration with YOLO Training

The generated spectrograms with bounding boxes can be used directly for YOLO model training:

1. **Images**: The generated PNG files with bounding boxes
2. **Labels**: The original `.txt` files in YOLO format
3. **Classes**: 5 classes (WLAN, collision, BT_classic, BLE_1MHz, BLE_2MHz)

## Key Features

### Enhanced Visualization
- **Color-coded bounding boxes** for easy class identification
- **Class labels** displayed on each bounding box
- **High-quality output** suitable for training data

### Robust Processing
- **Multi-threaded generation** for fast processing
- **Error handling** for missing label files
- **Flexible configuration** via command-line arguments

### YOLO Compatibility
- **Standard YOLO format** labels
- **Normalized coordinates** (0.0 to 1.0)
- **Multiple classes** support
- **Training-ready output**

## Example Workflow

1. **Prepare dataset**: Ensure you have the spectrogram training data
2. **Generate labeled spectrograms**: Run the enhanced scripts
3. **Verify output**: Check the generated images have proper bounding boxes
4. **Train YOLO model**: Use the generated data for object detection training

## Troubleshooting

### Common Issues

1. **Missing label files**: Some signal files may not have corresponding `.txt` label files
2. **Resolution warnings**: Ensure resolution is a multiple of 32 for YOLO compatibility
3. **Memory issues**: Use `--max-samples` to process smaller batches

### Performance Tips

- Use `--max-samples` for testing with small datasets
- Process in batches for large datasets
- Monitor disk space for output files

## Dependencies

- matplotlib
- numpy
- pandas
- joblib
- pathlib

All dependencies are included in the original package requirements.
