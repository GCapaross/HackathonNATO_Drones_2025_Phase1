# Merged Packets Spectrogram Generator

This script generates spectrograms from mixed signals in the `merged_packets` directory with YOLO format labels derived from CSV annotations.

## 🎯 **What It Does**

1. **Reads mixed signals** from `merged_packets/bw_*/` directories
2. **Reads CSV labels** with detailed signal annotations
3. **Converts CSV to YOLO format** (bounding box coordinates)
4. **Generates spectrograms** with colored bounding boxes
5. **Saves both images and label files** for training

## 🚀 **Usage**

### Basic Usage
```bash
python3 spectrogram_images/generator_from_merged_packets.py \
    -p spectrogram_training_data_20220711/merged_packets \
    -o output_directory \
    -r 1024 192
```

### Test with Small Sample
```bash
python3 spectrogram_images/generator_from_merged_packets.py \
    -p spectrogram_training_data_20220711/merged_packets \
    -o test_output \
    --max-samples 5
```

### Custom Settings
```bash
python3 spectrogram_images/generator_from_merged_packets.py \
    -p spectrogram_training_data_20220711/merged_packets \
    -o custom_output \
    -c viridis \
    -r 1024 192 \
    --max-samples 100
```

## 📊 **Parameters**

- `-p, --path`: Path to merged_packets directory
- `-o, --output-dir`: Output directory for generated spectrograms
- `-r, --resolution`: Image resolution as "width height" (default: 1024 192)
- `-c, --colormap`: Matplotlib colormap (default: viridis)
- `--max-samples`: Maximum number of samples to process (for testing)

## 🎨 **Output Features**

### Generated Files
- **Spectrogram images**: PNG files with bounding boxes
- **YOLO labels**: TXT files with bounding box coordinates
- **Color-coded boxes**: Different colors for each class

### Class Colors
- **Red**: WLAN signals
- **Yellow**: Collision events
- **Blue**: BT_classic signals
- **Green**: BLE_1MHz signals
- **Purple**: BLE_2MHz signals

## 🔄 **Data Flow**

```
merged_packets/ → CSV Labels → YOLO Format → Spectrograms
     ↓              ↓            ↓              ↓
Mixed signals → Annotations → Bounding boxes → Training data
```

## 📁 **Directory Structure**

```
merged_packets/
├── bw_25e6/          # 25 MHz bandwidth
│   ├── frame_*.csv   # Label files
│   └── frame_*       # Signal files
├── bw_45e6/          # 45 MHz bandwidth
├── bw_60e6/          # 60 MHz bandwidth
└── bw_125e6/         # 125 MHz bandwidth
```

## 🧪 **Testing**

Run the test script to verify everything works:

```bash
python3 test_merged_packets_generator.py
```

This will:
1. Generate a small sample of spectrograms
2. Verify the output format
3. Show example usage

## 🔧 **Key Features**

- **Multi-bandwidth support**: Handles 25MHz, 45MHz, 60MHz, 125MHz
- **CSV to YOLO conversion**: Automatic label format conversion
- **Parallel processing**: Fast generation using multiple cores
- **Error handling**: Robust processing with error reporting
- **Flexible output**: Configurable resolution and colormap

## 📈 **Performance**

- **Parallel processing**: Uses all CPU cores
- **Memory efficient**: Processes files one at a time
- **Error resilient**: Continues processing even if some files fail
- **Progress tracking**: Shows processing status

## 🎯 **Use Cases**

1. **Training data generation**: Create labeled spectrograms for ML training
2. **Data visualization**: Visualize mixed RF signals with annotations
3. **Label verification**: Check CSV to YOLO conversion accuracy
4. **Custom datasets**: Generate spectrograms with specific parameters

This generator bridges the gap between the raw mixed signals and the final training data, providing a complete pipeline from CSV annotations to YOLO format spectrograms!
