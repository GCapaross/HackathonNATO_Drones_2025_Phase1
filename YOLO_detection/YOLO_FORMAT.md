# YOLO Label Format Explanation

## YOLO Label Format

```
class_id x_center y_center width height
```

### Example: `0 0.095281 0.5 0.163841 0.8`

| Position | Value | Meaning | Description |
|----------|-------|---------|-------------|
| **1** | `0` | **Class ID** | Signal type (0=Background, 1=WLAN, 2=Bluetooth, 3=BLE) |
| **2** | `0.095281` | **X Center** | Horizontal center position (normalized 0-1) |
| **3** | `0.5` | **Y Center** | Vertical center position (normalized 0-1) |
| **4** | `0.163841` | **Width** | Bounding box width (normalized 0-1) |
| **5** | `0.8` | **Height** | Bounding box height (normalized 0-1) |

## What This Specific Label Means

### Signal Type: Background (Class 0)
- This is a **background/noise signal** (not WLAN, Bluetooth, or BLE)

### Position: 
- **X Center**: 9.5% from the left edge of the image
- **Y Center**: 50% from the top (middle of the image vertically)

### Size:
- **Width**: 16.4% of the image width
- **Height**: 80% of the image height

## Converting to Pixel Coordinates

For a **1024×192 spectrogram**:

```python
# Convert normalized coordinates to pixels
x_center_px = 0.095281 * 1024 = 97.5 pixels
y_center_px = 0.5 * 192 = 96 pixels
width_px = 0.163841 * 1024 = 167.8 pixels  
height_px = 0.8 * 192 = 153.6 pixels

# Convert to corner coordinates
x1 = 97.5 - 167.8/2 = 13.6 pixels
y1 = 96 - 153.6/2 = 19.2 pixels
x2 = 97.5 + 167.8/2 = 181.4 pixels
y2 = 96 + 153.6/2 = 172.8 pixels
```

## Visual Interpretation

This label represents a **background signal** that:
- **Starts** at pixel (14, 19)
- **Ends** at pixel (181, 173)
- **Size**: 168×154 pixels
- **Location**: Near the left edge, middle height
- **Shape**: Wide and tall (covers most of the vertical space)

## Why Normalized Coordinates?

**Advantages**:
- **Resolution independent**: Works with any image size
- **Consistent**: Same coordinates regardless of image dimensions
- **Efficient**: Smaller file sizes
- **Standard**: YOLO format used by most object detection models

**Example**: The same label `0 0.095281 0.5 0.163841 0.8` works for:
- 1024×192 image → (14, 19) to (181, 173)
- 2048×384 image → (28, 38) to (362, 346)
- 512×96 image → (7, 10) to (91, 87)

## Class Mapping

| Class ID | Class Name | Color | Description |
|----------|------------|-------|-------------|
| **0** | Background | Gray | Noise, empty spectrum, or non-signal areas |
| **1** | WLAN | Red | WiFi signals (802.11 protocols) |
| **2** | Bluetooth | Blue | Bluetooth Classic signals |
| **3** | BLE | Green | Bluetooth Low Energy signals |

## Real Examples from Dataset

### Example 1: Background Signal
```
0 0.095281 0.5 0.163841 0.8
```
- **Class**: Background (0)
- **Position**: Left side, middle height
- **Size**: Medium width, very tall

### Example 2: WLAN Signal
```
1 0.5 0.3 0.2 0.1
```
- **Class**: WLAN (1)
- **Position**: Center horizontally, upper third vertically
- **Size**: Medium width, short height

### Example 3: Bluetooth Signal
```
2 0.8 0.7 0.1 0.2
```
- **Class**: Bluetooth (2)
- **Position**: Right side, lower third
- **Size**: Narrow width, medium height

## Conversion Functions

### Normalized to Pixels
```python
def normalize_to_pixels(normalized_coords, image_width, image_height):
    x_center, y_center, width, height = normalized_coords
    
    # Convert to pixel coordinates
    x_center_px = x_center * image_width
    y_center_px = y_center * image_height
    width_px = width * image_width
    height_px = height * image_height
    
    # Convert to corner coordinates
    x1 = int(x_center_px - width_px / 2)
    y1 = int(y_center_px - height_px / 2)
    x2 = int(x_center_px + width_px / 2)
    y2 = int(y_center_px + height_px / 2)
    
    return (x1, y1, x2, y2)
```

### Pixels to Normalized
```python
def pixels_to_normalize(x1, y1, x2, y2, image_width, image_height):
    # Calculate center and dimensions
    x_center = (x1 + x2) / 2 / image_width
    y_center = (y1 + y2) / 2 / image_height
    width = (x2 - x1) / image_width
    height = (y2 - y1) / image_height
    
    return (x_center, y_center, width, height)
```

## File Structure

Each spectrogram has a corresponding `.txt` file with multiple labels:

```
# result_frame_138769090412766230_bw_25E+6.txt
0 0.095281 0.5 0.163841 0.8
1 0.3 0.2 0.1 0.05
2 0.7 0.8 0.08 0.15
0 0.9 0.1 0.05 0.1
```

**Meaning**: This spectrogram contains:
- 2 Background signals (class 0)
- 1 WLAN signal (class 1) 
- 1 Bluetooth signal (class 2)

## Usage in YOLO Training

1. **Load image**: Read the spectrogram PNG file
2. **Load labels**: Parse the corresponding TXT file
3. **Convert coordinates**: Transform normalized to pixel coordinates
4. **Draw bounding boxes**: Visualize for verification
5. **Train model**: Use for YOLO object detection training

This format allows our YOLO model to learn to detect multiple RF signals within a single spectrogram image.
