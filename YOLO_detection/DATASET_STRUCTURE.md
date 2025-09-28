# Dataset File Structure Explanation

## **📁 File Structure for Each Spectrogram**

For each spectrogram sample, there are **4 files** with the same base name:

### **Example: `result_frame_138769090412766230_bw_25E+6`**

| File | Type | Size | Purpose |
|------|------|------|---------|
| `result_frame_138769090412766230_bw_25E+6` | OpenPGP Public Key | ~900KB | **Raw RF signal data** (encrypted/compressed) |
| `result_frame_138769090412766230_bw_25E+6.png` | PNG Image | ~280KB | **Spectrogram visualization** (training input) |
| `result_frame_138769090412766230_bw_25E+6_marked.png` | PNG Image | ~280KB | **Ground truth visualization** (with bounding boxes) |
| `result_frame_138769090412766230_bw_25E+6.txt` | Text File | ~30-800B | **YOLO labels** (bounding box coordinates) |

## **🔍 File Details**

### **1. Raw Data File (no extension)**
- **Format**: OpenPGP Public Key (encrypted/compressed)
- **Content**: Raw RF signal data
- **Usage**: Not used in our training (we use the PNG version)
- **Size**: ~900KB per file

### **2. Spectrogram PNG**
- **Format**: PNG image
- **Resolution**: 1024×192 pixels (typical)
- **Content**: Visual representation of RF signals
- **Usage**: **Main training input** for our YOLO model
- **Size**: ~280KB per file

### **3. Marked PNG (Ground Truth)**
- **Format**: PNG image
- **Resolution**: Same as spectrogram PNG
- **Content**: Spectrogram with bounding boxes drawn
- **Usage**: **Visual verification** of ground truth labels
- **Size**: ~280KB per file

### **4. YOLO Labels TXT**
- **Format**: Plain text, YOLO format
- **Content**: Bounding box coordinates and class labels
- **Usage**: **Training labels** for YOLO model
- **Size**: ~30-800 bytes per file

## **📊 Dataset Statistics**

- **Total files**: ~80,000 (20,000 spectrograms × 4 files each)
- **Total size**: ~60GB
- **Images used for training**: 20,000 PNG files
- **Label files**: 20,000 TXT files
- **Ground truth images**: 20,000 marked PNG files
- **Raw data files**: 20,000 encrypted files (not used)

## **🎯 What We Use for Training**

### **Input Files:**
- ✅ **Spectrogram PNG files** (`*.png`, excluding `*_marked.png`)
- ✅ **YOLO label files** (`*.txt`)

### **Not Used:**
- ❌ **Raw data files** (no extension) - encrypted, not needed
- ❌ **Marked PNG files** (`*_marked.png`) - only for visualization

## **🔧 File Filtering Logic**

```python
# Get spectrogram images (training input)
image_files = glob.glob(os.path.join(results_dir, '*.png'))
image_files = [f for f in image_files if 'marked' not in f]

# Get corresponding label files
label_files = [img.replace('.png', '.txt') for img in image_files]

# Filter to only include pairs that exist
valid_pairs = [(img, lbl) for img, lbl in zip(image_files, label_files) 
               if os.path.exists(lbl)]
```

## **📝 Filename Pattern**

```
result_frame_{timestamp}_bw_{bandwidth}E+6
```

**Example**: `result_frame_138769090412766230_bw_25E+6`

- **`result_frame`**: Prefix indicating this is a processed frame
- **`138769090412766230`**: Timestamp (Unix timestamp in microseconds)
- **`bw_25E+6`**: Bandwidth = 25 MHz (25×10^6 Hz)
- **Available bandwidths**: 25MHz, 45MHz, 60MHz, 125MHz

## **🚀 Usage in YOLO Training**

1. **Load spectrogram PNG** → Input image for YOLO
2. **Load corresponding TXT** → YOLO labels for training
3. **Convert normalized coordinates** → Pixel coordinates for visualization
4. **Train YOLO model** → Learn to detect signals in spectrograms
5. **Use marked PNG** → Verify ground truth during development

This structure allows us to train a YOLO model that can detect multiple RF signals within a single spectrogram, providing both classification and localization capabilities.
