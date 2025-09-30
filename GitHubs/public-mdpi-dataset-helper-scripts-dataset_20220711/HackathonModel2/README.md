# Balanced RF Spectrogram Detection (20 Epochs)

This folder contains scripts for training a YOLO model with a **balanced dataset** (33% each class) and **20 epochs** for quick testing.

## Quick Start

1. **Create balanced dataset**:
   ```bash
   python setup_balanced_dataset.py
   ```

2. **Train with 20 epochs**:
   ```bash
   python train_yolo_improved.py
   ```

3. **Test balanced model**:
   ```bash
   python test_balanced_model.py
   ```

## What's Different

### **Balanced Dataset (33% each class)**
- **WLAN**: 33% (downsampled from 71.3%)
- **Collision**: 33% (upsampled from 16.0%)
- **Bluetooth**: 33% (upsampled from 12.7%)

### **Quick Training (20 epochs)**
- **Epochs**: 20 (vs 200) - Fast training for testing
- **Model**: YOLOv8s - Better than nano
- **Loss**: Focal loss - Handles class imbalance
- **Time**: ~30-60 minutes on GPU

## Files

- `setup_balanced_dataset.py`: Creates balanced dataset (33% each class)
- `train_yolo_improved.py`: Trains model for 20 epochs
- `test_balanced_model.py`: Tests the balanced model
- `analyze_dataset.py`: Analyzes class distribution
- `README_IMPROVEMENTS.md`: Detailed improvement strategies

## Expected Results

### **With Balanced Dataset:**
- **Equal representation** of all classes
- **Better Bluetooth detection** (more training examples)
- **Reduced WLAN bias** (fewer examples)
- **Faster convergence** (20 epochs)

### **Training Time:**
- **GPU**: 30-60 minutes
- **CPU**: 2-3 hours (not recommended)

## Usage Examples

### **Complete Workflow:**
```bash
# 1. Create balanced dataset
python setup_balanced_dataset.py

# 2. Train with 20 epochs
python train_yolo_improved.py

# 3. Test the model
python test_balanced_model.py --num_images 10

# 4. Test with different confidence
python test_balanced_model.py --confidence 0.1
```

### **Quick Testing:**
```bash
# Test with low confidence for Bluetooth
python test_balanced_model.py --confidence 0.1 --num_images 5

# Test on validation set
python test_balanced_model.py --image_dir datasets_balanced/images/val --num_images 20
```

## Output Folders

- `datasets_balanced/`: Balanced training data (33% each class)
- `yolo_training_improved/`: Training outputs and model weights
- `balanced_test_results/`: Test visualization results

## Advantages of Balanced Approach

1. **Equal Learning**: All classes get equal attention during training
2. **Better Bluetooth Detection**: More Bluetooth examples in training
3. **Reduced Bias**: Less focus on majority class (WLAN)
4. **Faster Training**: 20 epochs vs 200 epochs
5. **Quick Testing**: Fast iteration and experimentation

## Comparison

| Aspect | Original | Balanced |
|--------|----------|----------|
| WLAN | 71.3% | 33% |
| Collision | 16.0% | 33% |
| Bluetooth | 12.7% | 33% |
| Epochs | 200 | 20 |
| Training Time | 4-6 hours | 30-60 min |
| Bluetooth Detection | Poor | Better |

The balanced approach should give you much better Bluetooth detection with faster training!
