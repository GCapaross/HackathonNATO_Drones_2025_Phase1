# CNN Training Updates and Fixes

## Issues Identified

### 1. **Multi-target Label Error**
**Error**: `0D or 1D target tensor expected, multi-target not supported`

**Root Cause**: The data loader was returning lists of class IDs instead of single class labels for each image. CrossEntropyLoss expects single target labels, not multi-target.

**Current Code (Problematic)**:
```python
def extract_class_labels(self, labels):
    """Extract unique class labels for classification task"""
    if not labels:
        return [0]  # Background if no labels
    
    # Get unique class IDs
    class_ids = list(set([label[0] for label in labels]))
    return class_ids  # Returns list, but CrossEntropyLoss needs single value
```

**Fix Needed**:
```python
def extract_class_labels(self, labels):
    """Extract primary class label for classification task"""
    if not labels:
        return 0  # Background if no labels
    
    # Get the most frequent class ID (primary signal type)
    class_ids = [label[0] for label in labels]
    from collections import Counter
    most_common = Counter(class_ids).most_common(1)[0][0]
    return most_common  # Returns single integer
```

### 2. **Class Imbalance Problem**
**Issue**: Model achieving 99.92% accuracy but only predicting class 0 (Background)

**Root Cause**: The dataset has severe class imbalance where most spectrograms are dominated by background signals (class 0), making the model learn to always predict background.

**Evidence from Training Output**:
```
Sample 1: Classes [0, 0, 1, 0, 0, 2, 2, 0, 0, 0, 0, 1, 0, 0, 0, 0, 0, 1, 0, 0, 0, 0, 0, 0, 0, 1, 0, 0] -> Primary: 0
```

**Fix Strategy**: Prioritize non-background signals when present:
```python
def extract_class_labels(self, labels):
    """Extract primary class label for classification task"""
    if not labels:
        return 0  # Background if no labels
    
    # Get class IDs and their counts
    class_ids = [label[0] for label in labels]
    from collections import Counter
    class_counts = Counter(class_ids)
    
    # Strategy: If there are any non-background signals, prioritize them
    non_background_classes = [cid for cid in class_counts.keys() if cid != 0]
    
    if non_background_classes:
        # If there are non-background signals, use the most frequent one
        most_common = Counter(class_ids).most_common(1)[0][0]
        # But if background is more frequent, still prefer non-background
        if most_common == 0 and len(non_background_classes) > 0:
            # Get the most frequent non-background class
            non_bg_counts = {k: v for k, v in class_counts.items() if k != 0}
            most_common = max(non_bg_counts, key=non_bg_counts.get)
    else:
        # Only background signals
        most_common = 0
    
    return most_common
```

### 3. **Evaluation Report Error**
**Error**: `Number of classes, 2, does not match size of target_names, 4`

**Root Cause**: The model only predicted 2 classes (0 and 1) but the evaluation tried to use all 4 class names.

**Fix Needed**:
```python
# In train_cnn.py evaluate_model method
# Get unique classes present in predictions and labels
unique_labels = sorted(list(set(all_labels + all_predictions)))
target_names_subset = [self.class_names[i] for i in unique_labels]

report = classification_report(all_labels, all_predictions, 
                             target_names=target_names_subset,
                             labels=unique_labels,
                             output_dict=True)
```

## Implementation Status

- [x] **Multi-target Label Error**: Fixed in data_loader.py
- [x] **Class Imbalance Strategy**: Implemented in data_loader.py  
- [x] **Evaluation Report Error**: Fixed in train_cnn.py
- [ ] **Testing**: Need to run test_setup.py to verify fixes
- [ ] **Training**: Need to retrain with improved data loader

## Code Changes Applied

### data_loader.py
- Fixed `extract_class_labels()` to return single integer instead of list
- Added class imbalance handling to prioritize non-background signals
- Enhanced debugging output to show class counts

### train_cnn.py  
- Fixed evaluation report to handle dynamic class labels
- Added logic to get unique classes present in predictions
- Updated classification report generation

## Next Steps

1. **Test the fixes**: Run `python test_setup.py` to verify all issues are resolved
2. **Retrain model**: Run `python train_cnn.py --epochs 50` with improved data loader
3. **Monitor class distribution**: Check if model now learns to distinguish between signal types
4. **Evaluate performance**: Verify that accuracy reflects true multi-class performance

## Testing Your Trained Model

After training, you have several ways to test your model:

### 1. **Comprehensive Testing** (`test_model.py`)
```bash
# Test on validation set with detailed metrics
python test_model.py --model_path best_model.pth

# Test on specific number of random samples
python test_model.py --model_path best_model.pth --random_samples 20

# Test a single image
python test_model.py --model_path best_model.pth --test_single path/to/image.png
```

**What it does:**
- Tests model on validation set
- Generates confusion matrix and per-class accuracy plots
- Shows detailed classification report
- Tests random samples for quick verification
- Analyzes prediction patterns and confidence

### 2. **Real-time Inference** (`inference.py`)
```bash
# Classify a single image
python inference.py --model_path best_model.pth --image path/to/spectrogram.png

# Classify multiple images
python inference.py --model_path best_model.pth --batch image1.png image2.png image3.png
```

**What it does:**
- Loads trained model for real-time classification
- Shows class probabilities with visual bars
- Supports batch processing of multiple images
- Perfect for testing on new spectrograms

### 3. **Quick Setup Test** (`test_setup.py`)
```bash
# Verify everything works before training
python test_setup.py
```

## Expected Results After Fixes

With the improved data loader, you should see:
- **Better class balance**: Model learns all 4 classes, not just background
- **Meaningful accuracy**: True multi-class performance instead of 99% background prediction
- **Working evaluation**: No more tensor dimension errors
- **Detailed metrics**: Confusion matrix and per-class accuracy plots

## Expected Improvements

After implementing these fixes:
- No more multi-target tensor errors
- Better class balance in training data
- Model should learn to distinguish between WLAN, Bluetooth, BLE, and Background
- Evaluation reports should work correctly
- More meaningful accuracy metrics

## Files Modified

- `data_loader.py`: Fixed label extraction and class imbalance handling
- `train_cnn.py`: Fixed evaluation report generation
- `updates.md`: This documentation file
