# YOLO Training Results - 2025-09-28_15-19-30

## Training Configuration
- **Model**: YOLOv8n (nano)
- **Dataset**: RF Signal Detection
- **Classes**: Background, WLAN, Bluetooth
- **Device**: CPU
- **Image Size**: 640x640
- **Batch Size**: 16
- **Total Epochs**: 19
- **Report Generated**: 2025-09-28 15:19:32

## Model Files
- **Best Model**: yolo_training/rf_detection/weights/best.pt
- **Last Model**: yolo_training/rf_detection/weights/last.pt
- **Training Logs**: yolo_training/rf_detection/results.csv
- **Training Plots**: yolo_training/rf_detection/labels.jpg, train_batch*.jpg

## Training Metrics Summary

### Final Epoch Results
- **Epoch**: 19.0
- **Training Box Loss**: 0.5116
- **Training Class Loss**: 0.3957
- **Training DFL Loss**: 0.8297
- **Validation Box Loss**: 0.4105
- **Validation Class Loss**: 0.3467
- **Validation DFL Loss**: 0.7954

### Detection Metrics
- **Precision**: 0.8391
- **Recall**: 0.5759
- **mAP@50**: 0.6581
- **mAP@50-95**: 0.5128

### Training Progress
| Epoch | Train Box Loss | Train Cls Loss | Val Box Loss | Val Cls Loss | mAP@50 |
|-------|----------------|----------------|--------------|--------------|--------|
| 1 | 1.0169 | 1.1793 | 0.7944 | 0.7448 | 0.4315 |
| 2 | 0.9027 | 0.7527 | 0.7207 | 0.6497 | 0.4805 |
| 3 | 0.8854 | 0.6925 | 0.6871 | 0.6066 | 0.4979 |
| 4 | 0.8117 | 0.6166 | 0.5912 | 0.5240 | 0.5382 |
| 5 | 0.7372 | 0.5554 | 0.5921 | 0.4860 | 0.5529 |
| 6 | 0.6942 | 0.5209 | 0.5373 | 0.4513 | 0.5787 |
| 7 | 0.6610 | 0.4964 | 0.5156 | 0.4421 | 0.5912 |
| 8 | 0.6393 | 0.4807 | 0.5027 | 0.4420 | 0.6010 |
| 9 | 0.6195 | 0.4688 | 0.4845 | 0.4138 | 0.6175 |
| 10 | 0.6019 | 0.4553 | 0.4804 | 0.4038 | 0.6229 |
| 11 | 0.5870 | 0.4453 | 0.4717 | 0.4069 | 0.6203 |
| 12 | 0.5747 | 0.4365 | 0.4607 | 0.3905 | 0.6316 |
| 13 | 0.5650 | 0.4304 | 0.4485 | 0.3844 | 0.6391 |
| 14 | 0.5504 | 0.4212 | 0.4389 | 0.3710 | 0.6461 |
| 15 | 0.5421 | 0.4158 | 0.4301 | 0.3649 | 0.6446 |
| 16 | 0.5387 | 0.4132 | 0.4290 | 0.3627 | 0.6515 |
| 17 | 0.5283 | 0.4070 | 0.4193 | 0.3572 | 0.6547 |
| 18 | 0.5199 | 0.4021 | 0.4169 | 0.3527 | 0.6339 |
| 19 | 0.5116 | 0.3957 | 0.4105 | 0.3467 | 0.6581 |

### Loss Trends
- **Training Loss**: Started at 1.0169, ended at 0.5116
- **Validation Loss**: Started at 0.7944, ended at 0.4105
- **mAP@50**: Started at 0.4315, ended at 0.6581

### Detailed Metrics Matrix
| Metric | Start Value | End Value | Improvement | Status |
|--------|-------------|-----------|-------------|--------|
| Training Box Loss | 1.0169 | 0.5116 | 0.5053 | Good |
| Validation Box Loss | 0.7944 | 0.4105 | 0.3839 | Good |
| mAP@50 | 0.4315 | 0.6581 | 0.2266 | Good |
| Precision | 0.5655 | 0.8391 | 0.2736 | Good |
| Recall | 0.4294 | 0.5759 | 0.1465 | Moderate |

### Performance Analysis
- **Good Performance**: mAP@50 > 0.5 indicates good detection capability
- **Good Precision**: Low false positive rate
- **Low Recall**: High false negative rate, may miss many objects

### Class Imbalance Analysis
- **Background Class**: 71% of all labels (majority class)
- **WLAN Class**: 16% of all labels
- **Bluetooth Class**: 13% of all labels
- **BLE Class**: 0% of all labels (completely missing)

**Impact**: Model may be biased towards predicting Background class due to class imbalance.

## Next Steps
1. Test the model using: `python3 test_model.py`
2. Analyze test results in `yolo_testing_results/` folder
3. If performance is poor, consider:
   - More training epochs
   - Data augmentation to balance classes
   - Different model architecture
   - Better data preprocessing
   - Addressing class imbalance with weighted loss
