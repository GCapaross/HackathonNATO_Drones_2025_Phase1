# YOLO Training Results - 2025-09-28_19-01-14

## Training Configuration
- **Model**: YOLOv8n (nano)
- **Dataset**: RF Signal Detection
- **Classes**: Background, WLAN, Bluetooth
- **Device**: CUDA
- **Image Size**: 640x640
- **Batch Size**: 16
- **Total Epochs**: 34
- **Report Generated**: 2025-09-28 19:01:16

## Model Files
- **Best Model**: yolo_training2/rf_detection/weights/best.pt
- **Last Model**: yolo_training2/rf_detection/weights/last.pt
- **Training Logs**: yolo_training2/rf_detection/results.csv
- **Training Plots**: yolo_training2/rf_detection/labels.jpg, train_batch*.jpg

## Training Metrics Summary

### Final Epoch Results
- **Epoch**: 34.0
- **Training Box Loss**: 0.4341
- **Training Class Loss**: 0.3493
- **Training DFL Loss**: 0.8174
- **Validation Box Loss**: 0.3546
- **Validation Class Loss**: 0.3069
- **Validation DFL Loss**: 0.7834

### Detection Metrics
- **Precision**: 0.8591
- **Recall**: 0.5990
- **mAP@50**: 0.6870
- **mAP@50-95**: 0.5570

### Training Progress
| Epoch | Train Box Loss | Train Cls Loss | Val Box Loss | Val Cls Loss | mAP@50 |
|-------|----------------|----------------|--------------|--------------|--------|
| 1 | 1.0175 | 1.1807 | 0.7875 | 0.7196 | 0.4392 |
| 2 | 0.9033 | 0.7536 | 0.7156 | 0.6514 | 0.4778 |
| 3 | 0.8866 | 0.6947 | 0.6658 | 0.6041 | 0.4931 |
| 4 | 0.8133 | 0.6169 | 0.6028 | 0.5143 | 0.5406 |
| 5 | 0.7396 | 0.5559 | 0.5965 | 0.4866 | 0.5325 |
| 6 | 0.6943 | 0.5222 | 0.5303 | 0.4513 | 0.5821 |
| 7 | 0.6608 | 0.4975 | 0.5138 | 0.4475 | 0.5760 |
| 8 | 0.6383 | 0.4808 | 0.5012 | 0.4264 | 0.6090 |
| 9 | 0.6193 | 0.4685 | 0.4890 | 0.4148 | 0.6179 |
| 10 | 0.6011 | 0.4549 | 0.4760 | 0.3981 | 0.6198 |
| 11 | 0.5857 | 0.4448 | 0.4696 | 0.3997 | 0.6268 |
| 12 | 0.5743 | 0.4363 | 0.4610 | 0.3882 | 0.6303 |
| 13 | 0.5642 | 0.4292 | 0.4483 | 0.3785 | 0.6413 |
| 14 | 0.5491 | 0.4204 | 0.4412 | 0.3706 | 0.6413 |
| 15 | 0.5412 | 0.4150 | 0.4360 | 0.3706 | 0.6423 |
| 16 | 0.5375 | 0.4120 | 0.4290 | 0.3608 | 0.6548 |
| 17 | 0.5281 | 0.4064 | 0.4224 | 0.3551 | 0.6568 |
| 18 | 0.5207 | 0.4025 | 0.4162 | 0.3514 | 0.6567 |
| 19 | 0.5122 | 0.3961 | 0.4083 | 0.3441 | 0.6602 |
| 20 | 0.5061 | 0.3930 | 0.4059 | 0.3447 | 0.6600 |
| 21 | 0.4980 | 0.3893 | 0.3995 | 0.3384 | 0.6672 |
| 22 | 0.4908 | 0.3840 | 0.3945 | 0.3356 | 0.6676 |
| 23 | 0.4821 | 0.3799 | 0.3870 | 0.3296 | 0.6715 |
| 24 | 0.4778 | 0.3771 | 0.3809 | 0.3271 | 0.6698 |
| 25 | 0.4722 | 0.3738 | 0.3773 | 0.3236 | 0.6747 |
| 26 | 0.4662 | 0.3692 | 0.3744 | 0.3221 | 0.6730 |
| 27 | 0.4646 | 0.3679 | 0.3701 | 0.3185 | 0.6779 |
| 28 | 0.4587 | 0.3652 | 0.3687 | 0.3191 | 0.6770 |
| 29 | 0.4543 | 0.3623 | 0.3641 | 0.3153 | 0.6779 |
| 30 | 0.4503 | 0.3592 | 0.3638 | 0.3120 | 0.6841 |
| 31 | 0.4464 | 0.3573 | 0.3612 | 0.3127 | 0.6824 |
| 32 | 0.4412 | 0.3539 | 0.3592 | 0.3101 | 0.6848 |
| 33 | 0.4370 | 0.3521 | 0.3559 | 0.3081 | 0.6870 |
| 34 | 0.4341 | 0.3493 | 0.3546 | 0.3069 | 0.6870 |

### Confusion Matrix
**Note**: YOLO training metrics do not include per-class confusion matrix data.
The training process only saves aggregate metrics (mAP, precision, recall).
To get a confusion matrix, you would need to:
1. Run the model on validation data
2. Collect predictions for each class
3. Compare with ground truth labels

**Available Metrics from Training:**
- **Precision**: 0.8591
- **Recall**: 0.5990
- **mAP@50**: 0.6870
- **mAP@50-95**: 0.5570

### Loss Trends
- **Training Loss**: Started at 1.0175, ended at 0.4341
- **Validation Loss**: Started at 0.7875, ended at 0.3546
- **mAP@50**: Started at 0.4392, ended at 0.6870

### Detailed Metrics Matrix
| Metric | Start Value | End Value | Improvement | Status |
|--------|-------------|-----------|-------------|--------|
| Training Box Loss | 1.0175 | 0.4341 | 0.5834 | Good |
| Validation Box Loss | 0.7875 | 0.3546 | 0.4329 | Good |
| mAP@50 | 0.4392 | 0.6870 | 0.2479 | Good |
| Precision | 0.5788 | 0.8591 | 0.2804 | Good |
| Recall | 0.4328 | 0.5990 | 0.1662 | Moderate |

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
