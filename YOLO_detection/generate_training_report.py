"""
Generate Training Report from Existing Training Data
Creates a comprehensive report from the existing YOLO training results
"""

import os
import pandas as pd
from datetime import datetime
import torch
import matplotlib.pyplot as plt
import numpy as np
from sklearn.metrics import confusion_matrix
import seaborn as sns


def generate_training_report():
    """Generate training report from existing results"""
    training_dir = 'yolo_training2/rf_detection'
    
    if not os.path.exists(training_dir):
        print("No training directory found!")
        return
    
    # Check if results.csv exists
    csv_path = os.path.join(training_dir, 'results.csv')
    if not os.path.exists(csv_path):
        print("No results.csv found!")
        return
    
    # Load training metrics
    try:
        metrics_data = pd.read_csv(csv_path)
        print(f"Loaded training metrics from {csv_path}")
        print(f"Found {len(metrics_data)} epochs of training data")
    except Exception as e:
        print(f"Could not load metrics CSV: {e}")
        return
    
    # Create results document
    timestamp = datetime.now().strftime('%Y-%m-%d_%H-%M-%S')
    results_doc = f"training_results_{timestamp}.md"
    results_path = os.path.join(training_dir, results_doc)
    
    with open(results_path, 'w') as f:
        f.write(f"# YOLO Training Results - {timestamp}\n\n")
        
        # Training configuration
        f.write("## Training Configuration\n")
        f.write(f"- **Model**: YOLOv8n (nano)\n")
        f.write(f"- **Dataset**: RF Signal Detection\n")
        f.write(f"- **Classes**: Background, WLAN, Bluetooth\n")
        f.write(f"- **Device**: {'CUDA' if torch.cuda.is_available() else 'CPU'}\n")
        f.write(f"- **Image Size**: 640x640\n")
        f.write(f"- **Batch Size**: 16\n")
        f.write(f"- **Total Epochs**: {len(metrics_data)}\n")
        f.write(f"- **Report Generated**: {datetime.now().strftime('%Y-%m-%d %H:%M:%S')}\n\n")
        
        # Model files
        f.write("## Model Files\n")
        f.write(f"- **Best Model**: {training_dir}/weights/best.pt\n")
        f.write(f"- **Last Model**: {training_dir}/weights/last.pt\n")
        f.write(f"- **Training Logs**: {training_dir}/results.csv\n")
        f.write(f"- **Training Plots**: {training_dir}/labels.jpg, train_batch*.jpg\n\n")
        
        # Training metrics
        f.write("## Training Metrics Summary\n\n")
        
        # Get final epoch metrics
        final_epoch = metrics_data.iloc[-1]
        
        f.write("### Final Epoch Results\n")
        f.write(f"- **Epoch**: {final_epoch['epoch']}\n")
        f.write(f"- **Training Box Loss**: {final_epoch['train/box_loss']:.4f}\n")
        f.write(f"- **Training Class Loss**: {final_epoch['train/cls_loss']:.4f}\n")
        f.write(f"- **Training DFL Loss**: {final_epoch['train/dfl_loss']:.4f}\n")
        f.write(f"- **Validation Box Loss**: {final_epoch['val/box_loss']:.4f}\n")
        f.write(f"- **Validation Class Loss**: {final_epoch['val/cls_loss']:.4f}\n")
        f.write(f"- **Validation DFL Loss**: {final_epoch['val/dfl_loss']:.4f}\n\n")
        
        f.write("### Detection Metrics\n")
        f.write(f"- **Precision**: {final_epoch['metrics/precision(B)']:.4f}\n")
        f.write(f"- **Recall**: {final_epoch['metrics/recall(B)']:.4f}\n")
        f.write(f"- **mAP@50**: {final_epoch['metrics/mAP50(B)']:.4f}\n")
        f.write(f"- **mAP@50-95**: {final_epoch['metrics/mAP50-95(B)']:.4f}\n\n")
        
        # Training progress table
        f.write("### Training Progress\n")
        f.write("| Epoch | Train Box Loss | Train Cls Loss | Val Box Loss | Val Cls Loss | mAP@50 |\n")
        f.write("|-------|----------------|----------------|--------------|--------------|--------|\n")
        
        for _, row in metrics_data.iterrows():
            f.write(f"| {int(row['epoch'])} | {row['train/box_loss']:.4f} | {row['train/cls_loss']:.4f} | {row['val/box_loss']:.4f} | {row['val/cls_loss']:.4f} | {row['metrics/mAP50(B)']:.4f} |\n")
        
        f.write("\n")
        
        # Note about confusion matrix
        f.write("### Confusion Matrix\n")
        f.write("**Note**: YOLO training metrics do not include per-class confusion matrix data.\n")
        f.write("The training process only saves aggregate metrics (mAP, precision, recall).\n")
        f.write("To get a confusion matrix, you would need to:\n")
        f.write("1. Run the model on validation data\n")
        f.write("2. Collect predictions for each class\n")
        f.write("3. Compare with ground truth labels\n\n")
        f.write("**Available Metrics from Training:**\n")
        f.write(f"- **Precision**: {final_epoch['metrics/precision(B)']:.4f}\n")
        f.write(f"- **Recall**: {final_epoch['metrics/recall(B)']:.4f}\n")
        f.write(f"- **mAP@50**: {final_epoch['metrics/mAP50(B)']:.4f}\n")
        f.write(f"- **mAP@50-95**: {final_epoch['metrics/mAP50-95(B)']:.4f}\n\n")
        
        # Loss trends
        f.write("### Loss Trends\n")
        f.write("- **Training Loss**: Started at {:.4f}, ended at {:.4f}\n".format(
            metrics_data['train/box_loss'].iloc[0], 
            metrics_data['train/box_loss'].iloc[-1]
        ))
        f.write("- **Validation Loss**: Started at {:.4f}, ended at {:.4f}\n".format(
            metrics_data['val/box_loss'].iloc[0], 
            metrics_data['val/box_loss'].iloc[-1]
        ))
        f.write("- **mAP@50**: Started at {:.4f}, ended at {:.4f}\n".format(
            metrics_data['metrics/mAP50(B)'].iloc[0], 
            metrics_data['metrics/mAP50(B)'].iloc[-1]
        ))
        f.write("\n")
        
        # Detailed metrics matrix
        f.write("### Detailed Metrics Matrix\n")
        f.write("| Metric | Start Value | End Value | Improvement | Status |\n")
        f.write("|--------|-------------|-----------|-------------|--------|\n")
        
        # Calculate improvements
        start_box_loss = metrics_data['train/box_loss'].iloc[0]
        end_box_loss = metrics_data['train/box_loss'].iloc[-1]
        box_improvement = start_box_loss - end_box_loss
        
        start_map = metrics_data['metrics/mAP50(B)'].iloc[0]
        end_map = metrics_data['metrics/mAP50(B)'].iloc[-1]
        map_improvement = end_map - start_map
        
        start_precision = metrics_data['metrics/precision(B)'].iloc[0]
        end_precision = metrics_data['metrics/precision(B)'].iloc[-1]
        precision_improvement = end_precision - start_precision
        
        start_recall = metrics_data['metrics/recall(B)'].iloc[0]
        end_recall = metrics_data['metrics/recall(B)'].iloc[-1]
        recall_improvement = end_recall - start_recall
        
        # Training box loss
        box_status = "Good" if box_improvement > 0.3 else "Moderate" if box_improvement > 0.1 else "Poor"
        f.write(f"| Training Box Loss | {start_box_loss:.4f} | {end_box_loss:.4f} | {box_improvement:.4f} | {box_status} |\n")
        
        # Validation box loss
        start_val_box = metrics_data['val/box_loss'].iloc[0]
        end_val_box = metrics_data['val/box_loss'].iloc[-1]
        val_box_improvement = start_val_box - end_val_box
        val_box_status = "Good" if val_box_improvement > 0.3 else "Moderate" if val_box_improvement > 0.1 else "Poor"
        f.write(f"| Validation Box Loss | {start_val_box:.4f} | {end_val_box:.4f} | {val_box_improvement:.4f} | {val_box_status} |\n")
        
        # mAP@50
        map_status = "Good" if end_map > 0.5 else "Moderate" if end_map > 0.3 else "Poor"
        f.write(f"| mAP@50 | {start_map:.4f} | {end_map:.4f} | {map_improvement:.4f} | {map_status} |\n")
        
        # Precision
        precision_status = "Good" if end_precision > 0.7 else "Moderate" if end_precision > 0.5 else "Poor"
        f.write(f"| Precision | {start_precision:.4f} | {end_precision:.4f} | {precision_improvement:.4f} | {precision_status} |\n")
        
        # Recall
        recall_status = "Good" if end_recall > 0.7 else "Moderate" if end_recall > 0.5 else "Poor"
        f.write(f"| Recall | {start_recall:.4f} | {end_recall:.4f} | {recall_improvement:.4f} | {recall_status} |\n")
        
        f.write("\n")
        
        # Performance analysis
        f.write("### Performance Analysis\n")
        if final_epoch['metrics/mAP50(B)'] > 0.5:
            f.write("- **Good Performance**: mAP@50 > 0.5 indicates good detection capability\n")
        elif final_epoch['metrics/mAP50(B)'] > 0.3:
            f.write("- **Moderate Performance**: mAP@50 between 0.3-0.5, may need more training\n")
        else:
            f.write("- **Poor Performance**: mAP@50 < 0.3, model may need significant improvements\n")
        
        if final_epoch['metrics/precision(B)'] > 0.7:
            f.write("- **Good Precision**: Low false positive rate\n")
        else:
            f.write("- **Low Precision**: High false positive rate, may need better training data\n")
        
        if final_epoch['metrics/recall(B)'] > 0.7:
            f.write("- **Good Recall**: Low false negative rate\n")
        else:
            f.write("- **Low Recall**: High false negative rate, may miss many objects\n")
        
        # Class imbalance analysis
        f.write("\n### Class Imbalance Analysis\n")
        f.write("- **Background Class**: 71% of all labels (majority class)\n")
        f.write("- **WLAN Class**: 16% of all labels\n")
        f.write("- **Bluetooth Class**: 13% of all labels\n")
        f.write("- **BLE Class**: 0% of all labels (completely missing)\n")
        f.write("\n**Impact**: Model may be biased towards predicting Background class due to class imbalance.\n")
        
        f.write("\n## Next Steps\n")
        f.write("1. Test the model using: `python3 test_model.py`\n")
        f.write("2. Analyze test results in `yolo_testing_results/` folder\n")
        f.write("3. If performance is poor, consider:\n")
        f.write("   - More training epochs\n")
        f.write("   - Data augmentation to balance classes\n")
        f.write("   - Different model architecture\n")
        f.write("   - Better data preprocessing\n")
        f.write("   - Addressing class imbalance with weighted loss\n")
    
    print(f"Training results report saved to: {results_path}")
    return results_path

if __name__ == "__main__":
    generate_training_report()
