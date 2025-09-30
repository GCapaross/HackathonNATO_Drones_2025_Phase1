"""
Train YOLO Model for RF Signal Detection
Simple script to train YOLO using Ultralytics YOLOv8
"""

import os
# Set CUDA environment before importing PyTorch
os.environ['CUDA_VISIBLE_DEVICES'] = '0'

import torch
from ultralytics import YOLO
import yaml
import shutil
from pathlib import Path
import pandas as pd
from datetime import datetime
import numpy as np
# import matplotlib.pyplot as plt
# import seaborn as sns
# from sklearn.metrics import confusion_matrix
import random

def setup_dataset():
    """Setup dataset for YOLO training"""
    print("=== Setting up dataset ===")
    
    # Create dataset directories
    os.makedirs('datasets/train/images', exist_ok=True)
    os.makedirs('datasets/train/labels', exist_ok=True)
    os.makedirs('datasets/val/images', exist_ok=True)
    os.makedirs('datasets/val/labels', exist_ok=True)
    
    # Get all image files from the dataset
    results_dir = '../spectrogram_training_data_20220711/results'
    image_files = []
    for ext in ['*.png']:
        image_files.extend(Path(results_dir).glob(ext))
    
    # Filter out marked images
    image_files = [f for f in image_files if 'marked' not in str(f)]
    
    print(f"Found {len(image_files)} images to process")
    
    # Split into train/val (80/20)
    split_idx = int(len(image_files) * 0.8)
    train_files = image_files[:split_idx]
    val_files = image_files[split_idx:]
    
    # Process training files
    print("Processing training files...")
    for i, img_path in enumerate(train_files):
        if i % 1000 == 0:
            print(f"  {i}/{len(train_files)}")
        
        # Copy image
        dst_img = f'datasets/train/images/{img_path.name}'
        shutil.copy2(img_path, dst_img)
        
        # Copy label (if exists)
        label_path = img_path.with_suffix('.txt')
        if label_path.exists():
            dst_label = f'datasets/train/labels/{label_path.name}'
            shutil.copy2(label_path, dst_label)
    
    # Process validation files
    print("Processing validation files...")
    for i, img_path in enumerate(val_files):
        if i % 1000 == 0:
            print(f"  {i}/{len(val_files)}")
        
        # Copy image
        dst_img = f'datasets/val/images/{img_path.name}'
        shutil.copy2(img_path, dst_img)
        
        # Copy label (if exists)
        label_path = img_path.with_suffix('.txt')
        if label_path.exists():
            dst_label = f'datasets/val/labels/{label_path.name}'
            shutil.copy2(label_path, dst_label)
    
    print(f"Dataset setup complete:")
    print(f"  Training: {len(train_files)} images")
    print(f"  Validation: {len(val_files)} images")

def create_dataset_config():
    """Create YOLO dataset configuration"""
    dataset_config = {
        'path': os.path.abspath('.'),
        'train': 'datasets/train',
        'val': 'datasets/val',
        'nc': 3,  # Number of classes
        'names': ['Background', 'WLAN', 'Bluetooth']
    }
    
    with open('dataset.yaml', 'w') as f:
        yaml.dump(dataset_config, f, default_flow_style=False)
    
    print("Created dataset.yaml configuration")
    return 'dataset.yaml'

def save_training_results(results, training_dir):
    """Save training results to a comprehensive document"""
    timestamp = datetime.now().strftime('%Y-%m-%d_%H-%M-%S')
    
    # Create results document
    results_doc = f"training_results_{timestamp}.md"
    results_path = os.path.join(training_dir, results_doc)
    
    # Get training metrics from results.csv if it exists
    csv_path = os.path.join(training_dir, 'results.csv')
    metrics_data = None
    if os.path.exists(csv_path):
        try:
            metrics_data = pd.read_csv(csv_path)
            print(f"Loaded training metrics from {csv_path}")
        except Exception as e:
            print(f"Could not load metrics CSV: {e}")
    
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
        f.write(f"- **Training Date**: {datetime.now().strftime('%Y-%m-%d %H:%M:%S')}\n\n")
        
        # Model files
        f.write("## Model Files\n")
        f.write(f"- **Best Model**: {training_dir}/weights/best.pt\n")
        f.write(f"- **Last Model**: {training_dir}/weights/last.pt\n")
        f.write(f"- **Training Logs**: {training_dir}/results.csv\n")
        f.write(f"- **Training Plots**: {training_dir}/labels.jpg, train_batch*.jpg\n\n")
        
        # Training metrics
        if metrics_data is not None:
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
            
            # Training progress
            f.write("### Training Progress\n")
            f.write("| Epoch | Train Box Loss | Train Cls Loss | Val Box Loss | Val Cls Loss | mAP@50 |\n")
            f.write("|-------|----------------|----------------|--------------|--------------|--------|\n")
            
            for _, row in metrics_data.iterrows():
                f.write(f"| {int(row['epoch'])} | {row['train/box_loss']:.4f} | {row['train/cls_loss']:.4f} | {row['val/box_loss']:.4f} | {row['val/cls_loss']:.4f} | {row['metrics/mAP50(B)']:.4f} |\n")
            
            f.write("\n")
            
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
            
            # Analysis
            f.write("### Training Analysis\n")
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
        
        f.write("\n## Next Steps\n")
        f.write("1. Test the model using: `python3 test_model.py`\n")
        f.write("2. Analyze test results in `yolo_testing_results/` folder\n")
        f.write("3. If performance is poor, consider:\n")
        f.write("   - More training epochs\n")
        f.write("   - Data augmentation\n")
        f.write("   - Different model architecture\n")
        f.write("   - Better data preprocessing\n")
    
    print(f"Training results saved to: {results_path}")
    return results_path

# def create_confusion_matrix_during_training(model, training_dir):
#     """Create confusion matrix by running model on validation data during training"""
#     try:
#         print("Creating confusion matrix from validation data...")
#         
#         # Get validation images
#         val_images = list(Path('datasets/val/images').glob('*.png'))
#         if len(val_images) == 0:
#             print("No validation images found for confusion matrix")
#             return None
#         
#         # Sample some validation images (not all to avoid memory issues)
#         sample_size = min(100, len(val_images))
#         val_images = random.sample(val_images, sample_size)
#         
#         all_true_labels = []
#         all_pred_labels = []
#         
#         for img_path in val_images:
#             # Get predictions
#             results = model(str(img_path), conf=0.5)
#             result = results[0]
#             
#             # Collect predictions
#             if result.boxes is not None and len(result.boxes) > 0:
#                 for box in result.boxes:
#                     pred_class = int(box.cls.item())
#                     all_pred_labels.append(pred_class)
#             
#             # Collect ground truth labels
#             label_path = img_path.with_suffix('.txt')
#             if label_path.exists():
#                 with open(label_path, 'r') as f:
#                     for line in f:
#                         parts = line.strip().split()
#                         if len(parts) >= 5:
#                             true_class = int(parts[0])
#                             all_true_labels.append(true_class)
#         
#         if len(all_true_labels) == 0 or len(all_pred_labels) == 0:
#             print("No labels found for confusion matrix")
#             return None
#         
#         # Create confusion matrix
#         class_names = ['Background', 'WLAN', 'Bluetooth', 'BLE']
#         cm = confusion_matrix(all_true_labels, all_pred_labels)
#         
#         # Create plot
#         plt.figure(figsize=(10, 8))
#         sns.heatmap(cm, annot=True, fmt='d', cmap='Blues', 
#                    xticklabels=class_names, yticklabels=class_names)
#         plt.title('Confusion Matrix - Training Validation')
#         plt.xlabel('Predicted Class')
#         plt.ylabel('True Class')
#         
#         # Save confusion matrix
#         cm_path = os.path.join(training_dir, 'confusion_matrix_training.png')
#         plt.savefig(cm_path, dpi=300, bbox_inches='tight')
#         plt.close()
#         
#         print(f"Confusion matrix saved to: {cm_path}")
#         print(f"Matrix created with {len(all_true_labels)} ground truth and {len(all_pred_labels)} predictions")
#         return cm_path
#         
#     except Exception as e:
#         print(f"Could not create confusion matrix during training: {e}")
#         return None

def check_cuda_usage():
    """Check current CUDA usage and memory"""
    if torch.cuda.is_available():
        allocated = torch.cuda.memory_allocated(0) / 1024**3
        cached = torch.cuda.memory_reserved(0) / 1024**3
        total = torch.cuda.get_device_properties(0).total_memory / 1024**3
        print(f"GPU Memory: {allocated:.2f}GB allocated, {cached:.2f}GB cached, {total:.1f}GB total")
        return True
    return False

def train_model():
    """Train the YOLO model"""
    print("=== Training YOLO Model ===")
    
    # Check CUDA availability
    if torch.cuda.is_available():
        device = 'cuda'
        print(f"Using GPU: {torch.cuda.get_device_name(0)}")
    else:
        device = 'cpu'
        print("Using CPU")
    
    # Setup dataset
    setup_dataset()
    
    # Create dataset config
    dataset_yaml = create_dataset_config()
    
    # Load YOLO model (nano version for faster training)
    print("Loading YOLO model...")
    model = YOLO('yolov8n.pt')
    
    # Train the model
    print("Starting training...")
    results = model.train(
        data=dataset_yaml,
        epochs=50,
        imgsz=1280,  # Higher resolution to preserve signal details
        batch=8,     # Reduced batch size for higher resolution
        device=device,
        # Parameters to help with class imbalance
        cls=0.5,  # Class loss weight (higher = more focus on classification)
        box=7.5,  # Box loss weight
        dfl=1.5,  # Distribution focal loss weight
        # Data augmentation to help with minority classes
        augment=True,
        mosaic=1.0,
        mixup=0.1,
        copy_paste=0.1,
        project='yolo_training3',
        name='rf_detection',
        save=True,
        plots=True,
        verbose=True
    )
    
    print("Training completed!")
    
    # Create confusion matrix during training (commented out for speed)
    training_dir = 'yolo_training3/rf_detection'
    # cm_path = create_confusion_matrix_during_training(model, training_dir)
    
    # Save comprehensive training results
    results_doc_path = save_training_results(results, training_dir)
    
    return model, results_doc_path

def main():
    """Main function"""
    # Set CUDA environment before any PyTorch operations
    os.environ['CUDA_VISIBLE_DEVICES'] = '0'
    
    print("=== YOLO RF Signal Detection Training ===")
    
    # Check if model already exists
    if os.path.exists('yolo_training3/rf_detection/weights/best.pt'):
        print("Found existing trained model!")
        print("Model location: yolo_training3/rf_detection/weights/best.pt")
        
        # Check if we want to resume training or create results document
        response = input("Do you want to resume training? (y/n): ").lower().strip()
        if response == 'y':
            print("Resuming training from existing model...")
            # Force CUDA environment before loading model
            os.environ['CUDA_VISIBLE_DEVICES'] = '0'
            model = YOLO('yolo_training3/rf_detection/weights/best.pt')
            
            # Check device
            if torch.cuda.is_available():
                device = 'cuda'
                print(f"Resuming with GPU: {torch.cuda.get_device_name(0)}")
            else:
                device = 'cpu'
                print("Resuming with CPU")
            
            # Continue training
            results = model.train(
                data='dataset.yaml',
                epochs=50,
                imgsz=1280,  # Higher resolution to preserve signal details
                batch=8,     # Reduced batch size for higher resolution
                device=device,
                # Parameters to help with class imbalance
                cls=0.5,  # Class loss weight (higher = more focus on classification)
                box=7.5,  # Box loss weight
                dfl=1.5,  # Distribution focal loss weight
                # Data augmentation to help with minority classes
                augment=True,
                mosaic=1.0,
                mixup=0.1,
                copy_paste=0.1,
                project='yolo_training3',
                name='rf_detection',
                save=True,
                plots=True,
                verbose=True,
                resume=True  # Resume from existing model
            )
            
            # Save results for resumed training
            training_dir = 'yolo_training3/rf_detection'
            results_doc_path = save_training_results(results, training_dir)
            print("Resumed training completed!")
            return
        else:
            print("Skipping training. You can test the existing model with: python3 test_model.py")
            return
    
    # Train new model
    model, results_doc_path = train_model()
    
    print("\n=== Training Complete ===")
    print("Model saved to: yolo_training3/rf_detection/weights/best.pt")
    print("Training plots saved to: yolo_training3/rf_detection/")
    print(f"Training results document: {results_doc_path}")

if __name__ == "__main__":
    main()
