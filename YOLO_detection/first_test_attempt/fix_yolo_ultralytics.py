"""
Fix YOLO Implementation using Ultralytics YOLOv8
This replaces our custom YOLO with a proven implementation
"""

import torch
from ultralytics import YOLO
import os
import yaml
import shutil
from pathlib import Path

def create_yolo_dataset_config():
    """Create YOLO dataset configuration file"""
    dataset_config = {
        'path': '../spectrogram_training_data_20220711',
        'train': 'results',
        'val': 'results',  # We'll split the data
        'nc': 3,  # Number of classes
        'names': ['Background', 'WLAN', 'Bluetooth']
    }
    
    with open('dataset.yaml', 'w') as f:
        yaml.dump(dataset_config, f, default_flow_style=False)
    
    print("Created dataset.yaml configuration")
    return 'dataset.yaml'

def convert_labels_to_yolo_format():
    """Convert our existing labels to proper YOLO format for Ultralytics"""
    results_dir = '../spectrogram_training_data_20220711/results'
    
    # Create train/val directories
    os.makedirs('datasets/train/images', exist_ok=True)
    os.makedirs('datasets/train/labels', exist_ok=True)
    os.makedirs('datasets/val/images', exist_ok=True)
    os.makedirs('datasets/val/labels', exist_ok=True)
    
    # Get all image files
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
    for i, img_path in enumerate(train_files):
        if i % 1000 == 0:
            print(f"Processing training files: {i}/{len(train_files)}")
        
        # Copy image
        dst_img = f'datasets/train/images/{img_path.name}'
        shutil.copy2(img_path, dst_img)
        
        # Copy label (if exists)
        label_path = img_path.with_suffix('.txt')
        if label_path.exists():
            dst_label = f'datasets/train/labels/{label_path.name}'
            shutil.copy2(label_path, dst_label)
    
    # Process validation files
    for i, img_path in enumerate(val_files):
        if i % 1000 == 0:
            print(f"Processing validation files: {i}/{len(val_files)}")
        
        # Copy image
        dst_img = f'datasets/val/images/{img_path.name}'
        shutil.copy2(img_path, dst_img)
        
        # Copy label (if exists)
        label_path = img_path.with_suffix('.txt')
        if label_path.exists():
            dst_label = f'datasets/val/labels/{label_path.name}'
            shutil.copy2(label_path, dst_label)
    
    print(f"Dataset conversion complete:")
    print(f"  Training: {len(train_files)} images")
    print(f"  Validation: {len(val_files)} images")

def train_yolo_ultralytics():
    """Train YOLO using Ultralytics"""
    print("=== Training YOLO with Ultralytics ===")
    
    # Create dataset config
    dataset_yaml = create_yolo_dataset_config()
    
    # Convert dataset
    convert_labels_to_yolo_format()
    
    # Load YOLO model
    model = YOLO('yolov8n.pt')  # nano version for faster training
    
    # Train the model
    results = model.train(
        data=dataset_yaml,
        epochs=50,
        imgsz=640,
        batch=16,
        device='cuda' if torch.cuda.is_available() else 'cpu',
        project='yolo_training',
        name='rf_detection',
        save=True,
        plots=True
    )
    
    print("Training completed!")
    return model

def test_yolo_ultralytics(model_path='yolo_training/rf_detection/weights/best.pt'):
    """Test the trained YOLO model"""
    print("=== Testing YOLO Model ===")
    
    # Load trained model
    model = YOLO(model_path)
    
    # Test on validation set
    results = model.val()
    
    # Test on single image
    test_image = '../spectrogram_training_data_20220711/results/result_frame_138769090412766231_bw_25E+6.png'
    if os.path.exists(test_image):
        results = model(test_image, save=True, project='yolo_testing', name='predictions')
        print(f"Predictions saved to yolo_testing/predictions/")
    
    return model

def main():
    """Main function to fix YOLO implementation"""
    print("=== Fixing YOLO Implementation ===")
    print("Using Ultralytics YOLOv8 instead of custom implementation")
    
    # Check if we should train or test
    if os.path.exists('yolo_training/rf_detection/weights/best.pt'):
        print("Found existing trained model, testing...")
        model = test_yolo_ultralytics()
    else:
        print("No existing model found, training new model...")
        model = train_yolo_ultralytics()
        model = test_yolo_ultralytics()
    
    print("YOLO implementation fixed!")

if __name__ == "__main__":
    main()
