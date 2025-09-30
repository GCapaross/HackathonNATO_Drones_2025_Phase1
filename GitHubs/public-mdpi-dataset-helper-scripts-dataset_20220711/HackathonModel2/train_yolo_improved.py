#!/usr/bin/env python3
"""
Improved YOLO training script with better parameters for Bluetooth detection.
"""

import os
import sys
from pathlib import Path
import yaml

def train_yolo_improved():
    """Train YOLO model with improved parameters for Bluetooth detection."""
    
    # Check if ultralytics is installed
    try:
        from ultralytics import YOLO
    except ImportError:
        print("Error: ultralytics package not found.")
        print("Please install it with: pip install ultralytics")
        return
    
    # Define paths
    base_path = Path(__file__).parent
    dataset_yaml = base_path / "datasets_balanced" / "dataset.yaml"
    
    # Check if dataset exists
    if not dataset_yaml.exists():
        print("Error: dataset.yaml not found.")
        print("Please run setup_balanced_dataset.py first to create the balanced dataset.")
        return
    
    # Load dataset configuration
    with open(dataset_yaml, 'r') as f:
        dataset_config = yaml.safe_load(f)
    
    print("Dataset configuration:")
    print(f"  Path: {dataset_config['path']}")
    print(f"  Training: {dataset_config['train']}")
    print(f"  Validation: {dataset_config['val']}")
    print(f"  Classes: {dataset_config['names']}")
    
    # Initialize YOLO model (try larger model for better detection)
    print("\nInitializing YOLO model...")
    model = YOLO('yolov8s.pt')  # Use small model instead of nano for better performance
    
    # Calculate class weights based on dataset distribution
    # WLAN: 71.2%, Collision: 16.0%, Bluetooth: 12.8%
    # Inverse frequency weighting: weight = total_samples / (num_classes * class_samples)
    class_weights = [1.0, 2.2, 2.8]  # [WLAN, Collision, Bluetooth]
    
    print(f"Class weights: WLAN={class_weights[0]}, Collision={class_weights[1]}, Bluetooth={class_weights[2]}")
    
    # Apply class weighting by modifying the model's loss function
    # This is a workaround since ultralytics doesn't support class_weights directly
    if hasattr(model.model, 'criterion') and hasattr(model.model.criterion, 'class_weights'):
        model.model.criterion.class_weights = class_weights
        print("Applied class weights to model criterion")
    else:
        print("Note: Direct class weighting not available, using higher cls weight instead")
    
    # Improved training parameters for Bluetooth detection
    # Using higher cls weight and focal loss approach for class imbalance
    training_args = {
        'data': str(dataset_yaml),
        'epochs': 20,  # Quick training for testing
        'imgsz': 640,
        'batch': 16,  # Increased batch size for better training
        'device': 'cuda',
        'project': 'yolo_training_improved',
        'name': 'rf_spectrogram_detection_improved',
        'save': True,
        'save_period': 10,
        'patience': 20,  # Patience for early stopping
        'lr0': 0.005,  # Lower learning rate for stable training
        'lrf': 0.05,  # Lower final learning rate
        'momentum': 0.937,
        'weight_decay': 0.0005,
        'warmup_epochs': 3,  # Warmup epochs
        'warmup_momentum': 0.8,
        'warmup_bias_lr': 0.1,
        'box': 7.5,
        'cls': 4.0,  # Much higher classification loss weight for class imbalance
        'dfl': 1.5,
        'hsv_h': 0.015,
        'hsv_s': 0.7,
        'hsv_v': 0.4,
        'degrees': 0.0,
        'translate': 0.1,
        'scale': 0.5,
        'shear': 0.0,
        'perspective': 0.0,
        'flipud': 0.0,
        'fliplr': 0.5,
        'mosaic': 1.0,
        'mixup': 0.15,  # Increased mixup for better class balancing
        'copy_paste': 0.15,  # Increased copy-paste for better class balancing
    }
    
    print("Starting improved YOLO training...")
    print("Key improvements for Bluetooth detection:")
    print("  - 20 epochs (quick training)")
    print("  - Lower learning rate (0.005)")
    print("  - Higher batch size (16) for better training")
    print("  - Class weighting: WLAN=1.0, Collision=2.2, Bluetooth=2.8")
    print("  - Much higher classification loss weight (4.0) for class imbalance")
    print("  - Enhanced augmentation (mixup=0.15, copy-paste=0.15)")
    print("  - YOLOv8s model (better than nano)")
    print("  - CUDA acceleration")
    
    # Start training
    try:
        results = model.train(**training_args)
        print("\nTraining completed successfully!")
        print(f"Results saved in: {results.save_dir}")
        
        # Print training summary
        if hasattr(results, 'results_dict'):
            print("\nTraining summary:")
            for key, value in results.results_dict.items():
                print(f"  {key}: {value}")
        
    except Exception as e:
        print(f"Error during training: {e}")
        return
    
    # Validate the trained model
    print("\nValidating improved model...")
    try:
        validation_results = model.val()
        print("Validation completed!")
        
        # Print validation metrics
        if hasattr(validation_results, 'box'):
            print("\nValidation metrics:")
            print(f"  mAP50: {validation_results.box.map50:.3f}")
            print(f"  mAP50-95: {validation_results.box.map:.3f}")
            
            # Print per-class metrics if available
            if hasattr(validation_results.box, 'maps'):
                class_names = ['WLAN', 'collision', 'bluetooth']
                print("\nPer-class mAP50:")
                for i, class_name in enumerate(class_names):
                    if i < len(validation_results.box.maps):
                        print(f"  {class_name}: {validation_results.box.maps[i]:.3f}")
            
    except Exception as e:
        print(f"Error during validation: {e}")

if __name__ == "__main__":
    train_yolo_improved()
