#!/usr/bin/env python3
"""
YOLO training script for RF spectrogram detection.
This script trains a YOLO model to detect and classify RF frames in spectrograms.
"""

import os
import sys
from pathlib import Path
import yaml

def train_yolo_model():
    """Train YOLO model on the RF spectrogram dataset."""
    
    # Check if ultralytics is installed
    try:
        from ultralytics import YOLO
    except ImportError:
        print("Error: ultralytics package not found.")
        print("Please install it with: pip install ultralytics")
        return
    
    # Define paths
    base_path = Path(__file__).parent
    dataset_yaml = base_path / "datasets" / "dataset.yaml"
    
    # Check if dataset exists
    if not dataset_yaml.exists():
        print("Error: dataset.yaml not found.")
        print("Please run setup_yolo_dataset.py first to create the dataset.")
        return
    
    # Load dataset configuration
    with open(dataset_yaml, 'r') as f:
        dataset_config = yaml.safe_load(f)
    
    print("Dataset configuration:")
    print(f"  Path: {dataset_config['path']}")
    print(f"  Training: {dataset_config['train']}")
    print(f"  Validation: {dataset_config['val']}")
    print(f"  Classes: {dataset_config['names']}")
    
    # Initialize YOLO model
    print("\nInitializing YOLO model...")
    model = YOLO('yolov8n.pt')  # Use nano model for faster training
    
    # Training parameters
    training_args = {
        'data': str(dataset_yaml),
        'epochs': 100,
        'imgsz': 640,
        'batch': 16,
        'device': 'cuda',  # Use GPU for training
        'project': 'yolo_training',
        'name': 'rf_spectrogram_detection',
        'save': True,
        'save_period': 10,
        'patience': 20,
        'lr0': 0.01,
        'lrf': 0.1,
        'momentum': 0.937,
        'weight_decay': 0.0005,
        'warmup_epochs': 3,
        'warmup_momentum': 0.8,
        'warmup_bias_lr': 0.1,
        'box': 7.5,
        'cls': 0.5,
        'dfl': 1.5,
        'pose': 12.0,
        'kobj': 1.0,
        'label_smoothing': 0.0,
        'nbs': 64,
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
        'mixup': 0.0,
        'copy_paste': 0.0
    }
    
    print("Starting YOLO training...")
    print("Training parameters:")
    for key, value in training_args.items():
        if key not in ['data', 'project', 'name']:
            print(f"  {key}: {value}")
    
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
    print("\nValidating trained model...")
    try:
        validation_results = model.val()
        print("Validation completed!")
        
        # Print validation metrics
        if hasattr(validation_results, 'box'):
            print("\nValidation metrics:")
            print(f"  mAP50: {validation_results.box.map50:.3f}")
            print(f"  mAP50-95: {validation_results.box.map:.3f}")
            
    except Exception as e:
        print(f"Error during validation: {e}")

def test_trained_model():
    """Test the trained model on sample images."""
    
    try:
        from ultralytics import YOLO
    except ImportError:
        print("Error: ultralytics package not found.")
        return
    
    # Load the best trained model
    model_path = Path(__file__).parent / "yolo_training" / "rf_spectrogram_detection" / "weights" / "best.pt"
    
    if not model_path.exists():
        print("Error: Trained model not found.")
        print("Please run train_yolo_model() first.")
        return
    
    print(f"Loading trained model from: {model_path}")
    model = YOLO(str(model_path))
    
    # Test on a few sample images
    base_path = Path(__file__).parent
    test_images = list((base_path / "datasets" / "images" / "val").glob("*.png"))[:5]
    
    if not test_images:
        print("No test images found.")
        return
    
    print(f"Testing on {len(test_images)} sample images...")
    
    for img_path in test_images:
        print(f"\nTesting: {img_path.name}")
        
        # Run inference
        results = model(str(img_path))
        
        # Print results
        for result in results:
            if result.boxes is not None:
                boxes = result.boxes
                print(f"  Detected {len(boxes)} objects:")
                for i, box in enumerate(boxes):
                    class_id = int(box.cls[0])
                    confidence = float(box.conf[0])
                    class_names = ['WLAN', 'collision', 'bluetooth']
                    print(f"    {i+1}. {class_names[class_id]} (confidence: {confidence:.3f})")
            else:
                print("  No objects detected")

if __name__ == "__main__":
    import argparse
    
    parser = argparse.ArgumentParser(description="YOLO training for RF spectrogram detection")
    parser.add_argument("--train", action="store_true", help="Train the YOLO model")
    parser.add_argument("--test", action="store_true", help="Test the trained model")
    
    args = parser.parse_args()
    
    if args.train:
        train_yolo_model()
    elif args.test:
        test_trained_model()
    else:
        print("Please specify --train or --test")
        print("Usage:")
        print("  python train_yolo_model.py --train")
        print("  python train_yolo_model.py --test")
