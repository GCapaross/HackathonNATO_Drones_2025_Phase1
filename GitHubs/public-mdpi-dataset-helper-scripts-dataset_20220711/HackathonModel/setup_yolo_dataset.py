#!/usr/bin/env python3
"""
Setup script to organize the RF spectrogram dataset for YOLO training.
This script copies images and labels from the results folder into a proper YOLO dataset structure.
"""

import os
import shutil
from pathlib import Path
import random

def setup_yolo_dataset():
    """Organize the spectrogram dataset for YOLO training."""
    
    # Define paths
    base_path = Path(__file__).parent
    results_path = base_path.parent / "spectrogram_training_data_20220711" / "results"
    dataset_path = base_path / "datasets"
    
    # Create dataset structure
    train_images = dataset_path / "images" / "train"
    train_labels = dataset_path / "labels" / "train"
    val_images = dataset_path / "images" / "val"
    val_labels = dataset_path / "labels" / "val"
    
    # Create directories
    for path in [train_images, train_labels, val_images, val_labels]:
        path.mkdir(parents=True, exist_ok=True)
    
    print("Created dataset directory structure")
    
    # Get all image files from results
    image_files = list(results_path.glob("result_frame_*_bw_*.png"))
    print(f"Found {len(image_files)} spectrogram images")
    
    # Use the existing train/val split from the dataset
    train_list_path = base_path.parent / "spectrogram_training_data_20220711" / "list_images_train.txt"
    val_list_path = base_path.parent / "spectrogram_training_data_20220711" / "list_images_valid.txt"
    
    # Read train/val file lists
    train_files = []
    val_files = []
    
    if train_list_path.exists():
        with open(train_list_path, 'r') as f:
            train_files = [Path(line.strip()).name for line in f if line.strip()]
        print(f"Loaded {len(train_files)} training files from list")
    
    if val_list_path.exists():
        with open(val_list_path, 'r') as f:
            val_files = [Path(line.strip()).name for line in f if line.strip()]
        print(f"Loaded {len(val_files)} validation files from list")
    
    # Copy training files
    train_count = 0
    for img_file in image_files:
        if img_file.name in train_files:
            # Copy image
            shutil.copy2(img_file, train_images / img_file.name)
            
            # Copy corresponding label
            label_file = img_file.with_suffix('.txt')
            if label_file.exists():
                shutil.copy2(label_file, train_labels / label_file.name)
                train_count += 1
    
    print(f"Copied {train_count} training image-label pairs")
    
    # Copy validation files
    val_count = 0
    for img_file in image_files:
        if img_file.name in val_files:
            # Copy image
            shutil.copy2(img_file, val_images / img_file.name)
            
            # Copy corresponding label
            label_file = img_file.with_suffix('.txt')
            if label_file.exists():
                shutil.copy2(label_file, val_labels / label_file.name)
                val_count += 1
    
    print(f"Copied {val_count} validation image-label pairs")
    
    # Create dataset.yaml
    dataset_yaml_content = f"""# RF Spectrogram Dataset for YOLO Training
# Generated from the MDPI spectrogram dataset

path: {dataset_path.absolute()}
train: images/train
val: images/val

# Number of classes
nc: 3

# Class names
names: 
  0: WLAN
  1: collision  
  2: bluetooth

# Dataset info
# Total images: {train_count + val_count}
# Training: {train_count}
# Validation: {val_count}
# Classes: Wi-Fi frames, collisions, Bluetooth frames
"""
    
    with open(dataset_path / "dataset.yaml", 'w') as f:
        f.write(dataset_yaml_content)
    
    print(f"Created dataset.yaml configuration")
    print(f"Dataset ready for YOLO training!")
    print(f"   - Training: {train_count} images")
    print(f"   - Validation: {val_count} images")
    print(f"   - Classes: WLAN, collision, bluetooth")

if __name__ == "__main__":
    setup_yolo_dataset()
