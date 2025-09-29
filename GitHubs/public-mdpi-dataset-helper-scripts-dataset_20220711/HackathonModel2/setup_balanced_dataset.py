#!/usr/bin/env python3
"""
Setup script to create a balanced dataset (33% each class) for YOLO training.
This script creates a balanced dataset by oversampling minority classes.
"""

import os
import shutil
from pathlib import Path
import random
from collections import Counter

def setup_balanced_dataset():
    """Create a balanced dataset with 33% each class."""
    
    # Define paths
    base_path = Path(__file__).parent
    results_path = base_path.parent / "spectrogram_training_data_20220711" / "results"
    dataset_path = base_path / "datasets_balanced"
    
    # Create dataset structure
    train_images = dataset_path / "images" / "train"
    train_labels = dataset_path / "labels" / "train"
    val_images = dataset_path / "images" / "val"
    val_labels = dataset_path / "labels" / "val"
    
    # Create directories
    for path in [train_images, train_labels, val_images, val_labels]:
        path.mkdir(parents=True, exist_ok=True)
    
    print("Created balanced dataset directory structure")
    
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
    
    # Analyze class distribution in training data
    print("\nAnalyzing class distribution...")
    class_counts = Counter()
    class_files = {0: [], 1: [], 2: []}  # WLAN, collision, bluetooth
    
    for img_file in image_files:
        if img_file.name in train_files:
            label_file = img_file.with_suffix('.txt')
            if label_file.exists():
                with open(label_file, 'r') as f:
                    for line in f:
                        if line.strip():
                            class_id = int(line.strip().split()[0])
                            class_counts[class_id] += 1
                            class_files[class_id].append((img_file, label_file))
    
    print("Original class distribution:")
    class_names = ['WLAN', 'collision', 'bluetooth']
    for class_id, count in class_counts.items():
        print(f"  {class_names[class_id]}: {count} objects")
    
    # Find target count (use the smallest class as base)
    min_count = min(class_counts.values())
    target_count = min_count
    print(f"\nTarget count per class: {target_count}")
    
    # Create balanced dataset
    print("\nCreating balanced dataset...")
    
    # Copy validation files (keep original distribution)
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
    
    # Create balanced training set
    train_count = 0
    for class_id in [0, 1, 2]:  # WLAN, collision, bluetooth
        class_name = class_names[class_id]
        current_count = class_counts[class_id]
        
        if current_count >= target_count:
            # Sample target_count files from this class
            selected_files = random.sample(class_files[class_id], target_count)
            print(f"Selected {target_count} {class_name} files (from {current_count} available)")
        else:
            # Use all available files
            selected_files = class_files[class_id]
            print(f"Using all {current_count} {class_name} files (less than target)")
        
        # Copy selected files
        for img_file, label_file in selected_files:
            # Copy image
            shutil.copy2(img_file, train_images / img_file.name)
            
            # Copy corresponding label
            shutil.copy2(label_file, train_labels / label_file.name)
            train_count += 1
    
    print(f"Copied {train_count} training image-label pairs")
    
    # Create dataset.yaml
    dataset_yaml_content = f"""# Balanced RF Spectrogram Dataset for YOLO Training
# Generated with 33% distribution per class

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
# Classes: Balanced distribution (33% each)
"""
    
    with open(dataset_path / "dataset.yaml", 'w') as f:
        f.write(dataset_yaml_content)
    
    print(f"\nCreated dataset.yaml configuration")
    print(f"Balanced dataset ready for YOLO training!")
    print(f"   - Training: {train_count} images")
    print(f"   - Validation: {val_count} images")
    print(f"   - Classes: Balanced distribution")

if __name__ == "__main__":
    setup_balanced_dataset()
