#!/usr/bin/env python3
"""
Analyze the dataset to understand class distribution and potential issues.
"""

import os
from pathlib import Path
import matplotlib.pyplot as plt
import numpy as np
from collections import Counter

def analyze_dataset_distribution():
    """Analyze the class distribution in the dataset."""
    
    # Define paths
    base_path = Path(__file__).parent
    train_labels = base_path / "datasets" / "labels" / "train"
    val_labels = base_path / "datasets" / "labels" / "val"
    
    if not train_labels.exists():
        print("Dataset not found. Please run setup_yolo_dataset.py first.")
        return
    
    # Class names
    class_names = ['WLAN', 'collision', 'bluetooth']
    
    # Count classes in training data
    train_counts = Counter()
    train_files = list(train_labels.glob("*.txt"))
    
    print(f"Analyzing {len(train_files)} training label files...")
    
    for label_file in train_files:
        with open(label_file, 'r') as f:
            for line in f:
                if line.strip():
                    class_id = int(line.strip().split()[0])
                    train_counts[class_id] += 1
    
    # Count classes in validation data
    val_counts = Counter()
    val_files = list(val_labels.glob("*.txt"))
    
    print(f"Analyzing {len(val_files)} validation label files...")
    
    for label_file in val_files:
        with open(label_file, 'r') as f:
            for line in f:
                if line.strip():
                    class_id = int(line.strip().split()[0])
                    val_counts[class_id] += 1
    
    # Print results
    print("\n=== DATASET ANALYSIS ===")
    print("\nTraining Data:")
    total_train = sum(train_counts.values())
    for class_id, count in train_counts.items():
        percentage = (count / total_train) * 100
        print(f"  {class_names[class_id]}: {count} objects ({percentage:.1f}%)")
    
    print("\nValidation Data:")
    total_val = sum(val_counts.values())
    for class_id, count in val_counts.items():
        percentage = (count / total_val) * 100
        print(f"  {class_names[class_id]}: {count} objects ({percentage:.1f}%)")
    
    # Check for class imbalance
    print("\n=== CLASS IMBALANCE ANALYSIS ===")
    if train_counts[2] < train_counts[0] * 0.1:  # Bluetooth < 10% of WLAN
        print("WARNING: Severe class imbalance detected!")
        print("Bluetooth signals are much less frequent than WLAN signals.")
        print("This can cause the model to focus on the majority class (WLAN).")
    
    # Create visualization
    fig, (ax1, ax2) = plt.subplots(1, 2, figsize=(15, 6))
    
    # Training data pie chart
    train_labels_pie = [class_names[i] for i in train_counts.keys()]
    train_sizes = list(train_counts.values())
    ax1.pie(train_sizes, labels=train_labels_pie, autopct='%1.1f%%', startangle=90)
    ax1.set_title('Training Data Class Distribution')
    
    # Validation data pie chart
    val_labels_pie = [class_names[i] for i in val_counts.keys()]
    val_sizes = list(val_counts.values())
    ax2.pie(val_sizes, labels=val_labels_pie, autopct='%1.1f%%', startangle=90)
    ax2.set_title('Validation Data Class Distribution')
    
    plt.tight_layout()
    plt.savefig('dataset_analysis.png', dpi=150, bbox_inches='tight')
    print(f"\nSaved analysis plot to: dataset_analysis.png")
    
    return train_counts, val_counts

def suggest_improvements(train_counts, val_counts):
    """Suggest improvements based on the analysis."""
    
    class_names = ['WLAN', 'collision', 'bluetooth']
    
    print("\n=== IMPROVEMENT SUGGESTIONS ===")
    
    # Check for class imbalance
    wlan_count = train_counts[0]
    bluetooth_count = train_counts[2]
    
    if bluetooth_count < wlan_count * 0.2:
        print("1. CLASS IMBALANCE ISSUE:")
        print("   - Bluetooth signals are underrepresented")
        print("   - Consider using class weights in training")
        print("   - Or augment Bluetooth samples")
    
    print("\n2. TRAINING PARAMETERS TO ADJUST:")
    print("   - Increase epochs (try 200-300)")
    print("   - Use class weights in loss function")
    print("   - Lower confidence threshold for Bluetooth")
    print("   - Use focal loss for imbalanced classes")
    
    print("\n3. DATA AUGMENTATION:")
    print("   - Generate more Bluetooth samples")
    print("   - Use different channel models for Bluetooth")
    print("   - Create synthetic Bluetooth collisions")
    
    print("\n4. MODEL ARCHITECTURE:")
    print("   - Try larger YOLO model (yolov8s, yolov8m)")
    print("   - Use different anchor sizes")
    print("   - Adjust input image size")

if __name__ == "__main__":
    train_counts, val_counts = analyze_dataset_distribution()
    suggest_improvements(train_counts, val_counts)
