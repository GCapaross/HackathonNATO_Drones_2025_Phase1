#!/usr/bin/env python3
"""
Check the actual balance of the created dataset.
"""

from pathlib import Path
from collections import Counter

def check_dataset_balance():
    """Check the actual class distribution in the balanced dataset."""
    
    base_path = Path(__file__).parent
    train_labels = base_path / "datasets_balanced" / "labels" / "train"
    val_labels = base_path / "datasets_balanced" / "labels" / "val"
    
    if not train_labels.exists():
        print("Balanced dataset not found. Please run setup_balanced_dataset.py first.")
        return
    
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
    print("\n=== BALANCED DATASET ANALYSIS ===")
    print("\nTraining Data:")
    class_names = ['WLAN', 'collision', 'bluetooth']
    total_train = sum(train_counts.values())
    for class_id, count in train_counts.items():
        percentage = (count / total_train) * 100
        print(f"  {class_names[class_id]}: {count} objects ({percentage:.1f}%)")
    
    print("\nValidation Data:")
    total_val = sum(val_counts.values())
    for class_id, count in val_counts.items():
        percentage = (count / total_val) * 100
        print(f"  {class_names[class_id]}: {count} objects ({percentage:.1f}%)")
    
    print(f"\nTotal Training Objects: {total_train}")
    print(f"Total Validation Objects: {total_val}")
    
    # Check if it's actually balanced
    print("\n=== BALANCE CHECK ===")
    if total_train > 0:
        wlan_pct = (train_counts[0] / total_train) * 100
        collision_pct = (train_counts[1] / total_train) * 100
        bluetooth_pct = (train_counts[2] / total_train) * 100
        
        print(f"WLAN: {wlan_pct:.1f}%")
        print(f"Collision: {collision_pct:.1f}%")
        print(f"Bluetooth: {bluetooth_pct:.1f}%")
        
        if 30 <= wlan_pct <= 40 and 30 <= collision_pct <= 40 and 30 <= bluetooth_pct <= 40:
            print("✅ Dataset is reasonably balanced!")
        else:
            print("❌ Dataset is NOT balanced - the script only balanced IMAGES, not OBJECTS")
            print("   Each image can contain multiple objects of different classes")
            print("   The script selected images that contain each class, but didn't balance the objects within")

if __name__ == "__main__":
    check_dataset_balance()
