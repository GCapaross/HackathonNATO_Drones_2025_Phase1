"""
Data Loader for NATO Hackathon CNN Training
Loads spectrogram images and YOLO labels for training
"""

import os
import glob
import numpy as np
import torch
from torch.utils.data import Dataset, DataLoader
from PIL import Image
import torchvision.transforms as transforms
from sklearn.model_selection import train_test_split
import json

class SpectrogramDataset(Dataset):
    """
    Custom dataset for spectrogram images and YOLO labels
    """
    
    def __init__(self, image_paths, label_paths, transform=None, task='classification'):
        """
        Args:
            image_paths: List of paths to spectrogram images
            label_paths: List of paths to YOLO label files
            transform: Image transformations
            task: 'classification' or 'detection'
        """
        self.image_paths = image_paths
        self.label_paths = label_paths
        self.transform = transform
        self.task = task
        
        # Class mapping
        self.class_names = ['Background', 'WLAN', 'Bluetooth', 'BLE']
        self.num_classes = len(self.class_names)
        
    def __len__(self):
        return len(self.image_paths)
    
    def __getitem__(self, idx):
        # Load image
        image_path = self.image_paths[idx]
        image = Image.open(image_path).convert('RGB')
        
        if self.transform:
            image = self.transform(image)
        
        # Load labels
        label_path = self.label_paths[idx]
        
        # Extract filename for frequency info
        filename = os.path.basename(image_path)
        labels = self.load_yolo_labels(label_path)
        
        if self.task == 'classification':
            # For classification: return image, single class label, and filename
            class_label = self.extract_class_labels(labels)
            return image, torch.tensor(class_label, dtype=torch.long), filename
        
        elif self.task == 'detection':
            # For detection: return image, bounding boxes, and filename
            return image, labels, filename
    
    def load_yolo_labels(self, label_path):
        """Load YOLO format labels from text file"""
        labels = []
        if os.path.exists(label_path):
            with open(label_path, 'r') as f:
                for line in f:
                    line = line.strip()
                    if line:
                        parts = line.split()
                        if len(parts) >= 5:
                            class_id = int(parts[0])
                            x_center = float(parts[1])
                            y_center = float(parts[2])
                            width = float(parts[3])
                            height = float(parts[4])
                            labels.append([class_id, x_center, y_center, width, height])
        return labels
    
    def extract_class_labels(self, labels):
        """Extract primary class label for classification task"""
        if not labels:
            return 0  # Background if no labels
        
        # Get class IDs and their counts
        class_ids = [label[0] for label in labels]
        from collections import Counter
        class_counts = Counter(class_ids)
        
        # Strategy: If there are any non-background signals, prioritize them
        # Otherwise, use the most frequent class
        non_background_classes = [cid for cid in class_counts.keys() if cid != 0]
        
        if non_background_classes:
            # If there are non-background signals, use the most frequent one
            most_common = Counter(class_ids).most_common(1)[0][0]
            # But if background is more frequent, still prefer non-background
            if most_common == 0 and len(non_background_classes) > 0:
                # Get the most frequent non-background class
                non_bg_counts = {k: v for k, v in class_counts.items() if k != 0}
                most_common = max(non_bg_counts, key=non_bg_counts.get)
        else:
            # Only background signals
            most_common = 0
        
        # Debug: print class distribution for first few samples
        if hasattr(self, '_debug_count'):
            self._debug_count += 1
        else:
            self._debug_count = 1
            
        if self._debug_count <= 5:  # Print first 5 samples
            print(f"Sample {self._debug_count}: Classes {class_ids} -> Primary: {most_common} (Counts: {dict(class_counts)})")
        
        return most_common

def create_data_loaders(data_dir, batch_size=32, test_size=0.2, task='classification'):
    """
    Create training and validation data loaders
    
    Args:
        data_dir: Path to spectrogram_training_data_20220711 directory
        batch_size: Batch size for training
        test_size: Fraction of data to use for validation
        task: 'classification' or 'detection'
    
    Returns:
        train_loader, val_loader, class_names
    """
    
    # Define image transforms
    if task == 'classification':
        transform = transforms.Compose([
            transforms.Resize((224, 224)),  # Resize for CNN
            transforms.ToTensor(),
            transforms.Normalize(mean=[0.485, 0.456, 0.406], 
                               std=[0.229, 0.224, 0.225])  # ImageNet normalization
        ])
    else:  # detection
        transform = transforms.Compose([
            transforms.Resize((416, 416)),  # YOLO input size
            transforms.ToTensor()
        ])
    
    # Load image and label paths
    results_dir = os.path.join(data_dir, 'results')
    
    # Get all PNG files (excluding marked ones)
    image_files = glob.glob(os.path.join(results_dir, '*.png'))
    image_files = [f for f in image_files if 'marked' not in f]
    
    # Get corresponding label files
    label_files = []
    for img_path in image_files:
        label_path = img_path.replace('.png', '.txt')
        if os.path.exists(label_path):
            label_files.append(label_path)
        else:
            label_files.append('')  # Empty label if no file
    
    # Filter out images without labels
    valid_pairs = [(img, lbl) for img, lbl in zip(image_files, label_files) if lbl]
    image_files, label_files = zip(*valid_pairs)
    
    print(f"Found {len(image_files)} valid image-label pairs")
    
    # Split into train and validation
    train_images, val_images, train_labels, val_labels = train_test_split(
        image_files, label_files, test_size=test_size, random_state=42
    )
    
    # Create datasets
    train_dataset = SpectrogramDataset(
        train_images, train_labels, transform=transform, task=task
    )
    val_dataset = SpectrogramDataset(
        val_images, val_labels, transform=transform, task=task
    )
    
    # Create data loaders
    train_loader = DataLoader(
        train_dataset, batch_size=batch_size, shuffle=True, num_workers=4
    )
    val_loader = DataLoader(
        val_dataset, batch_size=batch_size, shuffle=False, num_workers=4
    )
    
    return train_loader, val_loader, train_dataset.class_names

def analyze_dataset(data_dir):
    """Analyze the dataset and print statistics"""
    results_dir = os.path.join(data_dir, 'spectrogram_training_data_20220711', 'results')
    
    # Count files
    image_files = glob.glob(os.path.join(results_dir, '*.png'))
    image_files = [f for f in image_files if 'marked' not in f]
    
    label_files = []
    for img_path in image_files:
        label_path = img_path.replace('.png', '.txt')
        if os.path.exists(label_path):
            label_files.append(label_path)
    
    print(f"Dataset Analysis:")
    print(f"  Total images: {len(image_files)}")
    print(f"  Images with labels: {len(label_files)}")
    print(f"  Label coverage: {len(label_files)/len(image_files)*100:.1f}%")
    
    # Analyze class distribution
    class_counts = {'Background': 0, 'WLAN': 0, 'Bluetooth': 0, 'BLE': 0}
    
    for label_file in label_files[:1000]:  # Sample first 1000 files
        with open(label_file, 'r') as f:
            for line in f:
                line = line.strip()
                if line:
                    class_id = int(line.split()[0])
                    if class_id == 0:
                        class_counts['Background'] += 1
                    elif class_id == 1:
                        class_counts['WLAN'] += 1
                    elif class_id == 2:
                        class_counts['Bluetooth'] += 1
                    elif class_id == 3:
                        class_counts['BLE'] += 1
    
    print(f"  Class distribution (sample):")
    for class_name, count in class_counts.items():
        print(f"    {class_name}: {count}")
    
    return len(image_files), len(label_files)

if __name__ == "__main__":
    # Test the data loader
    data_dir = "/home/gabriel/Desktop/HackathonNATO_Drones_2025"
    
    print("Analyzing dataset...")
    total_images, labeled_images = analyze_dataset(data_dir)
    
    print("\nCreating data loaders...")
    train_loader, val_loader, class_names = create_data_loaders(
        data_dir, batch_size=16, test_size=0.2, task='classification'
    )
    
    print(f"Training batches: {len(train_loader)}")
    print(f"Validation batches: {len(val_loader)}")
    print(f"Classes: {class_names}")
    
    # Test loading a batch
    print("\nTesting data loading...")
    for images, labels in train_loader:
        print(f"Batch shape: {images.shape}")
        print(f"Labels shape: {labels.shape}")
        print(f"Sample labels: {labels[:5]}")
        break
