"""
YOLO Data Loader for RF Signal Detection
Handles spectrogram images with YOLO format bounding boxes
"""

import os
import glob
import numpy as np
import torch
from torch.utils.data import Dataset, DataLoader
from PIL import Image
import torchvision.transforms as transforms
from sklearn.model_selection import train_test_split
import cv2
import yaml
from pathlib import Path

class SpectrogramYOLODataset(Dataset):
    """
    YOLO Dataset for spectrogram signal detection
    Handles multiple bounding boxes per image
    """
    
    def __init__(self, image_paths, label_paths, transform=None, img_size=640, preserve_aspect_ratio=True):
        """
        Args:
            image_paths: List of paths to spectrogram images
            label_paths: List of paths to YOLO label files
            transform: Image transformations
            img_size: Target image size for YOLO (default 640)
            preserve_aspect_ratio: Whether to preserve aspect ratio when resizing
        """
        self.image_paths = image_paths
        self.label_paths = label_paths
        self.transform = transform
        self.img_size = img_size
        self.preserve_aspect_ratio = preserve_aspect_ratio
        
        # Class mapping (same as original dataset)
        self.class_names = ['Background', 'WLAN', 'Bluetooth', 'BLE']
        self.num_classes = len(self.class_names)
        
    def __len__(self):
        return len(self.image_paths)
    
    def __getitem__(self, idx):
        # Load image
        image_path = self.image_paths[idx]
        image = Image.open(image_path).convert('RGB')
        original_size = image.size  # (width, height)
        
            # Removed print to speed up training
        
        # Load YOLO labels
        label_path = self.label_paths[idx]
        labels = self.load_yolo_labels(label_path)
        
        # Convert to numpy for processing
        image_np = np.array(image)
        
        # Handle different resolutions dynamically
        if self.preserve_aspect_ratio:
            # Resize while preserving aspect ratio
            image_resized = self._resize_preserve_aspect_ratio(image_np, self.img_size)
        else:
            # Simple resize (may distort aspect ratio)
            image_resized = cv2.resize(image_np, (self.img_size, self.img_size))
        
        # Skip transforms for now to avoid compatibility issues
        # TODO: Implement proper YOLO-compatible transforms later
        image_np = image_resized
        
        # Convert to tensor
        image_tensor = torch.from_numpy(image_np).permute(2, 0, 1).float() / 255.0
        
        # Convert labels to tensor format
        if len(labels) > 0:
            # YOLO format: [class_id, x_center, y_center, width, height]
            labels_tensor = torch.tensor(labels, dtype=torch.float32)
        else:
            # Empty tensor for images with no labels
            labels_tensor = torch.zeros((0, 5), dtype=torch.float32)
        
        return {
            'image': image_tensor,
            'labels': labels_tensor,
            'image_path': image_path,
            'original_size': original_size
        }
    
    def _resize_preserve_aspect_ratio(self, image, target_size):
        """
        Resize image while preserving aspect ratio
        """
        h, w = image.shape[:2]
        
        # Calculate scaling factor
        scale = min(target_size / w, target_size / h)
        new_w = int(w * scale)
        new_h = int(h * scale)
        
        # Resize image
        resized = cv2.resize(image, (new_w, new_h))
        
        # Create square image with padding
        square_image = np.zeros((target_size, target_size, 3), dtype=np.uint8)
        
        # Calculate padding
        pad_x = (target_size - new_w) // 2
        pad_y = (target_size - new_h) // 2
        
        # Place resized image in center
        square_image[pad_y:pad_y+new_h, pad_x:pad_x+new_w] = resized
        
        return square_image
    
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

def create_yolo_data_loaders(data_dir, batch_size=16, test_size=0.2, img_size=640):
    """
    Create training and validation data loaders for YOLO
    """
    # Skip transforms for now to avoid compatibility issues
    # TODO: Implement proper YOLO-compatible transforms later
    train_transform = None
    val_transform = None
    
    # Load image and label paths
    results_dir = os.path.join(data_dir, 'results')
    
    # Get all PNG files (excluding marked ones)
    image_files = glob.glob(os.path.join(results_dir, '*.png'))
    image_files = [f for f in image_files if 'marked' not in f and f.endswith('.png')]
    
    # Create corresponding label files
    label_files = []
    for img_path in image_files:
        label_path = img_path.replace('.png', '.txt')
        if os.path.exists(label_path):
            label_files.append(label_path)
        else:
            print(f"Warning: No label file found for {img_path}")
    
    # Filter to only include pairs that exist
    valid_pairs = [(img, lbl) for img, lbl in zip(image_files, label_files) if os.path.exists(lbl)]
    image_files, label_files = zip(*valid_pairs) if valid_pairs else ([], [])
    
    print(f"Found {len(image_files)} valid image-label pairs")
    
    # Split into train/validation
    train_images, val_images, train_labels, val_labels = train_test_split(
        image_files, label_files, test_size=test_size, random_state=42
    )
    
    # Create datasets
    train_dataset = SpectrogramYOLODataset(
        train_images, train_labels, transform=train_transform, img_size=img_size
    )
    val_dataset = SpectrogramYOLODataset(
        val_images, val_labels, transform=val_transform, img_size=img_size
    )
    
    # Create data loaders
    train_loader = DataLoader(
        train_dataset, batch_size=batch_size, shuffle=True, 
        collate_fn=yolo_collate_fn, num_workers=4
    )
    val_loader = DataLoader(
        val_dataset, batch_size=batch_size, shuffle=False,
        collate_fn=yolo_collate_fn, num_workers=4
    )
    
    return train_loader, val_loader, train_dataset.class_names

def yolo_collate_fn(batch):
    """
    Custom collate function for YOLO batches
    Handles variable number of labels per image
    """
    images = torch.stack([item['image'] for item in batch])
    labels = [item['labels'] for item in batch]
    image_paths = [item['image_path'] for item in batch]
    original_sizes = [item['original_size'] for item in batch]
    
    return {
        'images': images,
        'labels': labels,
        'image_paths': image_paths,
        'original_sizes': original_sizes
    }

def analyze_dataset(data_dir):
    """
    Analyze the dataset to understand class distribution and label statistics
    """
    results_dir = os.path.join(data_dir, 'results')
    image_files = glob.glob(os.path.join(results_dir, '*.png'))
    image_files = [f for f in image_files if 'marked' not in f]
    
    class_counts = {0: 0, 1: 0, 2: 0, 3: 0}
    total_labels = 0
    images_with_labels = 0
    label_counts_per_image = []
    
    for img_path in image_files:
        label_path = img_path.replace('.png', '.txt')
        if os.path.exists(label_path):
            with open(label_path, 'r') as f:
                labels = []
                for line in f:
                    line = line.strip()
                    if line:
                        parts = line.split()
                        if len(parts) >= 5:
                            class_id = int(parts[0])
                            class_counts[class_id] += 1
                            labels.append(class_id)
                
                if labels:
                    images_with_labels += 1
                    label_counts_per_image.append(len(labels))
                    total_labels += len(labels)
    
    print("=== Dataset Analysis ===")
    print(f"Total images: {len(image_files)}")
    print(f"Images with labels: {images_with_labels}")
    print(f"Label coverage: {images_with_labels/len(image_files)*100:.1f}%")
    print(f"Total labels: {total_labels}")
    print(f"Average labels per image: {np.mean(label_counts_per_image):.2f}")
    print(f"Max labels in single image: {max(label_counts_per_image) if label_counts_per_image else 0}")
    print(f"Min labels in single image: {min(label_counts_per_image) if label_counts_per_image else 0}")
    print("\nClass distribution:")
    for class_id, count in class_counts.items():
        class_name = ['Background', 'WLAN', 'Bluetooth', 'BLE'][class_id]
        print(f"  {class_name}: {count}")
    
    return {
        'total_images': len(image_files),
        'images_with_labels': images_with_labels,
        'total_labels': total_labels,
        'class_counts': class_counts,
        'label_counts_per_image': label_counts_per_image
    }

if __name__ == "__main__":
    # Test the data loader
    data_dir = "../spectrogram_training_data_20220711"
    
    print("Analyzing dataset...")
    stats = analyze_dataset(data_dir)
    
    print("\nCreating data loaders...")
    train_loader, val_loader, class_names = create_yolo_data_loaders(
        data_dir, batch_size=8, test_size=0.2, img_size=640
    )
    
    print(f"Classes: {class_names}")
    print(f"Training batches: {len(train_loader)}")
    print(f"Validation batches: {len(val_loader)}")
    
    # Test a batch
    print("\nTesting data loader...")
    for batch in train_loader:
        print(f"Batch images shape: {batch['images'].shape}")
        print(f"Number of label lists: {len(batch['labels'])}")
        print(f"Sample label shapes: {[l.shape for l in batch['labels'][:3]]}")
        break
