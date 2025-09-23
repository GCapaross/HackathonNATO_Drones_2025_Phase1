#!/usr/bin/env python3
"""
Custom CNN training script for RF spectrogram classification.
Trains a CNN to classify drone types from spectrogram images.
"""

import torch
import torch.nn as nn
import torch.optim as optim
from torch.utils.data import Dataset, DataLoader, random_split
from torchvision import transforms
from PIL import Image
import os
import glob
import numpy as np
from sklearn.metrics import classification_report, confusion_matrix, f1_score, precision_score, recall_score
import matplotlib.pyplot as plt
import seaborn as sns

class SpectrogramDataset(Dataset):
    def __init__(self, image_dir, transform=None):
        self.image_paths = glob.glob(os.path.join(image_dir, "*.png"))
        self.transform = transform
        
        # Create label mapping from filename
        self.label_map = {
            '00000': 0,  # Background
            '10000': 1,  # Bebop
            '10001': 1,  # Bebop
            '10010': 1,  # Bebop
            '10011': 1,  # Bebop
            '10100': 2,  # AR
            '10101': 2,  # AR
            '10110': 2,  # AR
            '10111': 2,  # AR
            '11000': 3,  # Phantom
        }
        
        print(f"Found {len(self.image_paths)} spectrogram images")
        
        # Print class distribution
        class_counts = {}
        for img_path in self.image_paths:
            filename = os.path.basename(img_path)
            parts = filename.split('_')
            if len(parts) >= 2:
                bui_code = parts[1].rstrip('HL')
                label = self.label_map.get(bui_code, 0)
                class_counts[label] = class_counts.get(label, 0) + 1
        
        class_names = ['Background', 'Bebop', 'AR', 'Phantom']
        print("Class distribution:")
        for i, count in class_counts.items():
            print(f"  {class_names[i]}: {count}")
        
    def __len__(self):
        return len(self.image_paths)
    
    def __getitem__(self, idx):
        img_path = self.image_paths[idx]
        image = Image.open(img_path).convert('RGB')
        
        # Extract label from filename
        filename = os.path.basename(img_path)
        parts = filename.split('_')
        if len(parts) >= 2:
            bui_code = parts[1].rstrip('HL')
            label = self.label_map.get(bui_code, 0)
        else:
            label = 0
        
        if self.transform:
            image = self.transform(image)
            
        return image, label

class RFClassifier(nn.Module):
    def __init__(self, num_classes=4):
        super().__init__()
        self.features = nn.Sequential(
            # First conv block
            nn.Conv2d(3, 32, 3, padding=1),
            nn.BatchNorm2d(32),
            nn.ReLU(),
            nn.MaxPool2d(2),
            
            # Second conv block
            nn.Conv2d(32, 64, 3, padding=1),
            nn.BatchNorm2d(64),
            nn.ReLU(),
            nn.MaxPool2d(2),
            
            # Third conv block
            nn.Conv2d(64, 128, 3, padding=1),
            nn.BatchNorm2d(128),
            nn.ReLU(),
            nn.MaxPool2d(2),
            
            # Fourth conv block
            nn.Conv2d(128, 256, 3, padding=1),
            nn.BatchNorm2d(256),
            nn.ReLU(),
            nn.AdaptiveAvgPool2d((1, 1))
        )
        
        self.classifier = nn.Sequential(
            nn.Dropout(0.5),
            nn.Linear(256, 128),
            nn.ReLU(),
            nn.Dropout(0.3),
            nn.Linear(128, num_classes)
        )
    
    def forward(self, x):
        x = self.features(x)
        x = x.view(x.size(0), -1)
        return self.classifier(x)

def calculate_metrics(predictions, targets, class_names):
    """Calculate comprehensive metrics"""
    # Basic metrics
    accuracy = (predictions == targets).mean()
    
    # Per-class metrics
    precision = precision_score(targets, predictions, average=None, zero_division=0)
    recall = recall_score(targets, predictions, average=None, zero_division=0)
    f1 = f1_score(targets, predictions, average=None, zero_division=0)
    
    # Macro averages
    precision_macro = precision_score(targets, predictions, average='macro', zero_division=0)
    recall_macro = recall_score(targets, predictions, average='macro', zero_division=0)
    f1_macro = f1_score(targets, predictions, average='macro', zero_division=0)
    
    # Weighted averages
    precision_weighted = precision_score(targets, predictions, average='weighted', zero_division=0)
    recall_weighted = recall_score(targets, predictions, average='weighted', zero_division=0)
    f1_weighted = f1_score(targets, predictions, average='weighted', zero_division=0)
    
    return {
        'accuracy': accuracy,
        'precision_per_class': precision,
        'recall_per_class': recall,
        'f1_per_class': f1,
        'precision_macro': precision_macro,
        'recall_macro': recall_macro,
        'f1_macro': f1_macro,
        'precision_weighted': precision_weighted,
        'recall_weighted': recall_weighted,
        'f1_weighted': f1_weighted
    }

def train_model(model, train_loader, val_loader, num_epochs=50, device='cuda'):
    criterion = nn.CrossEntropyLoss()
    optimizer = optim.Adam(model.parameters(), lr=0.001, weight_decay=1e-4)
    scheduler = optim.lr_scheduler.StepLR(optimizer, step_size=20, gamma=0.1)
    
    train_losses = []
    val_accuracies = []
    val_f1_scores = []
    
    class_names = ['Background', 'Bebop', 'AR', 'Phantom']
    
    for epoch in range(num_epochs):
        # Training
        model.train()
        running_loss = 0.0
        for images, labels in train_loader:
            images, labels = images.to(device), labels.to(device)
            
            optimizer.zero_grad()
            outputs = model(images)
            loss = criterion(outputs, labels)
            loss.backward()
            optimizer.step()
            
            running_loss += loss.item()
        
        # Validation
        model.eval()
        all_preds = []
        all_labels = []
        with torch.no_grad():
            for images, labels in val_loader:
                images, labels = images.to(device), labels.to(device)
                outputs = model(images)
                _, predicted = torch.max(outputs.data, 1)
                all_preds.extend(predicted.cpu().numpy())
                all_labels.extend(labels.cpu().numpy())
        
        # Calculate metrics
        metrics = calculate_metrics(np.array(all_preds), np.array(all_labels), class_names)
        
        avg_loss = running_loss / len(train_loader)
        train_losses.append(avg_loss)
        val_accuracies.append(metrics['accuracy'])
        val_f1_scores.append(metrics['f1_macro'])
        
        print(f'Epoch [{epoch+1}/{num_epochs}]')
        print(f'  Loss: {avg_loss:.4f}')
        print(f'  Val Acc: {metrics["accuracy"]:.4f}')
        print(f'  Val F1: {metrics["f1_macro"]:.4f}')
        print(f'  Val Precision: {metrics["precision_macro"]:.4f}')
        print(f'  Val Recall: {metrics["recall_macro"]:.4f}')
        
        # Print per-class metrics every 10 epochs
        if (epoch + 1) % 10 == 0:
            print(f'  Per-class F1: {[f"{f:.3f}" for f in metrics["f1_per_class"]]}')
        
        scheduler.step()
    
    return train_losses, val_accuracies, val_f1_scores

def evaluate_model(model, test_loader, device='cuda'):
    model.eval()
    all_preds = []
    all_labels = []
    
    with torch.no_grad():
        for images, labels in test_loader:
            images, labels = images.to(device), labels.to(device)
            outputs = model(images)
            _, predicted = torch.max(outputs, 1)
            
            all_preds.extend(predicted.cpu().numpy())
            all_labels.extend(labels.cpu().numpy())
    
    class_names = ['Background', 'Bebop', 'AR', 'Phantom']
    
    # Calculate comprehensive metrics
    metrics = calculate_metrics(np.array(all_preds), np.array(all_labels), class_names)
    
    print("\n" + "="*50)
    print("FINAL EVALUATION RESULTS")
    print("="*50)
    print(f"Overall Accuracy: {metrics['accuracy']:.4f}")
    print(f"Macro F1 Score: {metrics['f1_macro']:.4f}")
    print(f"Weighted F1 Score: {metrics['f1_weighted']:.4f}")
    print(f"Macro Precision: {metrics['precision_macro']:.4f}")
    print(f"Macro Recall: {metrics['recall_macro']:.4f}")
    
    print("\nPer-class metrics:")
    for i, class_name in enumerate(class_names):
        print(f"  {class_name}:")
        print(f"    Precision: {metrics['precision_per_class'][i]:.4f}")
        print(f"    Recall: {metrics['recall_per_class'][i]:.4f}")
        print(f"    F1: {metrics['f1_per_class'][i]:.4f}")
    
    # Print classification report
    print("\nDetailed Classification Report:")
    print(classification_report(all_labels, all_preds, target_names=class_names))
    
    # Plot confusion matrix
    cm = confusion_matrix(all_labels, all_preds)
    plt.figure(figsize=(10, 8))
    sns.heatmap(cm, annot=True, fmt='d', cmap='Blues', 
                xticklabels=class_names, yticklabels=class_names)
    plt.title('Confusion Matrix')
    plt.ylabel('True Label')
    plt.xlabel('Predicted Label')
    plt.tight_layout()
    plt.savefig('confusion_matrix.png', dpi=300, bbox_inches='tight')
    plt.show()
    
    return metrics

def main():
    # Force GPU detection and usage
    print("Checking CUDA availability...")
    print(f"CUDA available: {torch.cuda.is_available()}")
    if torch.cuda.is_available():
        print(f"CUDA device count: {torch.cuda.device_count()}")
        print(f"Current CUDA device: {torch.cuda.current_device()}")
        print(f"CUDA device name: {torch.cuda.get_device_name()}")
    
    # Set device
    device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
    print(f"Using device: {device}")
    
    # Data transforms with augmentation
    train_transform = transforms.Compose([
        transforms.Resize((224, 224)),
        transforms.RandomHorizontalFlip(p=0.3),
        transforms.RandomRotation(5),
        transforms.ColorJitter(brightness=0.2, contrast=0.2),
        transforms.ToTensor(),
        transforms.Normalize(mean=[0.485, 0.456, 0.406], std=[0.229, 0.224, 0.225])
    ])
    
    val_transform = transforms.Compose([
        transforms.Resize((224, 224)),
        transforms.ToTensor(),
        transforms.Normalize(mean=[0.485, 0.456, 0.406], std=[0.229, 0.224, 0.225])
    ])
    
    # Load dataset
    dataset = SpectrogramDataset('/path/to/your/results', transform=None)
    
    # Split dataset
    train_size = int(0.7 * len(dataset))
    val_size = int(0.15 * len(dataset))
    test_size = len(dataset) - train_size - val_size
    
    train_dataset, val_dataset, test_dataset = random_split(
        dataset, [train_size, val_size, test_size]
    )
    
    # Apply transforms
    train_dataset.dataset.transform = train_transform
    val_dataset.dataset.transform = val_transform
    test_dataset.dataset.transform = val_transform
    
    # Create data loaders
    train_loader = DataLoader(train_dataset, batch_size=32, shuffle=True, num_workers=4)
    val_loader = DataLoader(val_dataset, batch_size=32, shuffle=False, num_workers=4)
    test_loader = DataLoader(test_dataset, batch_size=32, shuffle=False, num_workers=4)
    
    print(f"Train: {len(train_dataset)}, Val: {len(val_dataset)}, Test: {len(test_dataset)}")
    
    # Create model
    model = RFClassifier(num_classes=4).to(device)
    print(f"Model parameters: {sum(p.numel() for p in model.parameters()):,}")
    
    # Train model
    print("Starting training...")
    train_losses, val_accuracies, val_f1_scores = train_model(
        model, train_loader, val_loader, num_epochs=50, device=device
    )
    
    # Evaluate model
    print("Evaluating model...")
    metrics = evaluate_model(model, test_loader, device=device)
    
    # Save model
    torch.save(model.state_dict(), 'rf_spectrogram_model.pth')
    print("Model saved as 'rf_spectrogram_model.pth'")
    
    # Plot training curves
    plt.figure(figsize=(15, 5))
    
    plt.subplot(1, 3, 1)
    plt.plot(train_losses)
    plt.title('Training Loss')
    plt.xlabel('Epoch')
    plt.ylabel('Loss')
    plt.grid(True)
    
    plt.subplot(1, 3, 2)
    plt.plot(val_accuracies)
    plt.title('Validation Accuracy')
    plt.xlabel('Epoch')
    plt.ylabel('Accuracy')
    plt.grid(True)
    
    plt.subplot(1, 3, 3)
    plt.plot(val_f1_scores)
    plt.title('Validation F1 Score')
    plt.xlabel('Epoch')
    plt.ylabel('F1 Score')
    plt.grid(True)
    
    plt.tight_layout()
    plt.savefig('training_curves.png', dpi=300, bbox_inches='tight')
    plt.show()

if __name__ == "__main__":
    main()
