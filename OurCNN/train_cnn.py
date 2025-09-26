"""
Training Script for NATO Hackathon CNN
Trains CNN model for spectrogram classification
"""

import torch
import torch.nn as nn
import torch.optim as optim
from torch.utils.data import DataLoader
import numpy as np
import matplotlib.pyplot as plt
from sklearn.metrics import classification_report, confusion_matrix
import seaborn as sns
import os
import json
from datetime import datetime
import argparse

from data_loader import create_data_loaders, analyze_dataset
from cnn_model import create_model, get_optimizer, get_scheduler, count_parameters

class CNNTrainer:
    """
    CNN Trainer for spectrogram classification
    """
    
    def __init__(self, model, train_loader, val_loader, class_names, device='cuda'):
        self.model = model.to(device)
        self.train_loader = train_loader
        self.val_loader = val_loader
        self.class_names = class_names
        self.device = device
        
        # Training history
        self.train_losses = []
        self.val_losses = []
        self.train_accuracies = []
        self.val_accuracies = []
        
        # Loss and optimizer
        self.criterion = nn.CrossEntropyLoss()
        self.optimizer = get_optimizer(self.model, optimizer_type='adam', lr=0.001)
        self.scheduler = get_scheduler(self.optimizer, scheduler_type='step')
        
    def train_epoch(self):
        """Train for one epoch"""
        self.model.train()
        running_loss = 0.0
        correct = 0
        total = 0
        
        for batch_idx, (images, labels) in enumerate(self.train_loader):
            images, labels = images.to(self.device), labels.to(self.device)
            
            # Zero gradients
            self.optimizer.zero_grad()
            
            # Forward pass
            outputs = self.model(images)
            loss = self.criterion(outputs, labels)
            
            # Backward pass
            loss.backward()
            self.optimizer.step()
            
            # Statistics
            running_loss += loss.item()
            _, predicted = torch.max(outputs.data, 1)
            total += labels.size(0)
            correct += (predicted == labels).sum().item()
            
            if batch_idx % 100 == 0:
                print(f'Batch {batch_idx}/{len(self.train_loader)}, Loss: {loss.item():.4f}')
        
        epoch_loss = running_loss / len(self.train_loader)
        epoch_acc = 100. * correct / total
        
        return epoch_loss, epoch_acc
    
    def validate_epoch(self):
        """Validate for one epoch"""
        self.model.eval()
        running_loss = 0.0
        correct = 0
        total = 0
        all_predictions = []
        all_labels = []
        
        with torch.no_grad():
            for images, labels in self.val_loader:
                images, labels = images.to(self.device), labels.to(self.device)
                
                outputs = self.model(images)
                loss = self.criterion(outputs, labels)
                
                running_loss += loss.item()
                _, predicted = torch.max(outputs.data, 1)
                total += labels.size(0)
                correct += (predicted == labels).sum().item()
                
                all_predictions.extend(predicted.cpu().numpy())
                all_labels.extend(labels.cpu().numpy())
        
        epoch_loss = running_loss / len(self.val_loader)
        epoch_acc = 100. * correct / total
        
        return epoch_loss, epoch_acc, all_predictions, all_labels
    
    def train(self, num_epochs=50, save_best=True, save_dir='./checkpoints'):
        """Train the model"""
        
        # Create save directory
        os.makedirs(save_dir, exist_ok=True)
        
        best_val_acc = 0.0
        best_epoch = 0
        
        print(f"Starting training for {num_epochs} epochs...")
        print(f"Model parameters: {count_parameters(self.model):,}")
        print(f"Device: {self.device}")
        print(f"Classes: {self.class_names}")
        
        for epoch in range(num_epochs):
            print(f"\nEpoch {epoch+1}/{num_epochs}")
            print("-" * 50)
            
            # Train
            train_loss, train_acc = self.train_epoch()
            
            # Validate
            val_loss, val_acc, predictions, labels = self.validate_epoch()
            
            # Update learning rate
            self.scheduler.step()
            
            # Store history
            self.train_losses.append(train_loss)
            self.val_losses.append(val_loss)
            self.train_accuracies.append(train_acc)
            self.val_accuracies.append(val_acc)
            
            # Print results
            print(f"Train Loss: {train_loss:.4f}, Train Acc: {train_acc:.2f}%")
            print(f"Val Loss: {val_loss:.4f}, Val Acc: {val_acc:.2f}%")
            print(f"Learning Rate: {self.optimizer.param_groups[0]['lr']:.6f}")
            
            # Save best model
            if val_acc > best_val_acc:
                best_val_acc = val_acc
                best_epoch = epoch + 1
                
                if save_best:
                    torch.save({
                        'epoch': epoch + 1,
                        'model_state_dict': self.model.state_dict(),
                        'optimizer_state_dict': self.optimizer.state_dict(),
                        'val_acc': val_acc,
                        'val_loss': val_loss,
                        'class_names': self.class_names
                    }, os.path.join(save_dir, 'best_model.pth'))
                    print(f"New best model saved! Val Acc: {val_acc:.2f}%")
            
            # Save checkpoint every 10 epochs
            if (epoch + 1) % 10 == 0:
                torch.save({
                    'epoch': epoch + 1,
                    'model_state_dict': self.model.state_dict(),
                    'optimizer_state_dict': self.optimizer.state_dict(),
                    'val_acc': val_acc,
                    'val_loss': val_loss,
                    'class_names': self.class_names
                }, os.path.join(save_dir, f'checkpoint_epoch_{epoch+1}.pth'))
        
        print(f"\nTraining completed!")
        print(f"Best validation accuracy: {best_val_acc:.2f}% at epoch {best_epoch}")
        
        return best_val_acc, best_epoch
    
    def plot_training_history(self, save_path='./training_history.png'):
        """Plot training history"""
        fig, (ax1, ax2) = plt.subplots(1, 2, figsize=(15, 5))
        
        # Loss plot
        ax1.plot(self.train_losses, label='Train Loss')
        ax1.plot(self.val_losses, label='Validation Loss')
        ax1.set_title('Training and Validation Loss')
        ax1.set_xlabel('Epoch')
        ax1.set_ylabel('Loss')
        ax1.legend()
        ax1.grid(True)
        
        # Accuracy plot
        ax2.plot(self.train_accuracies, label='Train Accuracy')
        ax2.plot(self.val_accuracies, label='Validation Accuracy')
        ax2.set_title('Training and Validation Accuracy')
        ax2.set_xlabel('Epoch')
        ax2.set_ylabel('Accuracy (%)')
        ax2.legend()
        ax2.grid(True)
        
        plt.tight_layout()
        plt.savefig(save_path, dpi=300, bbox_inches='tight')
        plt.show()
    
    def evaluate_model(self):
        """Evaluate model and generate detailed metrics"""
        self.model.eval()
        all_predictions = []
        all_labels = []
        
        with torch.no_grad():
            for images, labels in self.val_loader:
                images, labels = images.to(self.device), labels.to(self.device)
                outputs = self.model(images)
                _, predicted = torch.max(outputs, 1)
                
                all_predictions.extend(predicted.cpu().numpy())
                all_labels.extend(labels.cpu().numpy())
        
        # Classification report
        # Get unique classes present in predictions and labels
        unique_labels = sorted(list(set(all_labels + all_predictions)))
        target_names_subset = [self.class_names[i] for i in unique_labels]
        
        report = classification_report(all_labels, all_predictions, 
                                     target_names=target_names_subset,
                                     labels=unique_labels,
                                     output_dict=True)
        
        print("Classification Report:")
        print(classification_report(all_labels, all_predictions, 
                                   target_names=target_names_subset,
                                   labels=unique_labels))
        
        # Confusion matrix
        cm = confusion_matrix(all_labels, all_predictions)
        plt.figure(figsize=(8, 6))
        sns.heatmap(cm, annot=True, fmt='d', cmap='Blues', 
                   xticklabels=self.class_names, 
                   yticklabels=self.class_names)
        plt.title('Confusion Matrix')
        plt.xlabel('Predicted')
        plt.ylabel('Actual')
        plt.tight_layout()
        plt.savefig('./confusion_matrix.png', dpi=300, bbox_inches='tight')
        plt.show()
        
        return report

def main():
    """Main training function"""
    parser = argparse.ArgumentParser(description='Train Custom CNN for spectrogram classification')
    parser.add_argument('--data_dir', type=str, 
                       default='/home/gabriel/Desktop/HackathonNATO_Drones_2025',
                       help='Path to dataset directory')
    parser.add_argument('--batch_size', type=int, default=32,
                       help='Batch size for training')
    parser.add_argument('--epochs', type=int, default=50,
                       help='Number of training epochs')
    parser.add_argument('--lr', type=float, default=0.001,
                       help='Learning rate')
    parser.add_argument('--device', type=str, default='cuda',
                       help='Device to use (cuda/cpu)')
    
    args = parser.parse_args()
    
    # Set device
    device = torch.device(args.device if torch.cuda.is_available() else 'cpu')
    print(f"Using device: {device}")
    
    # Analyze dataset
    print("Analyzing dataset...")
    total_images, labeled_images = analyze_dataset(args.data_dir)
    
    # Create data loaders
    print("Creating data loaders...")
    train_loader, val_loader, class_names = create_data_loaders(
        args.data_dir, batch_size=args.batch_size, test_size=0.2, task='classification'
    )
    
    print(f"Training batches: {len(train_loader)}")
    print(f"Validation batches: {len(val_loader)}")
    print(f"Classes: {class_names}")
    
    # Create model
    print("Creating Custom SpectrogramCNN model...")
    model, model_name = create_model(num_classes=len(class_names))
    print(f"Model: {model_name}")
    print(f"Parameters: {count_parameters(model):,}")
    
    # Create trainer
    trainer = CNNTrainer(model, train_loader, val_loader, class_names, device)
    
    # Train model
    print("Starting training...")
    best_acc, best_epoch = trainer.train(num_epochs=args.epochs)
    
    # Plot training history
    trainer.plot_training_history()
    
    # Evaluate model
    print("Evaluating model...")
    report = trainer.evaluate_model()
    
    # Save training results
    results = {
        'model_name': model_name,
        'best_accuracy': best_acc,
        'best_epoch': best_epoch,
        'total_parameters': count_parameters(model),
        'training_samples': len(train_loader.dataset),
        'validation_samples': len(val_loader.dataset),
        'class_names': class_names,
        'classification_report': report
    }
    
    with open('./training_results.json', 'w') as f:
        json.dump(results, f, indent=2)
    
    print("Training completed successfully!")
    print(f"Best accuracy: {best_acc:.2f}% at epoch {best_epoch}")

if __name__ == "__main__":
    main()
