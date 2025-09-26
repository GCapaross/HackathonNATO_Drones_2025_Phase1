"""
Enhanced CNN training with K-Fold Cross-Validation
More robust evaluation than single train/validation split
"""

import torch
import torch.nn as nn
import torch.optim as optim
from torch.utils.data import DataLoader, Subset
from sklearn.model_selection import KFold
import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns
from sklearn.metrics import confusion_matrix, classification_report
import os
import time
import argparse

# Import our modules
from data_loader import SpectrogramDataset, create_data_loaders
from cnn_model import create_model, count_parameters, get_optimizer, get_scheduler

class KFoldCNNTrainer:
    def __init__(self, data_dir, num_folds=5, batch_size=32, epochs=50, lr=0.001):
        self.data_dir = data_dir
        self.num_folds = num_folds
        self.batch_size = batch_size
        self.epochs = epochs
        self.lr = lr
        self.device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
        
        # Load full dataset
        self.full_dataset = SpectrogramDataset(data_dir, task='classification')
        self.class_names = self.full_dataset.class_names
        
        print(f"Full dataset size: {len(self.full_dataset)}")
        print(f"Classes: {self.class_names}")
        print(f"Device: {self.device}")
        
        # Results storage
        self.fold_results = []
        self.best_models = []

    def train_fold(self, fold, train_indices, val_indices):
        """Train model for one fold"""
        print(f"\n{'='*50}")
        print(f"FOLD {fold + 1}/{self.num_folds}")
        print(f"{'='*50}")
        
        # Create data loaders for this fold
        train_subset = Subset(self.full_dataset, train_indices)
        val_subset = Subset(self.full_dataset, val_indices)
        
        train_loader = DataLoader(train_subset, batch_size=self.batch_size, shuffle=True, num_workers=4)
        val_loader = DataLoader(val_subset, batch_size=self.batch_size, shuffle=False, num_workers=4)
        
        print(f"Train samples: {len(train_subset)}")
        print(f"Validation samples: {len(val_subset)}")
        
        # Create model
        model, model_name = create_model(num_classes=len(self.class_names))
        model = model.to(self.device)
        
        # Setup training
        criterion = nn.CrossEntropyLoss()
        optimizer = get_optimizer(model, optimizer_type='adam', lr=self.lr)
        scheduler = get_scheduler(optimizer, scheduler_type='step')
        
        # Training history
        train_losses = []
        val_losses = []
        train_accuracies = []
        val_accuracies = []
        
        best_val_acc = 0.0
        best_model_state = None
        
        print(f"\nTraining {model_name} with {count_parameters(model):,} parameters...")
        
        # Training loop
        for epoch in range(self.epochs):
            # Training phase
            model.train()
            train_loss = 0.0
            train_correct = 0
            train_total = 0
            
            for images, labels in train_loader:
                images, labels = images.to(self.device), labels.to(self.device)
                
                optimizer.zero_grad()
                outputs = model(images)
                loss = criterion(outputs, labels)
                loss.backward()
                optimizer.step()
                
                train_loss += loss.item() * images.size(0)
                _, predicted = torch.max(outputs.data, 1)
                train_total += labels.size(0)
                train_correct += (predicted == labels).sum().item()
            
            # Validation phase
            model.eval()
            val_loss = 0.0
            val_correct = 0
            val_total = 0
            
            with torch.no_grad():
                for images, labels in val_loader:
                    images, labels = images.to(self.device), labels.to(self.device)
                    outputs = model(images)
                    loss = criterion(outputs, labels)
                    
                    val_loss += loss.item() * images.size(0)
                    _, predicted = torch.max(outputs.data, 1)
                    val_total += labels.size(0)
                    val_correct += (predicted == labels).sum().item()
            
            # Calculate metrics
            train_loss_avg = train_loss / train_total
            val_loss_avg = val_loss / val_total
            train_acc = train_correct / train_total
            val_acc = val_correct / val_total
            
            train_losses.append(train_loss_avg)
            val_losses.append(val_loss_avg)
            train_accuracies.append(train_acc)
            val_accuracies.append(val_acc)
            
            # Save best model for this fold
            if val_acc > best_val_acc:
                best_val_acc = val_acc
                best_model_state = model.state_dict().copy()
            
            scheduler.step()
            
            # Print progress
            if epoch % 10 == 0 or epoch == self.epochs - 1:
                print(f"Epoch {epoch+1:3d}/{self.epochs}: "
                      f"Train Loss: {train_loss_avg:.4f}, Train Acc: {train_acc:.4f} | "
                      f"Val Loss: {val_loss_avg:.4f}, Val Acc: {val_acc:.4f}")
        
        # Store results for this fold
        fold_result = {
            'fold': fold + 1,
            'best_val_acc': best_val_acc,
            'train_losses': train_losses,
            'val_losses': val_losses,
            'train_accuracies': train_accuracies,
            'val_accuracies': val_accuracies,
            'best_model_state': best_model_state
        }
        
        self.fold_results.append(fold_result)
        self.best_models.append(best_model_state)
        
        print(f"\nFold {fold + 1} completed - Best Val Acc: {best_val_acc:.4f}")
        
        return fold_result

    def run_kfold(self):
        """Run k-fold cross-validation"""
        print(f"\nStarting {self.num_folds}-Fold Cross-Validation")
        print(f"Total samples: {len(self.full_dataset)}")
        
        # Create k-fold splitter
        kfold = KFold(n_splits=self.num_folds, shuffle=True, random_state=42)
        
        # Get all indices
        all_indices = np.arange(len(self.full_dataset))
        
        # Run each fold
        for fold, (train_indices, val_indices) in enumerate(kfold.split(all_indices)):
            self.train_fold(fold, train_indices, val_indices)
        
        # Analyze results
        self.analyze_results()
        self.plot_results()
        
        return self.fold_results

    def analyze_results(self):
        """Analyze k-fold results"""
        print(f"\n{'='*60}")
        print("K-FOLD CROSS-VALIDATION RESULTS")
        print(f"{'='*60}")
        
        # Extract best accuracies
        best_accs = [result['best_val_acc'] for result in self.fold_results]
        
        print(f"Fold Results:")
        for i, acc in enumerate(best_accs):
            print(f"  Fold {i+1}: {acc:.4f}")
        
        print(f"\nOverall Statistics:")
        print(f"  Mean Accuracy: {np.mean(best_accs):.4f} ± {np.std(best_accs):.4f}")
        print(f"  Best Accuracy: {np.max(best_accs):.4f}")
        print(f"  Worst Accuracy: {np.min(best_accs):.4f}")
        
        # Find best overall model
        best_fold_idx = np.argmax(best_accs)
        print(f"\nBest Model: Fold {best_fold_idx + 1} with accuracy {best_accs[best_fold_idx]:.4f}")
        
        # Save best model
        best_model_path = 'best_model_kfold.pth'
        torch.save(self.best_models[best_fold_idx], best_model_path)
        print(f"Best model saved as: {best_model_path}")

    def plot_results(self):
        """Plot k-fold results"""
        # Create subplots
        fig, axes = plt.subplots(2, 2, figsize=(15, 10))
        
        # Plot 1: Accuracy comparison across folds
        fold_numbers = range(1, self.num_folds + 1)
        best_accs = [result['best_val_acc'] for result in self.fold_results]
        
        axes[0, 0].bar(fold_numbers, best_accs, color='skyblue', alpha=0.7)
        axes[0, 0].set_title('Best Validation Accuracy per Fold')
        axes[0, 0].set_xlabel('Fold')
        axes[0, 0].set_ylabel('Accuracy')
        axes[0, 0].set_ylim(0, 1)
        
        # Add value labels on bars
        for i, acc in enumerate(best_accs):
            axes[0, 0].text(i + 1, acc + 0.01, f'{acc:.3f}', ha='center', va='bottom')
        
        # Plot 2: Training curves for all folds
        for i, result in enumerate(self.fold_results):
            epochs = range(1, len(result['train_accuracies']) + 1)
            axes[0, 1].plot(epochs, result['train_accuracies'], alpha=0.7, label=f'Fold {i+1} Train')
            axes[0, 1].plot(epochs, result['val_accuracies'], alpha=0.7, linestyle='--', label=f'Fold {i+1} Val')
        
        axes[0, 1].set_title('Training Curves - All Folds')
        axes[0, 1].set_xlabel('Epoch')
        axes[0, 1].set_ylabel('Accuracy')
        axes[0, 1].legend(bbox_to_anchor=(1.05, 1), loc='upper left')
        axes[0, 1].grid(True, alpha=0.3)
        
        # Plot 3: Loss curves
        for i, result in enumerate(self.fold_results):
            epochs = range(1, len(result['train_losses']) + 1)
            axes[1, 0].plot(epochs, result['train_losses'], alpha=0.7, label=f'Fold {i+1} Train')
            axes[1, 0].plot(epochs, result['val_losses'], alpha=0.7, linestyle='--', label=f'Fold {i+1} Val')
        
        axes[1, 0].set_title('Loss Curves - All Folds')
        axes[1, 0].set_xlabel('Epoch')
        axes[1, 0].set_ylabel('Loss')
        axes[1, 0].legend(bbox_to_anchor=(1.05, 1), loc='upper left')
        axes[1, 0].grid(True, alpha=0.3)
        
        # Plot 4: Accuracy distribution
        axes[1, 1].hist(best_accs, bins=min(10, self.num_folds), alpha=0.7, color='lightgreen')
        axes[1, 1].axvline(np.mean(best_accs), color='red', linestyle='--', label=f'Mean: {np.mean(best_accs):.3f}')
        axes[1, 1].axvline(np.median(best_accs), color='orange', linestyle='--', label=f'Median: {np.median(best_accs):.3f}')
        axes[1, 1].set_title('Accuracy Distribution')
        axes[1, 1].set_xlabel('Accuracy')
        axes[1, 1].set_ylabel('Frequency')
        axes[1, 1].legend()
        axes[1, 1].grid(True, alpha=0.3)
        
        plt.tight_layout()
        plt.savefig('./kfold_results.png', dpi=300, bbox_inches='tight')
        plt.show()
        print("K-fold results plot saved as 'kfold_results.png'")

def main():
    parser = argparse.ArgumentParser(description='Train CNN with K-Fold Cross-Validation')
    parser.add_argument('--data_dir', type=str, 
                        default='/home/gabriel/Desktop/HackathonNATO_Drones_2025/spectrogram_training_data_20220711',
                        help='Path to dataset directory')
    parser.add_argument('--num_folds', type=int, default=5,
                        help='Number of folds for cross-validation')
    parser.add_argument('--epochs', type=int, default=50,
                        help='Number of training epochs per fold')
    parser.add_argument('--batch_size', type=int, default=32,
                        help='Batch size for training')
    parser.add_argument('--lr', type=float, default=0.001,
                        help='Learning rate')
    
    args = parser.parse_args()
    
    # Create trainer
    trainer = KFoldCNNTrainer(
        data_dir=args.data_dir,
        num_folds=args.num_folds,
        batch_size=args.batch_size,
        epochs=args.epochs,
        lr=args.lr
    )
    
    # Run k-fold cross-validation
    results = trainer.run_kfold()
    
    print(f"\nK-Fold Cross-Validation completed!")
    print(f"Results saved and plots generated.")

if __name__ == "__main__":
    main()
