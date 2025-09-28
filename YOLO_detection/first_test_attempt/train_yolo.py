"""
Training script for YOLO RF Signal Detection
"""

import torch
import torch.nn as nn
import torch.nn.functional as F
import torch.optim as optim
from torch.utils.data import DataLoader
import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns
from sklearn.metrics import precision_recall_fscore_support
import os
import time
from tqdm import tqdm
import argparse
import json
from datetime import datetime

# Import our modules
from yolo_data_loader import create_yolo_data_loaders, analyze_dataset
from yolo_model import create_yolo_model, count_parameters

class YOLOTrainer:
    """
    YOLO trainer for RF signal detection
    """
    
    def __init__(self, model, train_loader, val_loader, class_names, device='cuda'):
        self.model = model
        self.train_loader = train_loader
        self.val_loader = val_loader
        self.class_names = class_names
        self.device = torch.device(device if torch.cuda.is_available() else 'cpu')
        
        # Move model to device
        self.model.to(self.device)
        
        # Loss function (simplified YOLO loss)
        self.criterion = YOLOLoss(num_classes=len(class_names))
        
        # Optimizer
        self.optimizer = optim.AdamW(
            self.model.parameters(), 
            lr=0.001, 
            weight_decay=0.0005
        )
        
        # Learning rate scheduler
        self.scheduler = optim.lr_scheduler.CosineAnnealingLR(
            self.optimizer, T_max=100, eta_min=1e-6
        )
        
        # Training history
        self.train_losses = []
        self.val_losses = []
        self.val_maps = []
        
        print(f"Using device: {self.device}")
        print(f"Model parameters: {count_parameters(self.model):,}")
        print(f"Classes: {self.class_names}")
    
    def train_epoch(self, epoch):
        """Train for one epoch"""
        self.model.train()
        total_loss = 0.0
        num_batches = len(self.train_loader)
        
        pbar = tqdm(self.train_loader, desc=f'Epoch {epoch+1}')
        for batch_idx, batch in enumerate(pbar):
            images = batch['images'].to(self.device)
            labels = batch['labels']
            
            # Forward pass
            self.optimizer.zero_grad()
            outputs = self.model(images)
            
            # Compute loss (simplified)
            loss = self.compute_yolo_loss(outputs, labels)
            
            # Backward pass
            loss.backward()
            self.optimizer.step()
            
            total_loss += loss.item()
            
            # Update progress bar
            pbar.set_postfix({
                'Loss': f'{loss.item():.4f}',
                'Avg Loss': f'{total_loss/(batch_idx+1):.4f}'
            })
        
        avg_loss = total_loss / num_batches
        self.train_losses.append(avg_loss)
        
        return avg_loss
    
    def validate_epoch(self, epoch):
        """Validate for one epoch"""
        self.model.eval()
        total_loss = 0.0
        num_batches = len(self.val_loader)
        
        with torch.no_grad():
            for batch in tqdm(self.val_loader, desc='Validation'):
                images = batch['images'].to(self.device)
                labels = batch['labels']
                
                # Forward pass
                outputs = self.model(images)
                
                # Compute loss
                loss = self.compute_yolo_loss(outputs, labels)
                total_loss += loss.item()
        
        avg_loss = total_loss / num_batches
        self.val_losses.append(avg_loss)
        
        return avg_loss
    
    def compute_yolo_loss(self, outputs, labels):
        """
        Compute proper YOLO loss for object detection
        Our model outputs shape: (B, 3, 5 + num_classes)
        """
        batch_size = outputs.shape[0]
        device = outputs.device
        
        # Loss components
        obj_loss = 0.0
        no_obj_loss = 0.0
        coord_loss = 0.0
        class_loss = 0.0
        
        for b in range(batch_size):
            # Get labels for this batch item
            if b < len(labels) and len(labels[b]) > 0:
                batch_labels = labels[b]  # Shape: (N, 5) where N is number of objects
                
                if len(batch_labels) > 0:
                    # Convert labels to tensor if needed and ensure it's on the correct device
                    if not isinstance(batch_labels, torch.Tensor):
                        batch_labels = torch.tensor(batch_labels, dtype=torch.float32, device=device)
                    else:
                        batch_labels = batch_labels.to(device)
                    
                    # For each anchor (3 anchors per cell)
                    for anchor_idx in range(3):
                        anchor_output = outputs[b, anchor_idx]  # Shape: (5 + num_classes)
                        
                        # Extract components
                        pred_x = torch.sigmoid(anchor_output[0])
                        pred_y = torch.sigmoid(anchor_output[1])
                        pred_w = anchor_output[2]
                        pred_h = anchor_output[3]
                        pred_obj = torch.sigmoid(anchor_output[4])
                        pred_classes = anchor_output[5:]
                        
                        # Find best matching ground truth box for this anchor
                        best_iou = 0.0
                        best_gt_idx = -1
                        
                        for gt_idx, gt_box in enumerate(batch_labels):
                            gt_x, gt_y, gt_w, gt_h, gt_class = gt_box
                            
                            # Calculate IoU between predicted and ground truth
                            iou = self.calculate_iou(
                                pred_x, pred_y, pred_w, pred_h,
                                gt_x, gt_y, gt_w, gt_h
                            )
                            
                            if iou > best_iou:
                                best_iou = iou
                                best_gt_idx = gt_idx
                        
                        # Objectness loss
                        if best_iou > 0.5:  # Good match
                            obj_target = torch.tensor(1.0, device=device, dtype=torch.float32)
                            obj_loss += F.binary_cross_entropy(pred_obj, obj_target)
                            
                            # Coordinate loss
                            if best_gt_idx >= 0:
                                gt_box = batch_labels[best_gt_idx]
                                coord_loss += F.mse_loss(pred_x, gt_box[0].to(device))
                                coord_loss += F.mse_loss(pred_y, gt_box[1].to(device))
                                coord_loss += F.mse_loss(pred_w, gt_box[2].to(device))
                                coord_loss += F.mse_loss(pred_h, gt_box[3].to(device))
                                
                                # Classification loss
                                gt_class = gt_box[4].long().to(device)
                                class_loss += F.cross_entropy(pred_classes.unsqueeze(0), gt_class.unsqueeze(0))
                        else:
                            # No object
                            no_obj_target = torch.tensor(0.0, device=device, dtype=torch.float32)
                            no_obj_loss += F.binary_cross_entropy(pred_obj, no_obj_target)
                else:
                    # No objects in this image - all anchors should predict no object
                    for anchor_idx in range(3):
                        pred_obj = torch.sigmoid(outputs[b, anchor_idx, 4])
                        no_obj_target = torch.tensor(0.0, device=device, dtype=torch.float32)
                        no_obj_loss += F.binary_cross_entropy(pred_obj, no_obj_target)
            else:
                # No labels for this batch item
                for anchor_idx in range(3):
                    pred_obj = torch.sigmoid(outputs[b, anchor_idx, 4])
                    no_obj_target = torch.tensor(0.0, device=device, dtype=torch.float32)
                    no_obj_loss += F.binary_cross_entropy(pred_obj, no_obj_target)
        
        # Combine losses with weights
        total_loss = (
            5.0 * coord_loss +      # Coordinate loss weight
            1.0 * obj_loss +       # Object loss weight
            0.5 * no_obj_loss +    # No-object loss weight
            1.0 * class_loss       # Classification loss weight
        )
        
        return total_loss / batch_size
    
    def calculate_iou(self, pred_x, pred_y, pred_w, pred_h, gt_x, gt_y, gt_w, gt_h):
        """Calculate Intersection over Union between predicted and ground truth boxes"""
        # Convert center coordinates to corner coordinates
        pred_x1 = pred_x - pred_w / 2
        pred_y1 = pred_y - pred_h / 2
        pred_x2 = pred_x + pred_w / 2
        pred_y2 = pred_y + pred_h / 2
        
        gt_x1 = gt_x - gt_w / 2
        gt_y1 = gt_y - gt_h / 2
        gt_x2 = gt_x + gt_w / 2
        gt_y2 = gt_y + gt_h / 2
        
        # Calculate intersection
        inter_x1 = torch.max(pred_x1, gt_x1)
        inter_y1 = torch.max(pred_y1, gt_y1)
        inter_x2 = torch.min(pred_x2, gt_x2)
        inter_y2 = torch.min(pred_y2, gt_y2)
        
        inter_w = torch.clamp(inter_x2 - inter_x1, min=0)
        inter_h = torch.clamp(inter_y2 - inter_y1, min=0)
        inter_area = inter_w * inter_h
        
        # Calculate union
        pred_area = (pred_x2 - pred_x1) * (pred_y2 - pred_y1)
        gt_area = (gt_x2 - gt_x1) * (gt_y2 - gt_y1)
        union_area = pred_area + gt_area - inter_area
        
        # Calculate IoU
        iou = inter_area / (union_area + 1e-6)  # Add small epsilon to avoid division by zero
        
        return iou
    
    def train(self, num_epochs=50, save_dir='checkpoints'):
        """Train the model"""
        os.makedirs(save_dir, exist_ok=True)
        
        best_val_loss = float('inf')
        best_epoch = 0
        
        print(f"Starting training for {num_epochs} epochs...")
        print(f"Model parameters: {count_parameters(self.model):,}")
        print(f"Device: {self.device}")
        print(f"Classes: {self.class_names}")
        
        start_time = time.time()
        
        for epoch in range(num_epochs):
            print(f"\nEpoch {epoch+1}/{num_epochs}")
            print("-" * 50)
            
            # Train
            train_loss = self.train_epoch(epoch)
            
            # Validate
            val_loss = self.validate_epoch(epoch)
            
            # Update learning rate
            self.scheduler.step()
            current_lr = self.optimizer.param_groups[0]['lr']
            
            # Print epoch results
            print(f"Train Loss: {train_loss:.4f}")
            print(f"Val Loss: {val_loss:.4f}")
            print(f"Learning Rate: {current_lr:.6f}")
            
            # Save best model
            if val_loss < best_val_loss:
                best_val_loss = val_loss
                best_epoch = epoch
                
                # Save model
                checkpoint = {
                    'epoch': epoch,
                    'model_state_dict': self.model.state_dict(),
                    'optimizer_state_dict': self.optimizer.state_dict(),
                    'scheduler_state_dict': self.scheduler.state_dict(),
                    'val_loss': val_loss,
                    'class_names': self.class_names
                }
                
                torch.save(checkpoint, os.path.join(save_dir, 'best_model.pth'))
                print(f"New best model saved! Val Loss: {val_loss:.4f}")
            
            # Save checkpoint every 10 epochs
            if (epoch + 1) % 10 == 0:
                checkpoint = {
                    'epoch': epoch,
                    'model_state_dict': self.model.state_dict(),
                    'optimizer_state_dict': self.optimizer.state_dict(),
                    'scheduler_state_dict': self.scheduler.state_dict(),
                    'val_loss': val_loss,
                    'class_names': self.class_names
                }
                torch.save(checkpoint, os.path.join(save_dir, f'checkpoint_epoch_{epoch+1}.pth'))
        
        training_time = time.time() - start_time
        print(f"\nTraining completed in {training_time:.2f} seconds")
        print(f"Best validation loss: {best_val_loss:.4f} at epoch {best_epoch+1}")
        
        # Plot training curves
        self.plot_training_curves()
        
        return best_val_loss
    
    def plot_training_curves(self):
        """Plot training and validation curves"""
        plt.figure(figsize=(12, 4))
        
        # Loss curves
        plt.subplot(1, 2, 1)
        plt.plot(self.train_losses, label='Train Loss', color='blue')
        plt.plot(self.val_losses, label='Val Loss', color='red')
        plt.xlabel('Epoch')
        plt.ylabel('Loss')
        plt.title('Training and Validation Loss')
        plt.legend()
        plt.grid(True)
        
        # Learning rate
        plt.subplot(1, 2, 2)
        lrs = [group['lr'] for group in self.optimizer.param_groups]
        plt.plot(lrs, label='Learning Rate', color='green')
        plt.xlabel('Epoch')
        plt.ylabel('Learning Rate')
        plt.title('Learning Rate Schedule')
        plt.legend()
        plt.grid(True)
        
        plt.tight_layout()
        plt.savefig('training_curves.png', dpi=300, bbox_inches='tight')
        plt.show()
        print("Training curves saved as 'training_curves.png'")

class YOLOLoss(nn.Module):
    """
    Proper YOLO loss function for object detection
    """
    
    def __init__(self, num_classes=4):
        super(YOLOLoss, self).__init__()
        self.num_classes = num_classes
    
    def forward(self, predictions, targets):
        """
        Compute YOLO loss
        Our model outputs shape: (B, 3, 5 + num_classes)
        """
        # Use the same loss computation as in the trainer
        # This is a simplified version - in practice, you'd implement the full YOLO loss here
        batch_size = predictions.shape[0]
        device = predictions.device
        
        # Simple MSE loss for now - the trainer has the proper implementation
        total_loss = 0.0
        
        for b in range(batch_size):
            # Use a small regularization loss
            loss = F.mse_loss(predictions[b], torch.zeros_like(predictions[b])) * 0.1
            total_loss += loss
        
        return total_loss / batch_size

def main():
    parser = argparse.ArgumentParser(description='Train YOLO for RF Signal Detection')
    parser.add_argument('--data_dir', type=str, default='../spectrogram_training_data_20220711',
                       help='Path to dataset directory')
    parser.add_argument('--batch_size', type=int, default=16,
                       help='Batch size for training')
    parser.add_argument('--epochs', type=int, default=50,
                       help='Number of training epochs')
    parser.add_argument('--img_size', type=int, default=640,
                       help='Input image size')
    parser.add_argument('--device', type=str, default='cuda',
                       help='Device to use (cuda/cpu)')
    
    args = parser.parse_args()
    
    print("=== YOLO RF Signal Detection Training ===")
    print(f"Data directory: {args.data_dir}")
    print(f"Batch size: {args.batch_size}")
    print(f"Epochs: {args.epochs}")
    print(f"Image size: {args.img_size}")
    print(f"Device: {args.device}")
    
    # Analyze dataset
    print("\nAnalyzing dataset...")
    stats = analyze_dataset(args.data_dir)
    
    # Create data loaders
    print("\nCreating data loaders...")
    train_loader, val_loader, class_names = create_yolo_data_loaders(
        args.data_dir, 
        batch_size=args.batch_size, 
        test_size=0.2, 
        img_size=args.img_size
    )
    
    # Create model
    print("\nCreating YOLO model...")
    model = create_yolo_model(num_classes=len(class_names), img_size=args.img_size)
    
    # Create trainer
    trainer = YOLOTrainer(model, train_loader, val_loader, class_names, device=args.device)
    
    # Train model
    print("\nStarting training...")
    best_val_loss = trainer.train(num_epochs=args.epochs)
    
    print(f"\nTraining completed!")
    print(f"Best validation loss: {best_val_loss:.4f}")
    print(f"Model saved in 'checkpoints/best_model.pth'")

if __name__ == "__main__":
    main()
