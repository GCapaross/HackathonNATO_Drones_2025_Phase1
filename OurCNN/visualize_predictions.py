"""
Visualize model predictions on spectrogram images
Saves 10 examples showing ground truth vs model predictions
"""

import torch
import torch.nn as nn
from torch.utils.data import DataLoader
import matplotlib.pyplot as plt
import numpy as np
import os
import random
from PIL import Image
import argparse

# Import our modules
from data_loader import create_data_loaders
from cnn_model import create_model

class PredictionVisualizer:
    def __init__(self, model_path, data_dir, device='cuda'):
        self.device = torch.device(device if torch.cuda.is_available() else 'cpu')
        self.data_dir = data_dir
        
        # Load data
        self.train_loader, self.val_loader, self.class_names = create_data_loaders(
            data_dir, batch_size=32, test_size=0.2, task='classification'
        )
        
        # Load model
        self.model = create_model(num_classes=len(self.class_names))[0]
        
        # Load checkpoint (handle both model-only and full checkpoint formats)
        checkpoint = torch.load(model_path, map_location=self.device)
        
        if 'model_state_dict' in checkpoint:
            # Full checkpoint format
            self.model.load_state_dict(checkpoint['model_state_dict'])
            print(f"Loaded checkpoint from epoch {checkpoint.get('epoch', 'unknown')}")
        else:
            # Model-only format
            self.model.load_state_dict(checkpoint)
            print("Loaded model weights")
        
        self.model.to(self.device)
        self.model.eval()
        
        print(f"Model loaded from: {model_path}")
        print(f"Classes: {self.class_names}")
        print(f"Device: {self.device}")

    def get_sample_predictions(self, num_samples=10):
        """Get sample predictions from validation set"""
        samples = []
        
        with torch.no_grad():
            for batch_idx, (images, labels) in enumerate(self.val_loader):
                images, labels = images.to(self.device), labels.to(self.device)
                
                # Get predictions
                outputs = self.model(images)
                probabilities = torch.softmax(outputs, dim=1)
                _, predicted = torch.max(outputs, 1)
                
                # Convert to CPU for processing
                images_cpu = images.cpu()
                labels_cpu = labels.cpu()
                predicted_cpu = predicted.cpu()
                probabilities_cpu = probabilities.cpu()
                
                # Store samples with additional info
                for i in range(len(images_cpu)):
                    sample = {
                        'image': images_cpu[i],
                        'true_label': labels_cpu[i].item(),
                        'predicted_label': predicted_cpu[i].item(),
                        'probabilities': probabilities_cpu[i].numpy(),
                        'confidence': probabilities_cpu[i].max().item(),
                        'batch_idx': batch_idx,
                        'sample_idx': i
                    }
                    samples.append(sample)
                    
                    if len(samples) >= num_samples:
                        break
                
                if len(samples) >= num_samples:
                    break
        
        return samples

    def create_visualization(self, sample, index):
        """Create visualization for a single sample"""
        # Denormalize image for display
        image = sample['image']
        
        # Denormalize using ImageNet stats
        mean = torch.tensor([0.485, 0.456, 0.406])
        std = torch.tensor([0.229, 0.224, 0.225])
        image = image * std.view(3, 1, 1) + mean.view(3, 1, 1)
        image = torch.clamp(image, 0, 1)
        
        # Convert to numpy
        image_np = image.permute(1, 2, 0).numpy()
        
        # Get labels
        true_class = self.class_names[sample['true_label']]
        pred_class = self.class_names[sample['predicted_label']]
        confidence = sample['confidence']
        probabilities = sample['probabilities']
        
        # Create figure with 3 subplots
        fig, (ax1, ax2, ax3) = plt.subplots(1, 3, figsize=(20, 6))
        
        # Plot original spectrogram
        ax1.imshow(image_np)
        ax1.set_title(f'Original Spectrogram {index + 1}', fontsize=14, fontweight='bold')
        ax1.axis('off')
        
        # Add ground truth and prediction text
        ax1.text(0.02, 0.98, f'Ground Truth: {true_class}', 
                transform=ax1.transAxes, fontsize=12, fontweight='bold',
                bbox=dict(boxstyle='round', facecolor='lightblue', alpha=0.8),
                verticalalignment='top')
        
        color = 'lightgreen' if sample['true_label'] == sample['predicted_label'] else 'lightcoral'
        ax1.text(0.02, 0.88, f'Predicted: {pred_class}', 
                transform=ax1.transAxes, fontsize=12, fontweight='bold',
                bbox=dict(boxstyle='round', facecolor=color, alpha=0.8),
                verticalalignment='top')
        
        ax1.text(0.02, 0.78, f'Confidence: {confidence:.3f}', 
                transform=ax1.transAxes, fontsize=11,
                bbox=dict(boxstyle='round', facecolor='lightyellow', alpha=0.8),
                verticalalignment='top')
        
        # Plot spectrogram with YOLO labels overlaid
        ax2.imshow(image_np)
        ax2.set_title('Ground Truth Labels', fontsize=14, fontweight='bold')
        ax2.axis('off')
        
        # Show the ground truth class information
        ax2.text(0.5, 0.5, f'True Class: {true_class}\n\nThis is what the model\nshould have predicted', 
                transform=ax2.transAxes, fontsize=12, ha='center', va='center',
                bbox=dict(boxstyle='round', facecolor='lightblue', alpha=0.8))
        
        # Plot probability distribution
        bars = ax3.bar(self.class_names, probabilities, color=['lightblue', 'lightgreen', 'lightcoral', 'lightyellow'])
        ax3.set_title('Model Predictions', fontsize=14, fontweight='bold')
        ax3.set_ylabel('Probability')
        ax3.set_ylim(0, 1)
        
        # Highlight true and predicted classes
        bars[sample['true_label']].set_color('blue')
        bars[sample['predicted_label']].set_color('red')
        
        # Add value labels on bars
        for i, (bar, prob) in enumerate(zip(bars, probabilities)):
            height = bar.get_height()
            ax3.text(bar.get_x() + bar.get_width()/2, height + 0.01, 
                    f'{prob:.3f}', ha='center', va='bottom', fontsize=10)
        
        # Rotate x-axis labels
        ax3.tick_params(axis='x', rotation=45)
        
        # Add legend
        ax3.legend(['True Class', 'Predicted Class'], loc='upper right')
        
        plt.tight_layout()
        return fig

    def save_visualizations(self, num_samples=10, output_dir='predictions_visualization'):
        """Save visualization images"""
        # Create output directory
        os.makedirs(output_dir, exist_ok=True)
        
        print(f"Getting {num_samples} sample predictions...")
        samples = self.get_sample_predictions(num_samples)
        
        print(f"Creating visualizations...")
        for i, sample in enumerate(samples):
            print(f"Processing sample {i+1}/{len(samples)}")
            
            # Create visualization
            fig = self.create_visualization(sample, i)
            
            # Save figure
            output_path = os.path.join(output_dir, f'prediction_{i+1:02d}.png')
            fig.savefig(output_path, dpi=300, bbox_inches='tight')
            plt.close(fig)
            
            # Print sample info
            true_class = self.class_names[sample['true_label']]
            pred_class = self.class_names[sample['predicted_label']]
            correct = "✓" if sample['true_label'] == sample['predicted_label'] else "✗"
            
            print(f"  Sample {i+1}: {true_class} → {pred_class} {correct} (conf: {sample['confidence']:.3f})")
        
        print(f"\nVisualizations saved to: {output_dir}/")
        print(f"Files created:")
        for i in range(num_samples):
            print(f"  - prediction_{i+1:02d}.png")
        
        # Create summary
        self.create_summary(samples, output_dir)
        
        return samples

    def create_summary(self, samples, output_dir):
        """Create a summary of all predictions"""
        correct_predictions = sum(1 for s in samples if s['true_label'] == s['predicted_label'])
        accuracy = correct_predictions / len(samples)
        
        # Class-wise accuracy
        class_correct = {i: 0 for i in range(len(self.class_names))}
        class_total = {i: 0 for i in range(len(self.class_names))}
        
        for sample in samples:
            true_label = sample['true_label']
            class_total[true_label] += 1
            if sample['true_label'] == sample['predicted_label']:
                class_correct[true_label] += 1
        
        # Create summary text
        summary_text = f"""
PREDICTION VISUALIZATION SUMMARY
================================

Total Samples: {len(samples)}
Correct Predictions: {correct_predictions}
Overall Accuracy: {accuracy:.3f}

Per-Class Accuracy:
"""
        for i, class_name in enumerate(self.class_names):
            if class_total[i] > 0:
                class_acc = class_correct[i] / class_total[i]
                summary_text += f"  {class_name}: {class_correct[i]}/{class_total[i]} ({class_acc:.3f})\n"
            else:
                summary_text += f"  {class_name}: 0/0 (N/A)\n"
        
        summary_text += f"""
Sample Details:
"""
        for i, sample in enumerate(samples):
            true_class = self.class_names[sample['true_label']]
            pred_class = self.class_names[sample['predicted_label']]
            correct = "✓" if sample['true_label'] == sample['predicted_label'] else "✗"
            summary_text += f"  {i+1:2d}. {true_class} → {pred_class} {correct} (conf: {sample['confidence']:.3f})\n"
        
        # Save summary
        summary_path = os.path.join(output_dir, 'summary.txt')
        with open(summary_path, 'w') as f:
            f.write(summary_text)
        
        print(f"Summary saved to: {summary_path}")

def main():
    parser = argparse.ArgumentParser(description='Visualize model predictions on spectrograms')
    parser.add_argument('--model_path', type=str, default='checkpoints/best_model.pth',
                        help='Path to trained model')
    parser.add_argument('--data_dir', type=str, 
                        default='/home/gabriel/Desktop/HackathonNATO_Drones_2025/spectrogram_training_data_20220711',
                        help='Path to dataset directory')
    parser.add_argument('--num_samples', type=int, default=10,
                        help='Number of samples to visualize')
    parser.add_argument('--output_dir', type=str, default='predictions_visualization',
                        help='Output directory for visualizations')
    
    args = parser.parse_args()
    
    # Check if model exists
    if not os.path.exists(args.model_path):
        print(f"Error: Model file '{args.model_path}' not found!")
        print("Make sure you've trained a model first with: python train_cnn.py")
        return
    
    # Create visualizer
    visualizer = PredictionVisualizer(args.model_path, args.data_dir)
    
    # Save visualizations
    samples = visualizer.save_visualizations(args.num_samples, args.output_dir)
    
    print(f"\nVisualization complete!")
    print(f"Check the '{args.output_dir}' folder for results.")

if __name__ == "__main__":
    main()
