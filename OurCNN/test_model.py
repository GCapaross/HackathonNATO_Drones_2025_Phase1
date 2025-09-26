"""
Test script for trained CNN model
Tests the model on validation data and provides detailed performance metrics
"""

import torch
import torch.nn as nn
from torch.utils.data import DataLoader
import matplotlib.pyplot as plt
import seaborn as sns
import numpy as np
from sklearn.metrics import confusion_matrix, classification_report, accuracy_score
import argparse
import os
from PIL import Image
import random

# Import our modules
from data_loader import create_data_loaders
from cnn_model import create_model

class ModelTester:
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
            print(f"Best validation accuracy: {checkpoint.get('val_acc', 'unknown')}")
        else:
            # Model-only format
            self.model.load_state_dict(checkpoint)
            print("Loaded model weights")
        
        self.model.to(self.device)
        self.model.eval()
        
        print(f"Model loaded from: {model_path}")
        print(f"Classes: {self.class_names}")
        print(f"Device: {self.device}")

    def test_validation_set(self):
        """Test model on validation set"""
        print("\n=== Testing on Validation Set ===")
        
        all_predictions = []
        all_labels = []
        all_probabilities = []
        
        with torch.no_grad():
            for batch_idx, (images, labels) in enumerate(self.val_loader):
                images, labels = images.to(self.device), labels.to(self.device)
                
                # Get predictions
                outputs = self.model(images)
                probabilities = torch.softmax(outputs, dim=1)
                _, predicted = torch.max(outputs, 1)
                
                all_predictions.extend(predicted.cpu().numpy())
                all_labels.extend(labels.cpu().numpy())
                all_probabilities.extend(probabilities.cpu().numpy())
                
                if batch_idx % 50 == 0:
                    print(f"Processed batch {batch_idx}/{len(self.val_loader)}")
        
        # Calculate metrics
        accuracy = accuracy_score(all_labels, all_predictions)
        print(f"\nOverall Accuracy: {accuracy:.4f}")
        
        # Classification report
        print("\n=== Classification Report ===")
        # Get unique classes present in predictions and labels
        unique_labels = sorted(list(set(all_labels + all_predictions)))
        target_names_subset = [self.class_names[i] for i in unique_labels]
        
        print(classification_report(all_labels, all_predictions, 
                                   target_names=target_names_subset,
                                   labels=unique_labels))
        
        report = classification_report(all_labels, all_predictions, 
                                     target_names=target_names_subset,
                                     labels=unique_labels,
                                     output_dict=True)
        
        # Confusion matrix
        self.plot_confusion_matrix(all_labels, all_predictions)
        
        # Per-class accuracy
        self.plot_per_class_accuracy(all_labels, all_predictions)
        
        return all_predictions, all_labels, all_probabilities

    def test_single_image(self, image_path):
        """Test model on a single image"""
        print(f"\n=== Testing Single Image: {image_path} ===")
        
        # Load and preprocess image
        image = Image.open(image_path).convert('RGB')
        
        # Apply same transforms as validation
        from torchvision import transforms
        transform = transforms.Compose([
            transforms.Resize((224, 224)),
            transforms.ToTensor(),
            transforms.Normalize(mean=[0.485, 0.456, 0.406], std=[0.229, 0.224, 0.225])
        ])
        
        image_tensor = transform(image).unsqueeze(0).to(self.device)
        
        with torch.no_grad():
            outputs = self.model(image_tensor)
            probabilities = torch.softmax(outputs, dim=1)
            predicted_class = torch.argmax(outputs, 1).item()
            confidence = probabilities[0][predicted_class].item()
        
        print(f"Predicted Class: {self.class_names[predicted_class]} (ID: {predicted_class})")
        print(f"Confidence: {confidence:.4f}")
        print("\nClass Probabilities:")
        for i, (class_name, prob) in enumerate(zip(self.class_names, probabilities[0])):
            print(f"  {class_name}: {prob:.4f}")
        
        return predicted_class, confidence, probabilities[0].cpu().numpy()

    def test_random_samples(self, num_samples=10):
        """Test model on random samples from validation set"""
        print(f"\n=== Testing {num_samples} Random Samples ===")
        
        # Get random samples
        all_images = []
        all_labels = []
        
        for images, labels in self.val_loader:
            all_images.extend(images)
            all_labels.extend(labels)
            if len(all_images) >= num_samples * 2:  # Get extra to have good selection
                break
        
        # Select random samples
        indices = random.sample(range(len(all_images)), min(num_samples, len(all_images)))
        
        correct_predictions = 0
        total_predictions = 0
        
        for i, idx in enumerate(indices):
            image = all_images[idx].unsqueeze(0).to(self.device)
            true_label = all_labels[idx].item()
            
            with torch.no_grad():
                outputs = self.model(image)
                probabilities = torch.softmax(outputs, dim=1)
                predicted_class = torch.argmax(outputs, 1).item()
                confidence = probabilities[0][predicted_class].item()
            
            is_correct = predicted_class == true_label
            if is_correct:
                correct_predictions += 1
            total_predictions += 1
            
            print(f"\nSample {i+1}:")
            print(f"  True: {self.class_names[true_label]} (ID: {true_label})")
            print(f"  Predicted: {self.class_names[predicted_class]} (ID: {predicted_class})")
            print(f"  Confidence: {confidence:.4f}")
            print(f"  Correct: {'✓' if is_correct else '✗'}")
        
        accuracy = correct_predictions / total_predictions
        print(f"\nRandom Sample Accuracy: {accuracy:.4f} ({correct_predictions}/{total_predictions})")
        
        return accuracy

    def plot_confusion_matrix(self, true_labels, predictions):
        """Plot confusion matrix"""
        cm = confusion_matrix(true_labels, predictions)
        
        plt.figure(figsize=(10, 8))
        sns.heatmap(cm, annot=True, fmt='d', cmap='Blues', 
                   xticklabels=self.class_names, 
                   yticklabels=self.class_names)
        plt.title('Confusion Matrix')
        plt.xlabel('Predicted')
        plt.ylabel('True')
        plt.tight_layout()
        plt.savefig('./confusion_matrix_test.png', dpi=300, bbox_inches='tight')
        plt.show()
        print("Confusion matrix saved as 'confusion_matrix_test.png'")

    def plot_per_class_accuracy(self, true_labels, predictions):
        """Plot per-class accuracy"""
        cm = confusion_matrix(true_labels, predictions)
        per_class_accuracy = cm.diagonal() / cm.sum(axis=1)
        
        # Get unique classes present in the data
        unique_labels = sorted(list(set(true_labels + predictions)))
        class_names_subset = [self.class_names[i] for i in unique_labels]
        
        plt.figure(figsize=(10, 6))
        bars = plt.bar(class_names_subset, per_class_accuracy)
        plt.title('Per-Class Accuracy')
        plt.xlabel('Class')
        plt.ylabel('Accuracy')
        plt.ylim(0, 1)
        
        # Add value labels on bars
        for bar, acc in zip(bars, per_class_accuracy):
            plt.text(bar.get_x() + bar.get_width()/2, bar.get_height() + 0.01, 
                    f'{acc:.3f}', ha='center', va='bottom')
        
        plt.xticks(rotation=45)
        plt.tight_layout()
        plt.savefig('./per_class_accuracy.png', dpi=300, bbox_inches='tight')
        plt.show()
        print("Per-class accuracy plot saved as 'per_class_accuracy.png'")

    def analyze_predictions(self, predictions, labels, probabilities):
        """Analyze prediction patterns"""
        print("\n=== Prediction Analysis ===")
        
        # Class distribution in predictions vs true labels
        pred_counts = np.bincount(predictions, minlength=len(self.class_names))
        true_counts = np.bincount(labels, minlength=len(self.class_names))
        
        print("Class Distribution:")
        print("Class\t\tTrue\tPredicted\tDifference")
        for i, class_name in enumerate(self.class_names):
            diff = pred_counts[i] - true_counts[i]
            print(f"{class_name:<15}\t{true_counts[i]}\t{pred_counts[i]}\t\t{diff:+d}")
        
        # Confidence analysis
        confidences = [max(probs) for probs in probabilities]
        print(f"\nConfidence Statistics:")
        print(f"  Mean confidence: {np.mean(confidences):.4f}")
        print(f"  Std confidence: {np.std(confidences):.4f}")
        print(f"  Min confidence: {np.min(confidences):.4f}")
        print(f"  Max confidence: {np.max(confidences):.4f}")

def main():
    parser = argparse.ArgumentParser(description='Test trained CNN model')
    parser.add_argument('--model_path', type=str, default='best_model.pth',
                        help='Path to trained model')
    parser.add_argument('--data_dir', type=str, 
                        default='/home/gabriel/Desktop/HackathonNATO_Drones_2025/spectrogram_training_data_20220711',
                        help='Path to dataset directory')
    parser.add_argument('--test_single', type=str, default=None,
                        help='Path to single image to test')
    parser.add_argument('--random_samples', type=int, default=10,
                        help='Number of random samples to test')
    
    args = parser.parse_args()
    
    # Check if model exists
    if not os.path.exists(args.model_path):
        print(f"Error: Model file '{args.model_path}' not found!")
        print("Make sure you've trained a model first with: python train_cnn.py")
        return
    
    # Create tester
    tester = ModelTester(args.model_path, args.data_dir)
    
    # Test validation set
    predictions, labels, probabilities = tester.test_validation_set()
    
    # Test random samples
    tester.test_random_samples(args.random_samples)
    
    # Analyze predictions
    tester.analyze_predictions(predictions, labels, probabilities)
    
    # Test single image if provided
    if args.test_single and os.path.exists(args.test_single):
        tester.test_single_image(args.test_single)
    
    print("\n=== Testing Complete ===")
    print("Generated files:")
    print("  - confusion_matrix_test.png")
    print("  - per_class_accuracy.png")

if __name__ == "__main__":
    main()
