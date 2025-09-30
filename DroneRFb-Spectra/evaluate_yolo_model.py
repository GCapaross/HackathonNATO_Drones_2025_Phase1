#!/usr/bin/env python3
"""
Evaluate an existing YOLO model - no training, just testing.
"""

import numpy as np
import matplotlib.pyplot as plt
from pathlib import Path
from ultralytics import YOLO
from sklearn.metrics import confusion_matrix, classification_report, accuracy_score
import seaborn as sns
import argparse

# Class labels
CLASS_NAMES = [
    "Background",
    "DJI_Phantom_3",
    "DJI_Phantom_4_Pro",
    "DJI_MATRICE_200",
    "DJI_MATRICE_100",
    "DJI_Air_2S",
    "DJI_Mini_3_Pro",
    "DJI_Inspire_2",
    "DJI_Mavic_Pro",
    "DJI_Mini_2",
    "DJI_Mavic_3",
    "DJI_MATRICE_300",
    "DJI_Phantom_4_Pro_RTK",
    "DJI_MATRICE_30T",
    "DJI_AVATA",
    "DJI_DIY",
    "DJI_MATRICE_600_Pro",
    "VBar_Controller",
    "FrSky_X20",
    "Futaba_T16IZ",
    "Taranis_Plus",
    "RadioLink_AT9S",
    "Futaba_T14SG",
    "Skydroid"
]


def evaluate_yolo_model(model_path: Path, test_dir: Path, output_dir: Path):
    """Evaluate YOLO model and create confusion matrix."""
    print("\n" + "=" * 70)
    print("Evaluating YOLO Model")
    print("=" * 70)
    print(f"Model: {model_path}")
    print(f"Test data: {test_dir}")
    
    # Load model
    model = YOLO(str(model_path))
    
    test_images = list((test_dir / 'images').glob('*.jpg'))
    test_labels_dir = test_dir / 'labels'
    
    y_true = []
    y_pred = []
    
    print(f"\nEvaluating on {len(test_images)} test samples...")
    
    for i, img_path in enumerate(test_images):
        if (i + 1) % 100 == 0:
            print(f"  Processed {i+1}/{len(test_images)} images...")
        
        # Get true label
        label_path = test_labels_dir / f"{img_path.stem}.txt"
        if label_path.exists():
            with open(label_path, 'r') as f:
                true_class = int(f.readline().split()[0])
            
            # Predict
            results = model.predict(str(img_path), verbose=False)
            
            # Get predicted class from detection results
            # For detection model: results[0].boxes.cls contains the class IDs
            if results[0].boxes is not None and len(results[0].boxes) > 0:
                # Get the class with highest confidence
                pred_class = int(results[0].boxes.cls[0].cpu().numpy())
                y_true.append(true_class)
                y_pred.append(pred_class)
            else:
                # No detection made - treat as misclassification
                print(f"  Warning: No detection for {img_path.name}, true class: {CLASS_NAMES[true_class]}")
                y_true.append(true_class)
                y_pred.append(-1)  # Mark as "no detection"
    
    # Calculate metrics
    y_true = np.array(y_true)
    y_pred = np.array(y_pred)
    
    # Filter out "no detection" cases for accuracy calculation
    valid_mask = y_pred != -1
    if np.sum(valid_mask) > 0:
        accuracy = accuracy_score(y_true[valid_mask], y_pred[valid_mask])
        print(f"\n" + "=" * 70)
        print(f"Test Accuracy: {accuracy*100:.2f}%")
        print(f"Valid predictions: {np.sum(valid_mask)}/{len(y_pred)}")
        print("=" * 70)
        
        # Classification report
        print("\nClassification Report:")
        print(classification_report(y_true[valid_mask], y_pred[valid_mask], 
                                    target_names=CLASS_NAMES,
                                    zero_division=0))
        
        # Per-class accuracy
        print("\n" + "=" * 70)
        print("Per-Class Accuracy:")
        print("=" * 70)
        for i in range(24):
            mask = (y_true == i) & valid_mask
            if np.sum(mask) > 0:
                class_acc = accuracy_score(y_true[mask], y_pred[mask])
                total = np.sum(y_true == i)
                correct = np.sum((y_true == i) & (y_pred == i))
                print(f"  {i:2d}. {CLASS_NAMES[i]:30s}: {class_acc*100:6.2f}%  ({correct:3d}/{total:3d})")
        
        # Plot confusion matrix
        plot_confusion_matrix(y_true[valid_mask], y_pred[valid_mask], output_dir)
        
        return accuracy, y_true, y_pred
    else:
        print("ERROR: No valid predictions made!")
        return 0.0, y_true, y_pred


def plot_confusion_matrix(y_true, y_pred, output_dir):
    """Plot confusion matrix."""
    cm = confusion_matrix(y_true, y_pred)
    
    plt.figure(figsize=(20, 18))
    sns.heatmap(cm, annot=True, fmt='d', cmap='Greens',
                xticklabels=[f"{i}" for i in range(24)],
                yticklabels=[f"{i}" for i in range(24)])
    plt.xlabel('Predicted Class', fontsize=12)
    plt.ylabel('True Class', fontsize=12)
    plt.title('YOLO Single Drone Classification - Confusion Matrix', fontsize=14)
    plt.tight_layout()
    
    output_path = output_dir / 'evaluation_confusion_matrix.png'
    plt.savefig(output_path, dpi=150, bbox_inches='tight')
    print(f"\nSaved confusion matrix to {output_path}")
    plt.close()
    
    # Also create a per-class accuracy plot
    plt.figure(figsize=(16, 8))
    class_accuracies = []
    for i in range(24):
        mask = y_true == i
        if np.sum(mask) > 0:
            class_acc = accuracy_score(y_true[mask], y_pred[mask])
            class_accuracies.append(class_acc * 100)
        else:
            class_accuracies.append(0)
    
    plt.bar(range(24), class_accuracies, color='green', alpha=0.7)
    plt.axhline(y=100, color='r', linestyle='--', label='100% Accuracy')
    plt.xlabel('Class ID', fontsize=12)
    plt.ylabel('Accuracy (%)', fontsize=12)
    plt.title('Per-Class Accuracy', fontsize=14)
    plt.xticks(range(24), range(24))
    plt.ylim([0, 105])
    plt.legend()
    plt.grid(axis='y', alpha=0.3)
    plt.tight_layout()
    
    output_path = output_dir / 'evaluation_per_class_accuracy.png'
    plt.savefig(output_path, dpi=150, bbox_inches='tight')
    print(f"Saved per-class accuracy to {output_path}")
    plt.close()


def main():
    """Main evaluation function."""
    parser = argparse.ArgumentParser(description="Evaluate YOLO model on test set")
    parser.add_argument("--model", type=str, 
                       default="YOLO_Results/yolo_single_drone5/weights/best.pt",
                       help="Path to trained YOLO model")
    parser.add_argument("--test_dir", type=str,
                       default="YOLO_Dataset/test",
                       help="Path to test dataset directory")
    parser.add_argument("--output_dir", type=str,
                       default="YOLO_Results/evaluation",
                       help="Output directory for evaluation results")
    
    args = parser.parse_args()
    
    # Convert to Path objects
    model_path = Path(args.model)
    test_dir = Path(args.test_dir)
    output_dir = Path(args.output_dir)
    
    # Create output directory
    output_dir.mkdir(parents=True, exist_ok=True)
    
    # Check if model exists
    if not model_path.exists():
        print(f"ERROR: Model not found: {model_path}")
        print("\nAvailable models:")
        results_dir = Path("YOLO_Results")
        if results_dir.exists():
            for model_file in results_dir.rglob("best.pt"):
                print(f"  - {model_file}")
        return
    
    # Check if test directory exists
    if not test_dir.exists():
        print(f"ERROR: Test directory not found: {test_dir}")
        return
    
    # Evaluate
    print("\n" + "=" * 70)
    print("YOLO Model Evaluation")
    print("=" * 70)
    
    accuracy, y_true, y_pred = evaluate_yolo_model(model_path, test_dir, output_dir)
    
    print("\n" + "=" * 70)
    print("Evaluation Complete!")
    print("=" * 70)
    print(f"Final Accuracy: {accuracy*100:.2f}%")
    print(f"Results saved to: {output_dir}")
    print("=" * 70)


if __name__ == "__main__":
    main()

