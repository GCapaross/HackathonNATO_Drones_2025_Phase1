#!/usr/bin/env python3
"""
YOLO Single Drone Spectrogram Classifier - Training Script
===========================================================

Train a YOLO model to classify individual drone spectrograms.
Each spectrogram = ONE drone signal.

PREREQUISITES:
Run generate_yolo_dataset.py first to create the YOLO dataset!

This script trains YOLO on pre-generated spectrogram images.
"""

import numpy as np
import matplotlib.pyplot as plt
from pathlib import Path
from ultralytics import YOLO
from sklearn.metrics import confusion_matrix, classification_report, accuracy_score
import seaborn as sns

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


def train_yolo_model(yaml_path: Path, output_dir: Path, epochs: int = 20):
    """Train YOLO model on single drone spectrograms."""
    print("\n" + "=" * 70)
    print("Training YOLO Model")
    print("=" * 70)
    
    # Initialize YOLO model (using YOLOv8 nano for speed)
    model = YOLO('yolo11n-cls.pt')  # Classification model
    
    # Train
    results = model.train(
        data=str(yaml_path),
        epochs=epochs,
        imgsz=640,
        batch=32,
        project=str(output_dir),
        name='yolo_single_drone',
        verbose=True,
        plots=True
    )
    
    return model, results


def evaluate_yolo_model(model, test_dir: Path, output_dir: Path):
    """Evaluate YOLO model and create confusion matrix."""
    print("\n" + "=" * 70)
    print("Evaluating YOLO Model")
    print("=" * 70)
    
    test_images = list((test_dir / 'test' / 'images').glob('*.jpg'))
    test_labels_dir = test_dir / 'test' / 'labels'
    
    y_true = []
    y_pred = []
    
    print(f"Evaluating on {len(test_images)} test samples...")
    
    for img_path in test_images:
        # Get true label
        label_path = test_labels_dir / f"{img_path.stem}.txt"
        if label_path.exists():
            with open(label_path, 'r') as f:
                true_class = int(f.readline().split()[0])
            
            # Predict
            results = model.predict(str(img_path), verbose=False)
            
            # Get predicted class (YOLO classification returns top class)
            if hasattr(results[0], 'probs'):
                pred_class = results[0].probs.top1
                y_true.append(true_class)
                y_pred.append(pred_class)
    
    # Calculate metrics
    y_true = np.array(y_true)
    y_pred = np.array(y_pred)
    
    accuracy = accuracy_score(y_true, y_pred)
    print(f"\nTest Accuracy: {accuracy*100:.2f}%")
    
    # Classification report
    print("\nClassification Report:")
    print(classification_report(y_true, y_pred, 
                                target_names=[f"Class {i}" for i in range(24)],
                                zero_division=0))
    
    # Per-class accuracy
    print("\nPer-Class Accuracy:")
    for i in range(24):
        mask = y_true == i
        if np.sum(mask) > 0:
            class_acc = accuracy_score(y_true[mask], y_pred[mask])
            print(f"  Class {i:2d} ({CLASS_NAMES[i]:25s}): {class_acc*100:.2f}%")
    
    # Plot confusion matrix
    plot_confusion_matrix(y_true, y_pred, output_dir)
    
    return accuracy, y_true, y_pred


def plot_confusion_matrix(y_true, y_pred, output_dir):
    """Plot confusion matrix."""
    cm = confusion_matrix(y_true, y_pred)
    
    plt.figure(figsize=(16, 14))
    sns.heatmap(cm, annot=True, fmt='d', cmap='Greens',
                xticklabels=[f"{i}" for i in range(24)],
                yticklabels=[f"{i}" for i in range(24)])
    plt.xlabel('Predicted Class')
    plt.ylabel('True Class')
    plt.title('YOLO Single Drone Classification - Confusion Matrix')
    plt.tight_layout()
    plt.savefig(output_dir / 'yolo_confusion_matrix.png', dpi=150, bbox_inches='tight')
    print(f"\nSaved confusion matrix to {output_dir / 'yolo_confusion_matrix.png'}")
    plt.close()


def main():
    """Main training pipeline."""
    # Paths
    yolo_dataset_dir = Path("YOLO_Dataset")
    output_dir = Path("YOLO_Results")
    output_dir.mkdir(parents=True, exist_ok=True)
    
    # Check if dataset exists
    yaml_path = yolo_dataset_dir / 'dataset.yaml'
    if not yaml_path.exists():
        print("=" * 70)
        print("ERROR: YOLO dataset not found!")
        print("=" * 70)
        print(f"Expected location: {yaml_path}")
        print("\nPlease generate the dataset first:")
        print("  python3 generate_yolo_dataset.py")
        print("=" * 70)
        return 1
    
    print("=" * 70)
    print("YOLO Single Drone Classifier Training")
    print("=" * 70)
    print(f"Dataset: {yolo_dataset_dir}")
    print(f"Config: {yaml_path}")
    print()
    
    # Step 1: Train YOLO model
    print("Starting YOLO training...")
    model, results = train_yolo_model(yaml_path, output_dir, epochs=20)
    
    # Step 2: Evaluate
    print("\nEvaluating YOLO model...")
    accuracy, y_true, y_pred = evaluate_yolo_model(model, yolo_dataset_dir, output_dir)
    
    print("\n" + "=" * 70)
    print("YOLO Training Complete!")
    print("=" * 70)
    print(f"Final Test Accuracy: {accuracy*100:.2f}%")
    print(f"Results saved to: {output_dir}")
    print("=" * 70)
    
    return 0


if __name__ == "__main__":
    main()

