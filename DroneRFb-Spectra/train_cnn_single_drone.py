#!/usr/bin/env python3
"""
Single Drone Spectrogram Classifier
====================================

Train a CNN to classify individual drone spectrograms.
Each spectrogram = ONE drone signal.

Based on the paper: "Combined RF-Based Drone Detection and Classification"
IEEE TRANSACTIONS ON COGNITIVE COMMUNICATIONS AND NETWORKING, 2022
"""

import os
import numpy as np
import matplotlib.pyplot as plt
from pathlib import Path
from tensorflow import keras
from keras.models import Sequential
from keras.layers import Conv2D, MaxPooling2D, Dense, Flatten, Dropout, BatchNormalization
from keras.callbacks import EarlyStopping, ModelCheckpoint, ReduceLROnPlateau
from keras.utils import to_categorical
from sklearn.model_selection import train_test_split
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


def load_dataset(data_dir: Path, target_size: tuple = (256, 256), 
                max_samples_per_class: int = None):
    """
    Load individual drone spectrograms for classification.
    
    Args:
        data_dir: Path to Data directory
        target_size: Resize spectrograms to this size
        max_samples_per_class: Limit samples per class (None = all)
    """
    print("=" * 70)
    print("Loading DroneRFb Single Drone Classification Dataset")
    print("=" * 70)
    
    X_list = []
    y_list = []
    
    for class_id in range(24):
        class_dir = data_dir / str(class_id)
        if not class_dir.exists():
            print(f"Warning: Class {class_id} directory not found")
            continue
        
        npy_files = list(class_dir.glob("*.npy"))
        if max_samples_per_class:
            npy_files = npy_files[:max_samples_per_class]
        
        print(f"Class {class_id:2d} ({CLASS_NAMES[class_id]:25s}): Loading {len(npy_files)} samples...")
        
        for npy_file in npy_files:
            try:
                # Load spectrogram
                spec = np.load(npy_file).astype(np.float32)
                
                # Resize to target size
                from scipy.ndimage import zoom
                scale_x = target_size[0] / spec.shape[0]
                scale_y = target_size[1] / spec.shape[1]
                spec_resized = zoom(spec, (scale_x, scale_y), order=1)
                
                # Normalize
                if spec_resized.max() > spec_resized.min():
                    spec_resized = (spec_resized - spec_resized.min()) / (spec_resized.max() - spec_resized.min())
                
                # Add channel dimension
                spec_resized = np.expand_dims(spec_resized, axis=-1)
                
                X_list.append(spec_resized)
                y_list.append(class_id)
                
            except Exception as e:
                print(f"  Error loading {npy_file}: {e}")
    
    X = np.array(X_list)
    y = np.array(y_list)
    
    print(f"\n{'=' * 70}")
    print(f"Dataset loaded: {X.shape[0]} samples")
    print(f"Input shape: {X.shape[1:]}")
    print(f"Number of classes: {len(np.unique(y))}")
    print(f"{'=' * 70}\n")
    
    return X, y


def build_cnn_classifier(input_shape, num_classes):
    """
    Build CNN classifier for single drone spectrograms.
    Similar to the architecture used in the paper.
    """
    model = Sequential([
        # Block 1
        Conv2D(32, (3, 3), activation='relu', padding='same', input_shape=input_shape),
        BatchNormalization(),
        Conv2D(32, (3, 3), activation='relu', padding='same'),
        MaxPooling2D((2, 2)),
        Dropout(0.25),
        
        # Block 2
        Conv2D(64, (3, 3), activation='relu', padding='same'),
        BatchNormalization(),
        Conv2D(64, (3, 3), activation='relu', padding='same'),
        MaxPooling2D((2, 2)),
        Dropout(0.25),
        
        # Block 3
        Conv2D(128, (3, 3), activation='relu', padding='same'),
        BatchNormalization(),
        Conv2D(128, (3, 3), activation='relu', padding='same'),
        MaxPooling2D((2, 2)),
        Dropout(0.25),
        
        # Block 4
        Conv2D(256, (3, 3), activation='relu', padding='same'),
        BatchNormalization(),
        Conv2D(256, (3, 3), activation='relu', padding='same'),
        MaxPooling2D((2, 2)),
        Dropout(0.25),
        
        # Dense layers
        Flatten(),
        Dense(512, activation='relu'),
        BatchNormalization(),
        Dropout(0.5),
        Dense(256, activation='relu'),
        Dropout(0.5),
        Dense(num_classes, activation='softmax')
    ])
    
    model.compile(
        optimizer='adam',
        loss='categorical_crossentropy',
        metrics=['accuracy']
    )
    
    return model


def plot_training_history(history, output_dir):
    """Plot training and validation accuracy/loss."""
    fig, (ax1, ax2) = plt.subplots(1, 2, figsize=(12, 4))
    
    # Accuracy
    ax1.plot(history.history['accuracy'], label='Train')
    ax1.plot(history.history['val_accuracy'], label='Validation')
    ax1.set_title('Model Accuracy')
    ax1.set_xlabel('Epoch')
    ax1.set_ylabel('Accuracy')
    ax1.legend()
    ax1.grid(True)
    
    # Loss
    ax2.plot(history.history['loss'], label='Train')
    ax2.plot(history.history['val_loss'], label='Validation')
    ax2.set_title('Model Loss')
    ax2.set_xlabel('Epoch')
    ax2.set_ylabel('Loss')
    ax2.legend()
    ax2.grid(True)
    
    plt.tight_layout()
    plt.savefig(output_dir / 'training_history.png', dpi=150, bbox_inches='tight')
    print(f"Saved training history to {output_dir / 'training_history.png'}")
    plt.close()


def plot_confusion_matrix(y_true, y_pred, output_dir):
    """Plot confusion matrix."""
    cm = confusion_matrix(y_true, y_pred)
    
    plt.figure(figsize=(16, 14))
    sns.heatmap(cm, annot=True, fmt='d', cmap='Blues',
                xticklabels=[f"{i}" for i in range(24)],
                yticklabels=[f"{i}" for i in range(24)])
    plt.xlabel('Predicted Class')
    plt.ylabel('True Class')
    plt.title('Single Drone Classification - Confusion Matrix')
    plt.tight_layout()
    plt.savefig(output_dir / 'confusion_matrix.png', dpi=150, bbox_inches='tight')
    print(f"Saved confusion matrix to {output_dir / 'confusion_matrix.png'}")
    plt.close()


def main():
    """Main training pipeline."""
    # Paths
    data_dir = Path("/home/gabriel/Desktop/HackathonNATO_Drones_2025/DroneRFb-Spectra/Data")
    output_dir = Path("/home/gabriel/Desktop/HackathonNATO_Drones_2025/DroneRFb_SingleDrone_Results")
    output_dir.mkdir(parents=True, exist_ok=True)
    
    # Load data (use all samples with CUDA!)
    print("Loading dataset...")
    X, y = load_dataset(data_dir, target_size=(128, 128), max_samples_per_class=None)
    
    # Convert labels to categorical
    y_categorical = to_categorical(y, num_classes=24)
    
    # Split data
    X_train, X_test, y_train, y_test = train_test_split(
        X, y_categorical, test_size=0.2, random_state=42, stratify=y
    )
    X_train, X_val, y_train, y_val = train_test_split(
        X_train, y_train, test_size=0.1, random_state=42, 
        stratify=np.argmax(y_train, axis=1)
    )
    
    print(f"Training samples: {len(X_train)}")
    print(f"Validation samples: {len(X_val)}")
    print(f"Test samples: {len(X_test)}")
    print()
    
    # Build model
    print("Building CNN model...")
    model = build_cnn_classifier(X_train.shape[1:], 24)
    model.summary()
    
    # Callbacks
    callbacks = [
        EarlyStopping(
            monitor='val_loss',
            patience=20,
            restore_best_weights=True,
            verbose=1
        ),
        ModelCheckpoint(
            str(output_dir / 'best_model.h5'),
            monitor='val_accuracy',
            save_best_only=True,
            verbose=1
        ),
        ReduceLROnPlateau(
            monitor='val_loss',
            factor=0.5,
            patience=10,
            min_lr=1e-7,
            verbose=1
        )
    ]
    
    # Train
    print("\n" + "=" * 70)
    print("Training model...")
    print("=" * 70)
    
    history = model.fit(
        X_train, y_train,
        validation_data=(X_val, y_val),
        epochs=20,  # Full training with CUDA
        batch_size=32,  # Larger batch for GPU
        callbacks=callbacks,
        verbose=1
    )
    
    # Evaluate
    print("\n" + "=" * 70)
    print("Evaluating on test set...")
    print("=" * 70)
    
    test_loss, test_acc = model.evaluate(X_test, y_test, verbose=0)
    print(f"Test Accuracy: {test_acc*100:.2f}%")
    print(f"Test Loss: {test_loss:.4f}")
    
    # Predictions
    y_pred = model.predict(X_test, verbose=0)
    y_pred_classes = np.argmax(y_pred, axis=1)
    y_test_classes = np.argmax(y_test, axis=1)
    
    # Classification report
    print("\nClassification Report:")
    print(classification_report(y_test_classes, y_pred_classes, 
                                target_names=[f"Class {i}" for i in range(24)]))
    
    # Per-class accuracy
    print("\nPer-Class Accuracy:")
    for i in range(24):
        mask = y_test_classes == i
        if np.sum(mask) > 0:
            class_acc = accuracy_score(y_test_classes[mask], y_pred_classes[mask])
            print(f"  Class {i:2d} ({CLASS_NAMES[i]:25s}): {class_acc*100:.2f}%")
    
    # Save results
    plot_training_history(history, output_dir)
    plot_confusion_matrix(y_test_classes, y_pred_classes, output_dir)
    
    # Save final model
    model.save(output_dir / 'final_model.h5')
    print(f"\nSaved final model to {output_dir / 'final_model.h5'}")
    
    print("\n" + "=" * 70)
    print("Training complete!")
    print("=" * 70)


if __name__ == "__main__":
    main()
