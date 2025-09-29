#!/usr/bin/env python3
"""
Test script for the balanced model with 20 epochs training.
"""

import os
import cv2
import numpy as np
from pathlib import Path
import matplotlib.pyplot as plt
import matplotlib.patches as patches
from matplotlib.patches import Rectangle
import argparse

def load_model(model_path):
    """Load the trained YOLO model."""
    try:
        from ultralytics import YOLO
        model = YOLO(str(model_path))
        print(f"Loaded model from: {model_path}")
        return model
    except ImportError:
        print("Error: ultralytics package not found.")
        print("Please install it with: pip install ultralytics")
        return None
    except Exception as e:
        print(f"Error loading model: {e}")
        return None

def test_balanced_model(model, image_dir, num_images=5, confidence_threshold=0.3):
    """Test the balanced model on spectrograms."""
    
    class_names = ['WLAN', 'collision', 'bluetooth']
    image_files = list(image_dir.glob("*.png"))[:num_images]
    
    if not image_files:
        print(f"No PNG images found in {image_dir}")
        return
    
    print(f"Testing balanced model on {len(image_files)} images...")
    
    for img_path in image_files:
        print(f"\n=== Testing: {img_path.name} ===")
        
        # Run inference
        results = model(str(img_path))
        
        # Load image
        image = cv2.imread(str(img_path))
        image = cv2.cvtColor(image, cv2.COLOR_BGR2RGB)
        
        # Create visualization
        fig, ax = plt.subplots(1, 1, figsize=(12, 8))
        ax.imshow(image)
        ax.set_title(f"Balanced Model Predictions: {img_path.name}", fontsize=14)
        
        # Count detections by class
        detection_counts = {0: 0, 1: 0, 2: 0}
        
        if results[0].boxes is not None:
            boxes = results[0].boxes
            for box in boxes:
                confidence = float(box.conf[0])
                if confidence >= confidence_threshold:
                    x1, y1, x2, y2 = box.xyxy[0].cpu().numpy()
                    class_id = int(box.cls[0])
                    detection_counts[class_id] += 1
                    
                    width = x2 - x1
                    height = y2 - y1
                    
                    # Color coding
                    colors = ['red', 'blue', 'green']
                    color = colors[class_id]
                    
                    rect = Rectangle((x1, y1), width, height, linewidth=2,
                                   edgecolor=color, facecolor='none')
                    ax.add_patch(rect)
                    
                    label = f"{class_names[class_id]}: {confidence:.2f}"
                    ax.text(x1, y1-5, label, fontsize=10, color=color,
                           bbox=dict(boxstyle="round,pad=0.3", facecolor='white', alpha=0.8))
        
        ax.axis('off')
        
        # Add detection summary
        summary_text = f"Detections: WLAN={detection_counts[0]}, Collision={detection_counts[1]}, Bluetooth={detection_counts[2]}"
        ax.text(0.02, 0.98, summary_text, transform=ax.transAxes, fontsize=12,
               bbox=dict(boxstyle="round,pad=0.5", facecolor='yellow', alpha=0.8),
               verticalalignment='top')
        
        plt.tight_layout()
        
        # Save result
        output_path = Path("balanced_test_results") / f"balanced_prediction_{img_path.stem}.png"
        output_path.parent.mkdir(exist_ok=True)
        fig.savefig(output_path, dpi=150, bbox_inches='tight')
        print(f"Saved result to: {output_path}")
        
        plt.show()
        
        # Print detection summary
        print(f"  WLAN detections: {detection_counts[0]}")
        print(f"  Collision detections: {detection_counts[1]}")
        print(f"  Bluetooth detections: {detection_counts[2]}")

def main():
    parser = argparse.ArgumentParser(description="Test balanced YOLO model")
    parser.add_argument("--model", type=str, 
                       default="yolo_training_improved/rf_spectrogram_detection_improved/weights/best.pt",
                       help="Path to trained model")
    parser.add_argument("--image_dir", type=str, 
                       default="datasets_balanced/images/val",
                       help="Directory containing images to test")
    parser.add_argument("--num_images", type=int, default=5,
                       help="Number of images to test")
    parser.add_argument("--confidence", type=float, default=0.3,
                       help="Confidence threshold for detections")
    
    args = parser.parse_args()
    
    # Load model
    model = load_model(args.model)
    if model is None:
        return
    
    # Test balanced model
    test_balanced_model(model, Path(args.image_dir), args.num_images, args.confidence)

if __name__ == "__main__":
    main()
