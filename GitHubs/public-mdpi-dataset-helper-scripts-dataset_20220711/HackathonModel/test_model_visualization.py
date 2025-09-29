#!/usr/bin/env python3
"""
Test script for visualizing YOLO model predictions on spectrograms.
This script loads a trained model and shows how it labels spectrograms with bounding boxes.
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

def draw_predictions(image, results, class_names, confidence_threshold=0.5):
    """Draw bounding boxes and labels on the image."""
    fig, ax = plt.subplots(1, 1, figsize=(12, 8))
    ax.imshow(image)
    
    # Colors for different classes
    colors = ['red', 'blue', 'green', 'orange', 'purple']
    
    if results[0].boxes is not None:
        boxes = results[0].boxes
        for i, box in enumerate(boxes):
            confidence = float(box.conf[0])
            
            if confidence >= confidence_threshold:
                # Get bounding box coordinates
                x1, y1, x2, y2 = box.xyxy[0].cpu().numpy()
                class_id = int(box.cls[0])
                
                # Draw bounding box
                width = x2 - x1
                height = y2 - y1
                color = colors[class_id % len(colors)]
                
                rect = Rectangle((x1, y1), width, height, 
                               linewidth=2, edgecolor=color, facecolor='none')
                ax.add_patch(rect)
                
                # Add label
                label = f"{class_names[class_id]}: {confidence:.2f}"
                ax.text(x1, y1-5, label, fontsize=10, color=color, 
                       bbox=dict(boxstyle="round,pad=0.3", facecolor='white', alpha=0.8))
    
    ax.set_title("YOLO Predictions on RF Spectrogram", fontsize=14)
    ax.axis('off')
    return fig

def test_single_image(model, image_path, class_names, confidence_threshold=0.5, save_result=True):
    """Test the model on a single image."""
    print(f"Testing image: {image_path.name}")
    
    # Run inference
    results = model(str(image_path))
    
    # Load image
    image = cv2.imread(str(image_path))
    image = cv2.cvtColor(image, cv2.COLOR_BGR2RGB)
    
    # Draw predictions
    fig = draw_predictions(image, results, class_names, confidence_threshold)
    
    # Print detection summary
    if results[0].boxes is not None:
        boxes = results[0].boxes
        detections = []
        for box in boxes:
            if float(box.conf[0]) >= confidence_threshold:
                class_id = int(box.cls[0])
                confidence = float(box.conf[0])
                detections.append((class_names[class_id], confidence))
        
        print(f"  Detected {len(detections)} objects:")
        for class_name, conf in detections:
            print(f"    - {class_name}: {conf:.3f}")
    else:
        print("  No objects detected")
    
    # Save result if requested
    if save_result:
        output_path = Path("test_results") / f"prediction_{image_path.stem}.png"
        output_path.parent.mkdir(exist_ok=True)
        fig.savefig(output_path, dpi=150, bbox_inches='tight')
        print(f"  Saved result to: {output_path}")
    
    plt.show()
    return results

def test_multiple_images(model, image_dir, class_names, num_images=5, confidence_threshold=0.5):
    """Test the model on multiple images."""
    image_files = list(image_dir.glob("*.png"))[:num_images]
    
    if not image_files:
        print(f"No PNG images found in {image_dir}")
        return
    
    print(f"Testing {len(image_files)} images...")
    
    all_results = []
    for img_path in image_files:
        results = test_single_image(model, img_path, class_names, confidence_threshold)
        all_results.append((img_path.name, results))
        print("-" * 50)
    
    return all_results

def compare_with_ground_truth(model, image_path, label_path, class_names, confidence_threshold=0.5):
    """Compare model predictions with ground truth labels."""
    print(f"Comparing predictions with ground truth for: {image_path.name}")
    
    # Get model predictions
    results = model(str(image_path))
    
    # Load ground truth labels
    gt_boxes = []
    if label_path.exists():
        with open(label_path, 'r') as f:
            for line in f:
                if line.strip():
                    parts = line.strip().split()
                    class_id = int(parts[0])
                    x_center, y_center, width, height = map(float, parts[1:5])
                    gt_boxes.append((class_id, x_center, y_center, width, height))
    
    # Load image
    image = cv2.imread(str(image_path))
    image = cv2.cvtColor(image, cv2.COLOR_BGR2RGB)
    img_height, img_width = image.shape[:2]
    
    # Create comparison plot
    fig, (ax1, ax2) = plt.subplots(1, 2, figsize=(20, 8))
    
    # Plot 1: Ground Truth
    ax1.imshow(image)
    ax1.set_title("Ground Truth Labels", fontsize=14)
    
    colors = ['red', 'blue', 'green']
    for class_id, x_center, y_center, width, height in gt_boxes:
        # Convert normalized coordinates to pixel coordinates
        x1 = (x_center - width/2) * img_width
        y1 = (y_center - height/2) * img_height
        w = width * img_width
        h = height * img_height
        
        rect = Rectangle((x1, y1), w, h, linewidth=2, 
                        edgecolor=colors[class_id], facecolor='none')
        ax1.add_patch(rect)
        ax1.text(x1, y1-5, class_names[class_id], fontsize=10, color=colors[class_id])
    
    ax1.axis('off')
    
    # Plot 2: Model Predictions
    ax2.imshow(image)
    ax2.set_title("Model Predictions", fontsize=14)
    
    if results[0].boxes is not None:
        boxes = results[0].boxes
        for box in boxes:
            confidence = float(box.conf[0])
            if confidence >= confidence_threshold:
                x1, y1, x2, y2 = box.xyxy[0].cpu().numpy()
                class_id = int(box.cls[0])
                
                width = x2 - x1
                height = y2 - y1
                color = colors[class_id % len(colors)]
                
                rect = Rectangle((x1, y1), width, height, linewidth=2,
                               edgecolor=color, facecolor='none')
                ax2.add_patch(rect)
                
                label = f"{class_names[class_id]}: {confidence:.2f}"
                ax2.text(x1, y1-5, label, fontsize=10, color=color,
                        bbox=dict(boxstyle="round,pad=0.3", facecolor='white', alpha=0.8))
    
    ax2.axis('off')
    
    plt.tight_layout()
    
    # Save comparison
    output_path = Path("test_results") / f"comparison_{image_path.stem}.png"
    output_path.parent.mkdir(exist_ok=True)
    fig.savefig(output_path, dpi=150, bbox_inches='tight')
    print(f"Saved comparison to: {output_path}")
    
    plt.show()

def main():
    parser = argparse.ArgumentParser(description="Test YOLO model on spectrograms")
    parser.add_argument("--model", type=str, 
                       default="yolo_training/rf_spectrogram_detection/weights/best.pt",
                       help="Path to trained model")
    parser.add_argument("--image", type=str, help="Path to single image to test")
    parser.add_argument("--image_dir", type=str, 
                       default="datasets/images/val",
                       help="Directory containing images to test")
    parser.add_argument("--num_images", type=int, default=5,
                       help="Number of images to test from directory")
    parser.add_argument("--confidence", type=float, default=0.5,
                       help="Confidence threshold for detections")
    parser.add_argument("--compare", action="store_true",
                       help="Compare predictions with ground truth")
    
    args = parser.parse_args()
    
    # Class names
    class_names = ['WLAN', 'collision', 'bluetooth']
    
    # Load model
    model_path = Path(args.model)
    if not model_path.exists():
        print(f"Model not found at: {model_path}")
        print("Please train the model first with: python train_yolo_model.py --train")
        return
    
    model = load_model(model_path)
    if model is None:
        return
    
    # Test single image
    if args.image:
        image_path = Path(args.image)
        if not image_path.exists():
            print(f"Image not found: {image_path}")
            return
        
        test_single_image(model, image_path, class_names, args.confidence)
    
    # Test multiple images
    elif args.image_dir:
        image_dir = Path(args.image_dir)
        if not image_dir.exists():
            print(f"Image directory not found: {image_dir}")
            return
        
        if args.compare:
            # Compare with ground truth
            label_dir = Path("datasets/labels/val")
            image_files = list(image_dir.glob("*.png"))[:args.num_images]
            
            for img_path in image_files:
                label_path = label_dir / f"{img_path.stem}.txt"
                compare_with_ground_truth(model, img_path, label_path, 
                                        class_names, args.confidence)
                print("-" * 50)
        else:
            # Just show predictions
            test_multiple_images(model, image_dir, class_names, 
                               args.num_images, args.confidence)

if __name__ == "__main__":
    main()
