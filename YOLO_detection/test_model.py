"""
Test YOLO Model for RF Signal Detection
Enhanced script to test trained YOLO model with labeling analysis
"""

import torch
from ultralytics import YOLO
import os
import glob
import matplotlib.pyplot as plt
import numpy as np
from PIL import Image
import random
from datetime import datetime

def test_single_image(model_path, image_path, conf_threshold=0.5):
    """Test model on a single image"""
    print(f"Testing: {os.path.basename(image_path)}")
    
    # Load model
    model = YOLO(model_path)
    
    # Make prediction
    results = model(image_path, conf=conf_threshold)
    
    # Get results
    result = results[0]
    
    if result.boxes is not None and len(result.boxes) > 0:
        print(f"Detected {len(result.boxes)} RF signals:")
        
        for i, box in enumerate(result.boxes):
            conf = box.conf.item()
            cls = int(box.cls.item())
            class_names = ['Background', 'WLAN', 'Bluetooth']
            class_name = class_names[cls]
            
            print(f"  {i+1}. {class_name} (confidence: {conf:.3f})")
    else:
        print("No RF signals detected")
    
    # Create visualization
    create_visualization(image_path, result)
    
    return result

def load_yolo_labels(txt_path):
    """Load YOLO format labels from txt file"""
    labels = []
    if os.path.exists(txt_path):
        with open(txt_path, 'r') as f:
            for line in f:
                parts = line.strip().split()
                if len(parts) >= 5:
                    class_id = int(parts[0])
                    x_center = float(parts[1])
                    y_center = float(parts[2])
                    width = float(parts[3])
                    height = float(parts[4])
                    labels.append((class_id, x_center, y_center, width, height))
    return labels

def normalize_to_pixel_coords(x_center, y_center, width, height, image_shape):
    """Convert normalized YOLO coordinates to pixel coordinates"""
    img_height, img_width = image_shape[:2]
    
    # Convert to pixel coordinates
    x_center_px = x_center * img_width
    y_center_px = y_center * img_height
    width_px = width * img_width
    height_px = height * img_height
    
    # Convert to corner coordinates
    x1 = x_center_px - width_px / 2
    y1 = y_center_px - height_px / 2
    x2 = x_center_px + width_px / 2
    y2 = y_center_px + height_px / 2
    
    return int(x1), int(y1), int(x2), int(y2)

def create_enhanced_visualization(image_path, result, test_number):
    """Create enhanced visualization with labeling analysis"""
    # Load original image
    image = Image.open(image_path).convert('RGB')
    image_array = np.array(image)
    
    # Load YOLO labels
    txt_path = image_path.replace('.png', '.txt')
    yolo_labels = load_yolo_labels(txt_path)
    
    # Create figure with 2x2 layout
    fig, axes = plt.subplots(2, 2, figsize=(20, 16))
    
    class_names = ['Background', 'WLAN', 'Bluetooth', 'BLE']
    colors = ['red', 'blue', 'green', 'orange']
    
    # Panel 1: Original image
    axes[0, 0].imshow(image)
    axes[0, 0].set_title(f'Test #{test_number}: Original Spectrogram', fontsize=14, fontweight='bold')
    axes[0, 0].axis('off')
    
    # Panel 2: Model predictions
    axes[0, 1].imshow(image)
    axes[0, 1].set_title('Model Predictions', fontsize=14, fontweight='bold')
    axes[0, 1].axis('off')
    
    # Draw model bounding boxes
    model_detections = 0
    if result.boxes is not None and len(result.boxes) > 0:
        model_detections = len(result.boxes)
        for box in result.boxes:
            # Get box coordinates
            x1, y1, x2, y2 = box.xyxy[0].cpu().numpy()
            conf = box.conf.item()
            cls = int(box.cls.item())
            
            # Draw bounding box
            color = colors[cls] if cls < len(colors) else 'purple'
            rect = plt.Rectangle((x1, y1), x2-x1, y2-y1, 
                               fill=False, color=color, linewidth=2)
            axes[0, 1].add_patch(rect)
            
            # Add label
            class_name = class_names[cls] if cls < len(class_names) else f'Class_{cls}'
            axes[0, 1].text(x1, y1-5, f'{class_name} {conf:.2f}', 
                       color=color, fontsize=9, fontweight='bold',
                       bbox=dict(boxstyle='round,pad=0.2', facecolor='white', alpha=0.8))
    
    # Panel 3: YOLO Labels (from txt file)
    axes[1, 0].imshow(image)
    axes[1, 0].set_title('YOLO Labels (Ground Truth)', fontsize=14, fontweight='bold')
    axes[1, 0].axis('off')
    
    # Draw YOLO label bounding boxes
    yolo_detections = 0
    for class_id, x_center, y_center, width, height in yolo_labels:
        yolo_detections += 1
        x1, y1, x2, y2 = normalize_to_pixel_coords(x_center, y_center, width, height, image_array.shape)
        
        # Draw bounding box
        color = colors[class_id] if class_id < len(colors) else 'purple'
        rect = plt.Rectangle((x1, y1), x2-x1, y2-y1, 
                           fill=False, color=color, linewidth=2)
        axes[1, 0].add_patch(rect)
        
        # Add label
        class_name = class_names[class_id] if class_id < len(class_names) else f'Class_{class_id}'
        axes[1, 0].text(x1, y1-5, class_name, 
                   color=color, fontsize=9, fontweight='bold',
                   bbox=dict(boxstyle='round,pad=0.2', facecolor='white', alpha=0.8))
    
    # Panel 4: Human-marked image (if available)
    marked_path = image_path.replace('.png', '_marked.png')
    if os.path.exists(marked_path):
        marked_image = Image.open(marked_path).convert('RGB')
        axes[1, 1].imshow(marked_image)
        axes[1, 1].set_title('Human Marked (Reference)', fontsize=14, fontweight='bold')
    else:
        axes[1, 1].text(0.5, 0.5, 'No marked image available', 
                    ha='center', va='center', transform=axes[1, 1].transAxes,
                    fontsize=12, fontweight='bold')
        axes[1, 1].set_title('Human Marked (Not Available)', fontsize=14, fontweight='bold')
    
    axes[1, 1].axis('off')
    
    # Add summary text
    summary_text = f"Model Detections: {model_detections}\nYOLO Labels: {yolo_detections}"
    fig.text(0.5, 0.02, summary_text, ha='center', fontsize=12, fontweight='bold',
             bbox=dict(boxstyle='round,pad=0.5', facecolor='lightgray', alpha=0.8))
    
    plt.tight_layout()
    plt.subplots_adjust(bottom=0.1)
    
    # Save visualization
    output_dir = "yolo_testing_results"
    os.makedirs(output_dir, exist_ok=True)
    
    base_name = os.path.splitext(os.path.basename(image_path))[0]
    output_path = os.path.join(output_dir, f"test_{test_number:02d}_{base_name}.png")
    plt.savefig(output_path, dpi=300, bbox_inches='tight')
    print(f"Enhanced visualization saved as: {output_path}")
    
    plt.show()
    
    return model_detections, yolo_detections

def test_multiple_images(model_path, num_images=10, conf_threshold=0.5):
    """Test model on multiple images with enhanced analysis"""
    print(f"=== Enhanced YOLO Model Testing ===")
    print(f"Model: {model_path}")
    print(f"Confidence threshold: {conf_threshold}")
    print(f"Test timestamp: {datetime.now().strftime('%Y-%m-%d %H:%M:%S')}")
    
    # Get test images
    image_dir = '../spectrogram_training_data_20220711/results'
    image_files = glob.glob(os.path.join(image_dir, '*.png'))
    image_files = [f for f in image_files if 'marked' not in f]
    
    if len(image_files) == 0:
        print("No test images found!")
        return
    
    # Select random subset
    selected_files = random.sample(image_files, min(num_images, len(image_files)))
    
    print(f"Testing {len(selected_files)} images...")
    
    # Initialize tracking variables
    total_model_detections = 0
    total_yolo_labels = 0
    test_results = []
    
    for i, image_path in enumerate(selected_files):
        print(f"\n--- Test #{i+1}/{len(selected_files)} ---")
        print(f"Image: {os.path.basename(image_path)}")
        
        # Load model and make prediction
        model = YOLO(model_path)
        results = model(image_path, conf=conf_threshold)
        result = results[0]
        
        # Create enhanced visualization
        model_detections, yolo_detections = create_enhanced_visualization(image_path, result, i+1)
        
        # Track results
        total_model_detections += model_detections
        total_yolo_labels += yolo_detections
        test_results.append({
            'image': os.path.basename(image_path),
            'model_detections': model_detections,
            'yolo_labels': yolo_detections,
            'detection_ratio': model_detections / max(yolo_detections, 1)
        })
        
        print(f"Model detections: {model_detections}")
        print(f"YOLO labels: {yolo_detections}")
        print(f"Detection ratio: {model_detections / max(yolo_detections, 1):.2f}")
    
    # Create comprehensive test summary
    create_test_summary(test_results, model_path, conf_threshold, total_model_detections, total_yolo_labels)
    
    print(f"\n=== Enhanced Test Summary ===")
    print(f"Total images tested: {len(selected_files)}")
    print(f"Total model detections: {total_model_detections}")
    print(f"Total YOLO labels: {total_yolo_labels}")
    print(f"Average model detections per image: {total_model_detections/len(selected_files):.2f}")
    print(f"Average YOLO labels per image: {total_yolo_labels/len(selected_files):.2f}")
    print(f"Overall detection ratio: {total_model_detections / max(total_yolo_labels, 1):.2f}")

def create_test_summary(test_results, model_path, conf_threshold, total_model_detections, total_yolo_labels):
    """Create comprehensive test summary with matrix and analysis"""
    output_dir = "yolo_testing_results"
    os.makedirs(output_dir, exist_ok=True)
    
    # Create test summary file
    summary_path = os.path.join(output_dir, "test_summary.txt")
    
    with open(summary_path, 'w') as f:
        f.write("=== YOLO Model Test Summary ===\n")
        f.write(f"Test Date: {datetime.now().strftime('%Y-%m-%d %H:%M:%S')}\n")
        f.write(f"Model: {model_path}\n")
        f.write(f"Confidence Threshold: {conf_threshold}\n")
        f.write(f"Total Images Tested: {len(test_results)}\n\n")
        
        f.write("=== Individual Test Results ===\n")
        f.write("Test# | Image | Model Detections | YOLO Labels | Detection Ratio\n")
        f.write("-" * 70 + "\n")
        
        for i, result in enumerate(test_results):
            f.write(f"{i+1:5d} | {result['image'][:20]:20s} | {result['model_detections']:15d} | {result['yolo_labels']:11d} | {result['detection_ratio']:15.2f}\n")
        
        f.write("\n=== Summary Statistics ===\n")
        f.write(f"Total Model Detections: {total_model_detections}\n")
        f.write(f"Total YOLO Labels: {total_yolo_labels}\n")
        f.write(f"Average Model Detections per Image: {total_model_detections/len(test_results):.2f}\n")
        f.write(f"Average YOLO Labels per Image: {total_yolo_labels/len(test_results):.2f}\n")
        f.write(f"Overall Detection Ratio: {total_model_detections / max(total_yolo_labels, 1):.2f}\n")
        
        # Analysis notes
        f.write("\n=== Analysis Notes ===\n")
        if total_model_detections == 0:
            f.write("WARNING: Model made no detections on any test images.\n")
            f.write("This suggests the model may be:\n")
            f.write("- Over-trained to predict 'no objects'\n")
            f.write("- Using incorrect loss function\n")
            f.write("- Need to adjust confidence threshold\n")
        elif total_model_detections < total_yolo_labels * 0.5:
            f.write("NOTE: Model is detecting significantly fewer objects than ground truth.\n")
            f.write("This may indicate:\n")
            f.write("- Model is being too conservative\n")
            f.write("- Need to lower confidence threshold\n")
            f.write("- Training data imbalance issues\n")
        else:
            f.write("Model appears to be making reasonable detections.\n")
    
    print(f"Test summary saved to: {summary_path}")

def main():
    """Main function"""
    import argparse
    
    parser = argparse.ArgumentParser(description='Test YOLO Model')
    parser.add_argument('--model_path', type=str, default='yolo_training/rf_detection/weights/best.pt',
                       help='Path to trained model')
    parser.add_argument('--image_path', type=str, default=None,
                       help='Path to single image to test')
    parser.add_argument('--num_images', type=int, default=10,
                       help='Number of images to test')
    parser.add_argument('--conf_threshold', type=float, default=0.5,
                       help='Confidence threshold for detections')
    
    args = parser.parse_args()
    
    # Check if model exists
    if not os.path.exists(args.model_path):
        print(f"Model not found: {args.model_path}")
        print("Please train the model first with: python3 train_model.py")
        return
    
    if args.image_path:
        # Test single image with enhanced visualization
        if os.path.exists(args.image_path):
            model = YOLO(args.model_path)
            results = model(args.image_path, conf=args.conf_threshold)
            result = results[0]
            create_enhanced_visualization(args.image_path, result, 1)
        else:
            print(f"Image not found: {args.image_path}")
    else:
        # Test multiple images with enhanced analysis
        test_multiple_images(args.model_path, args.num_images, args.conf_threshold)

if __name__ == "__main__":
    main()
