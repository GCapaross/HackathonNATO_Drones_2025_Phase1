"""
Test YOLO Model on Unmarked Spectrograms
This script loads a trained YOLO model and tests it on new spectrograms,
showing how the model detects and labels RF signals.
"""

import torch
import torch.nn as nn
import numpy as np
import matplotlib.pyplot as plt
import cv2
from PIL import Image
import os
import argparse
import glob
from yolo_model import create_yolo_model
from yolo_data_loader import SpectrogramYOLODataset

class YOLOTester:
    """
    Test YOLO model on new spectrograms
    """
    
    def __init__(self, model_path, class_names, device='cuda'):
        self.device = torch.device(device if torch.cuda.is_available() else 'cpu')
        
        # Load model first to determine number of classes
        self.model, detected_classes = self.load_model(model_path)
        self.model.eval()
        
        # Use detected class names
        self.class_names = detected_classes
        
        self.class_colors = {
            0: 'red',      # Background
            1: 'blue',     # WLAN
            2: 'green',    # Bluetooth
        }
        # Add more colors if needed for additional classes
        if len(self.class_names) > 3:
            self.class_colors[3] = 'orange'  # BLE or other classes
        
        print(f"Model loaded from: {model_path}")
        print(f"Classes: {self.class_names}")
        print(f"Device: {self.device}")
    
    def load_model(self, model_path):
        """Load the trained model"""
        # Load checkpoint first to check the number of classes
        checkpoint = torch.load(model_path, map_location=self.device)
        
        # Determine number of classes from the saved model
        if 'model_state_dict' in checkpoint:
            state_dict = checkpoint['model_state_dict']
            # Check the detection head weight shape to determine number of classes
            detection_head_weight = state_dict['detection_head.5.weight']
            # Shape is [3 * (5 + num_classes), 1024]
            total_outputs = detection_head_weight.shape[0]
            num_classes = (total_outputs // 3) - 5
            print(f"Detected {num_classes} classes from saved model")
        else:
            # Fallback to 3 classes
            num_classes = 3
            print("Using default 3 classes")
        
        # Create model with the correct number of classes
        model = create_yolo_model(num_classes=num_classes, img_size=640)
        
        # Load the state dict
        if 'model_state_dict' in checkpoint:
            model.load_state_dict(checkpoint['model_state_dict'])
            print(f"Loaded checkpoint from epoch {checkpoint.get('epoch', 'unknown')}")
        else:
            model.load_state_dict(checkpoint)
            print("Loaded model state dict")
        
        model.to(self.device)
        
        # Create class names based on detected number of classes
        if num_classes == 3:
            class_names = ['Background', 'WLAN', 'Bluetooth']
        elif num_classes == 4:
            class_names = ['Background', 'WLAN', 'Bluetooth', 'BLE']
        else:
            class_names = [f'Class_{i}' for i in range(num_classes)]
        
        return model, class_names
    
    def preprocess_image(self, image_path):
        """Preprocess image for YOLO input - NO RESIZING"""
        # Load image
        image = Image.open(image_path).convert('RGB')
        original_size = image.size  # (width, height)
        
        # Keep original size - just convert to tensor
        image_np = np.array(image)
        
        # Convert to tensor (keep original dimensions)
        image_tensor = torch.from_numpy(image_np).permute(2, 0, 1).float() / 255.0
        image_tensor = image_tensor.unsqueeze(0)  # Add batch dimension
        
        return image_tensor.to(self.device), original_size, image_np
    
    # Removed resize function - we keep original image size
    
    def predict(self, image_tensor, conf_threshold=0.5, nms_threshold=0.4):
        """Make predictions on the image"""
        with torch.no_grad():
            outputs = self.model(image_tensor)
            
            # Extract predictions
            batch_size = outputs.shape[0]
            predictions = []
            
            for b in range(batch_size):
                batch_outputs = outputs[b]  # Shape: (3, 5 + num_classes)
                
                # Apply confidence threshold
                conf_scores = torch.sigmoid(batch_outputs[:, 4])  # Confidence scores
                conf_mask = conf_scores > conf_threshold
                
                if conf_mask.sum() > 0:
                    filtered_outputs = batch_outputs[conf_mask]
                    filtered_conf = conf_scores[conf_mask]
                    
                    # Get class predictions
                    class_scores = torch.softmax(filtered_outputs[:, 5:], dim=1)
                    class_conf, class_pred = torch.max(class_scores, dim=1)
                    
                    # Final confidence
                    final_conf = filtered_conf * class_conf
                    
                    # Extract bounding boxes (normalized coordinates)
                    boxes = filtered_outputs[:, :4]  # x_center, y_center, width, height
                    
                    # Apply NMS
                    keep_indices = self.apply_nms(boxes, final_conf, nms_threshold)
                    
                    if len(keep_indices) > 0:
                        final_boxes = boxes[keep_indices]
                        final_conf = final_conf[keep_indices]
                        final_classes = class_pred[keep_indices]
                        
                        predictions.append({
                            'boxes': final_boxes,
                            'confidences': final_conf,
                            'classes': final_classes
                        })
                    else:
                        predictions.append({
                            'boxes': torch.empty(0, 4),
                            'confidences': torch.empty(0),
                            'classes': torch.empty(0, dtype=torch.long)
                        })
                else:
                    predictions.append({
                        'boxes': torch.empty(0, 4),
                        'confidences': torch.empty(0),
                        'classes': torch.empty(0, dtype=torch.long)
                    })
            
            return predictions
    
    def apply_nms(self, boxes, scores, nms_threshold):
        """Apply Non-Maximum Suppression"""
        if len(boxes) == 0:
            return torch.empty(0, dtype=torch.long)
        
        # Convert to corner format for NMS
        x1 = boxes[:, 0] - boxes[:, 2] / 2
        y1 = boxes[:, 1] - boxes[:, 3] / 2
        x2 = boxes[:, 0] + boxes[:, 2] / 2
        y2 = boxes[:, 1] + boxes[:, 3] / 2
        
        # Calculate areas
        areas = (x2 - x1) * (y2 - y1)
        
        # Sort by scores
        _, indices = torch.sort(scores, descending=True)
        
        keep = []
        while len(indices) > 0:
            # Pick the box with highest score
            current = indices[0]
            keep.append(current)
            
            if len(indices) == 1:
                break
            
            # Calculate IoU with remaining boxes
            others = indices[1:]
            
            # Calculate intersection
            xx1 = torch.max(x1[current], x1[others])
            yy1 = torch.max(y1[current], y1[others])
            xx2 = torch.min(x2[current], x2[others])
            yy2 = torch.min(y2[current], y2[others])
            
            w = torch.clamp(xx2 - xx1, min=0)
            h = torch.clamp(yy2 - yy1, min=0)
            intersection = w * h
            
            # Calculate union
            union = areas[current] + areas[others] - intersection
            
            # Calculate IoU
            iou = intersection / union
            
            # Keep boxes with IoU below threshold
            indices = others[iou <= nms_threshold]
        
        return torch.tensor(keep, dtype=torch.long)
    
    def test_single_image(self, image_path, conf_threshold=0.5, nms_threshold=0.4):
        """Test model on a single image"""
        print(f"\nTesting: {os.path.basename(image_path)}")
        
        # Preprocess image (no resizing)
        image_tensor, original_size, original_image = self.preprocess_image(image_path)
        
        # Debug: Check raw model output
        with torch.no_grad():
            raw_output = self.model(image_tensor)
            print(f"Raw model output shape: {raw_output.shape}")
            print(f"Raw output sample values: {raw_output[0, 0, :5]}")  # First anchor, first 5 values
            print(f"Confidence values (sigmoid): {torch.sigmoid(raw_output[0, :, 4])}")  # Objectness scores
        
        # Make predictions
        predictions = self.predict(image_tensor, conf_threshold, nms_threshold)
        
        # Process results
        if len(predictions) > 0 and len(predictions[0]['boxes']) > 0:
            boxes = predictions[0]['boxes'].cpu().numpy()
            confidences = predictions[0]['confidences'].cpu().numpy()
            classes = predictions[0]['classes'].cpu().numpy()
            
            print(f"Detected {len(boxes)} RF signals:")
            for i, (box, conf, cls) in enumerate(zip(boxes, confidences, classes)):
                class_name = self.class_names[cls]
                print(f"  {i+1}. {class_name} (confidence: {conf:.3f})")
                print(f"     Position: ({box[0]:.3f}, {box[1]:.3f})")
                print(f"     Size: {box[2]:.3f} x {box[3]:.3f}")
        else:
            print("No RF signals detected")
        
        # Create 3-panel comparison
        self.create_comparison_visualization(original_image, predictions[0], image_path)
        
        return predictions[0]
    
    def create_comparison_visualization(self, original_image, predictions, image_path):
        """Create 3-panel comparison: Original -> Model Predictions -> Human Marked"""
        # Get base filename
        base_name = os.path.splitext(os.path.basename(image_path))[0]
        
        # Try to find the human-marked image
        marked_path = image_path.replace('.png', '_marked.png')
        marked_image = None
        if os.path.exists(marked_path):
            marked_image = np.array(Image.open(marked_path).convert('RGB'))
        
        # Create 3-panel figure
        fig, axes = plt.subplots(1, 3, figsize=(24, 8))
        
        # Panel 1: Original spectrogram (clean, no labels)
        axes[0].imshow(original_image)
        axes[0].set_title('Original Spectrogram (Clean)', fontsize=16, fontweight='bold')
        axes[0].axis('off')
        
        # Panel 2: Model predictions
        axes[1].imshow(original_image)
        axes[1].set_title('Model Predictions', fontsize=16, fontweight='bold')
        axes[1].axis('off')
        
        # Draw model predictions
        if len(predictions['boxes']) > 0:
            boxes = predictions['boxes'].cpu().numpy()
            confidences = predictions['confidences'].cpu().numpy()
            classes = predictions['classes'].cpu().numpy()
            
            for box, conf, cls in zip(boxes, confidences, classes):
                # Convert normalized coordinates to pixel coordinates
                h, w = original_image.shape[:2]
                x_center, y_center, width, height = box
                
                x1 = int((x_center - width/2) * w)
                y1 = int((y_center - height/2) * h)
                x2 = int((x_center + width/2) * w)
                y2 = int((y_center + height/2) * h)
                
                # Draw bounding box
                class_name = self.class_names[cls]
                color = self.class_colors[cls]
                
                rect = plt.Rectangle((x1, y1), x2-x1, y2-y1, 
                                   fill=False, color=color, linewidth=2)
                axes[1].add_patch(rect)
                
                # Add label (smaller, lighter, no box around text)
                axes[1].text(x1, y1-5, f'{class_name} {conf:.2f}', 
                           color=color, fontsize=8, fontweight='normal',
                           alpha=0.8)
        
        # Panel 3: Human marked image (ground truth)
        if marked_image is not None:
            axes[2].imshow(marked_image)
            axes[2].set_title('Human Marked (Ground Truth)', fontsize=16, fontweight='bold')
            axes[2].axis('off')
        else:
            axes[2].text(0.5, 0.5, 'No marked image available', 
                        ha='center', va='center', transform=axes[2].transAxes,
                        fontsize=14, fontweight='bold')
            axes[2].set_title('Human Marked (Not Available)', fontsize=16, fontweight='bold')
            axes[2].axis('off')
        
        plt.tight_layout()
        
        # Create output directory
        output_dir = "yolo_testing_results"
        os.makedirs(output_dir, exist_ok=True)
        
        # Save visualization
        output_path = os.path.join(output_dir, f"comparison_{base_name}.png")
        plt.savefig(output_path, dpi=300, bbox_inches='tight')
        print(f"3-panel comparison saved as: {output_path}")
        
        plt.show()
    
    def test_multiple_images(self, image_dir, num_images=10, conf_threshold=0.5):
        """Test model on multiple images"""
        # Get list of PNG files
        image_files = glob.glob(os.path.join(image_dir, '*.png'))
        image_files = [f for f in image_files if 'marked' not in f]  # Exclude marked images
        
        if len(image_files) == 0:
            print(f"No PNG files found in {image_dir}")
            return
        
        # Select random subset
        import random
        selected_files = random.sample(image_files, min(num_images, len(image_files)))
        
        print(f"Testing {len(selected_files)} images from {image_dir}")
        
        all_predictions = []
        for image_path in selected_files:
            predictions = self.test_single_image(image_path, conf_threshold)
            all_predictions.append((image_path, predictions))
        
        # Summary
        total_detections = sum(len(pred['boxes']) for _, pred in all_predictions)
        print(f"\n=== Test Summary ===")
        print(f"Total images tested: {len(selected_files)}")
        print(f"Total detections: {total_detections}")
        print(f"Average detections per image: {total_detections/len(selected_files):.2f}")
        
        # Save summary to file
        output_dir = "yolo_testing_results"
        os.makedirs(output_dir, exist_ok=True)
        
        summary_path = os.path.join(output_dir, "test_summary.txt")
        with open(summary_path, 'w') as f:
            f.write("YOLO Model Testing Summary\n")
            f.write("=" * 30 + "\n\n")
            f.write(f"Total images tested: {len(selected_files)}\n")
            f.write(f"Total detections: {total_detections}\n")
            f.write(f"Average detections per image: {total_detections/len(selected_files):.2f}\n\n")
            f.write("Individual Results:\n")
            f.write("-" * 20 + "\n")
            
            for i, (image_path, predictions) in enumerate(all_predictions):
                filename = os.path.basename(image_path)
                num_detections = len(predictions['boxes'])
                f.write(f"{i+1:2d}. {filename}: {num_detections} detections\n")
                
                if num_detections > 0:
                    boxes = predictions['boxes'].cpu().numpy()
                    confidences = predictions['confidences'].cpu().numpy()
                    classes = predictions['classes'].cpu().numpy()
                    
                    for j, (box, conf, cls) in enumerate(zip(boxes, confidences, classes)):
                        class_name = self.class_names[cls]
                        f.write(f"     {j+1}. {class_name} (conf: {conf:.3f})\n")
        
        print(f"Summary saved to: {summary_path}")

    def debug_model_output(self, image_path):
        """Debug function to see what the model is actually outputting"""
        print(f"\n=== DEBUGGING MODEL OUTPUT ===")
        print(f"Image: {os.path.basename(image_path)}")
        
        # Preprocess image
        image_tensor, original_size, original_image = self.preprocess_image(image_path)
        
        with torch.no_grad():
            raw_output = self.model(image_tensor)
            print(f"Raw output shape: {raw_output.shape}")
            
            # Check all anchors
            for anchor_idx in range(3):
                anchor_output = raw_output[0, anchor_idx]
                print(f"\nAnchor {anchor_idx}:")
                print(f"  Raw values: {anchor_output[:5]}")
                print(f"  Sigmoid confidence: {torch.sigmoid(anchor_output[4]):.6f}")
                print(f"  Class scores: {anchor_output[5:]}")
                print(f"  Softmax classes: {torch.softmax(anchor_output[5:], dim=0)}")
            
            # Test with very low threshold
            print(f"\nTesting with confidence threshold 0.01:")
            predictions = self.predict(image_tensor, conf_threshold=0.01, nms_threshold=0.4)
            if len(predictions) > 0 and len(predictions[0]['boxes']) > 0:
                print(f"Found {len(predictions[0]['boxes'])} detections with threshold 0.01")
            else:
                print("Still no detections even with threshold 0.01")

def main():
    parser = argparse.ArgumentParser(description='Test YOLO Model on Spectrograms')
    parser.add_argument('--model_path', type=str, default='checkpoints/best_model.pth',
                       help='Path to trained model')
    parser.add_argument('--image_path', type=str, default=None,
                       help='Path to single image to test')
    parser.add_argument('--image_dir', type=str, default='../spectrogram_training_data_20220711/results',
                       help='Directory containing images to test')
    parser.add_argument('--num_images', type=int, default=10,
                       help='Number of images to test')
    parser.add_argument('--conf_threshold', type=float, default=0.5,
                       help='Confidence threshold for detections')
    parser.add_argument('--nms_threshold', type=float, default=0.4,
                       help='NMS threshold for removing duplicate detections')
    parser.add_argument('--debug', action='store_true',
                       help='Enable debug mode to see raw model outputs')
    
    args = parser.parse_args()
    
    # Class names (3 classes)
    class_names = ['Background', 'WLAN', 'Bluetooth']
    
    # Create tester
    tester = YOLOTester(args.model_path, class_names)
    
    if args.debug:
        # Debug mode - test single image with detailed output
        if args.image_path and os.path.exists(args.image_path):
            tester.debug_model_output(args.image_path)
        else:
            # Find a random image to debug
            image_files = glob.glob(os.path.join(args.image_dir, '*.png'))
            image_files = [f for f in image_files if 'marked' not in f]
            if image_files:
                tester.debug_model_output(image_files[0])
            else:
                print("No images found for debugging")
    elif args.image_path:
        # Test single image
        if os.path.exists(args.image_path):
            tester.test_single_image(args.image_path, args.conf_threshold, args.nms_threshold)
        else:
            print(f"Image not found: {args.image_path}")
    else:
        # Test multiple images
        tester.test_multiple_images(args.image_dir, args.num_images, args.conf_threshold)

if __name__ == "__main__":
    main()
