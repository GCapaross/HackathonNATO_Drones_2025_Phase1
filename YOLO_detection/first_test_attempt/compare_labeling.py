"""
Compare our bounding box visualizations with the marked ground truth images
"""

import os
import glob
import numpy as np
import matplotlib.pyplot as plt
from PIL import Image
import cv2

def compare_with_marked_images(data_dir, num_samples=5):
    """
    Compare our bounding box visualizations with the marked ground truth images
    """
    results_dir = os.path.join(data_dir, 'results')
    
    # Get image files
    image_files = glob.glob(os.path.join(results_dir, '*.png'))
    image_files = [f for f in image_files if 'marked' not in f]
    
    # Select random samples
    import random
    sample_files = random.sample(image_files, min(num_samples, len(image_files)))
    
    print(f"Comparing {len(sample_files)} samples with marked images...")
    
    for i, image_path in enumerate(sample_files):
        print(f"\n=== Sample {i+1}: {os.path.basename(image_path)} ===")
        
        # Load original image
        original = Image.open(image_path).convert('RGB')
        original_np = np.array(original)
        
        # Load marked image
        marked_path = image_path.replace('.png', '_marked.png')
        if os.path.exists(marked_path):
            marked = Image.open(marked_path).convert('RGB')
            marked_np = np.array(marked)
            
            # Create comparison
            fig, axes = plt.subplots(1, 3, figsize=(18, 6))
            
            # Original image
            axes[0].imshow(original_np)
            axes[0].set_title('Original Spectrogram', fontsize=14, fontweight='bold')
            axes[0].axis('off')
            
            # Marked image (ground truth)
            axes[1].imshow(marked_np)
            axes[1].set_title('Ground Truth (Marked)', fontsize=14, fontweight='bold')
            axes[1].axis('off')
            
            # Side-by-side comparison
            axes[2].imshow(original_np)
            axes[2].set_title('Original + Our Bounding Boxes', fontsize=14, fontweight='bold')
            axes[2].axis('off')
            
            # Load and draw our bounding boxes
            label_path = image_path.replace('.png', '.txt')
            if os.path.exists(label_path):
                labels = load_yolo_labels(label_path)
                pixel_labels = normalize_to_pixel_coords(labels, original_np.shape)
                
                # Draw bounding boxes
                class_colors = {0: 'gray', 1: 'red', 2: 'blue', 3: 'green'}
                class_names = ['Background', 'WLAN', 'Bluetooth', 'BLE']
                
                for label in pixel_labels:
                    x1, y1, x2, y2 = label['x1'], label['y1'], label['x2'], label['y2']
                    class_id = label['class_id']
                    class_name = label['class_name']
                    
                    # Create rectangle
                    from matplotlib.patches import Rectangle
                    rect = Rectangle(
                        (x1, y1), x2-x1, y2-y1,
                        linewidth=2,
                        edgecolor=class_colors[class_id],
                        facecolor='none',
                        alpha=0.8
                    )
                    axes[2].add_patch(rect)
                    
                    # Add label text
                    axes[2].text(
                        x1, y1-5, f'{class_name} ({class_id})',
                        fontsize=10, fontweight='bold',
                        color=class_colors[class_id],
                        bbox=dict(boxstyle='round,pad=0.3', facecolor='white', alpha=0.8)
                    )
                
                print(f"  Found {len(labels)} labels")
                for j, label in enumerate(pixel_labels):
                    print(f"    Label {j+1}: {label['class_name']} at ({label['x1']}, {label['y1']}) to ({label['x2']}, {label['y2']})")
            
            plt.tight_layout()
            
            # Save comparison
            output_path = f'comparison_{i+1:02d}_{os.path.basename(image_path)}'
            plt.savefig(f'{output_path}.png', dpi=300, bbox_inches='tight')
            plt.close()
            print(f"  Saved comparison: {output_path}.png")
            
        else:
            print(f"  No marked image found for {os.path.basename(image_path)}")

def load_yolo_labels(label_path):
    """Load YOLO format labels"""
    labels = []
    if os.path.exists(label_path):
        with open(label_path, 'r') as f:
            for line in f:
                line = line.strip()
                if line:
                    parts = line.split()
                    if len(parts) >= 5:
                        class_id = int(parts[0])
                        x_center = float(parts[1])
                        y_center = float(parts[2])
                        width = float(parts[3])
                        height = float(parts[4])
                        
                        labels.append({
                            'class_id': class_id,
                            'x_center': x_center,
                            'y_center': y_center,
                            'width': width,
                            'height': height
                        })
    return labels

def normalize_to_pixel_coords(labels, image_shape):
    """Convert normalized coordinates to pixel coordinates"""
    height, width = image_shape[:2]
    pixel_labels = []
    
    for label in labels:
        # Convert normalized coordinates to pixels
        x_center_px = label['x_center'] * width
        y_center_px = label['y_center'] * height
        width_px = label['width'] * width
        height_px = label['height'] * height
        
        # Convert to corner coordinates
        x1 = int(x_center_px - width_px / 2)
        y1 = int(y_center_px - height_px / 2)
        x2 = int(x_center_px + width_px / 2)
        y2 = int(y_center_px + height_px / 2)
        
        pixel_labels.append({
            'class_id': label['class_id'],
            'class_name': ['Background', 'WLAN', 'Bluetooth', 'BLE'][label['class_id']],
            'x1': x1, 'y1': y1, 'x2': x2, 'y2': y2
        })
    
    return pixel_labels

if __name__ == "__main__":
    data_dir = "../spectrogram_training_data_20220711"
    compare_with_marked_images(data_dir, num_samples=5)
