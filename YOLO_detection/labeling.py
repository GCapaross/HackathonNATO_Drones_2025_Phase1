"""
Labeling Analysis Script for RF Signal Detection
Analyzes existing labeled data and visualizes bounding boxes
"""

import os
import glob
import numpy as np
import matplotlib.pyplot as plt
import matplotlib.patches as patches
from PIL import Image
import cv2
import random
from pathlib import Path

class SpectrogramLabelAnalyzer:
    """
    Analyzes spectrogram labeling data and creates visualizations
    """
    
    def __init__(self, data_dir):
        self.data_dir = data_dir
        self.results_dir = os.path.join(data_dir, 'results')
        
        # Class mapping
        self.class_names = ['Background', 'WLAN', 'Bluetooth', 'BLE']
        self.class_colors = {
            0: 'gray',      # Background
            1: 'red',       # WLAN
            2: 'blue',      # Bluetooth
            3: 'green'      # BLE
        }
        
        # Get all image files
        self.image_files = self._get_image_files()
        print(f"Found {len(self.image_files)} spectrogram images")
    
    def _get_image_files(self):
        """Get all PNG files (excluding marked ones)"""
        png_files = glob.glob(os.path.join(self.results_dir, '*.png'))
        # Filter out marked images and keep only the main spectrogram images
        png_files = [f for f in png_files if 'marked' not in f and f.endswith('.png')]
        return sorted(png_files)
    
    def load_yolo_labels(self, image_path):
        """Load YOLO format labels from corresponding .txt file"""
        label_path = image_path.replace('.png', '.txt')
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
                                'height': height,
                                'class_name': self.class_names[class_id]
                            })
        
        return labels
    
    def load_spectrogram_image(self, image_path):
        """Load spectrogram image"""
        image = Image.open(image_path).convert('RGB')
        return np.array(image)
    
    def load_marked_image(self, image_path):
        """Load marked image if it exists"""
        marked_path = image_path.replace('.png', '_marked.png')
        if os.path.exists(marked_path):
            image = Image.open(marked_path).convert('RGB')
            return np.array(image)
        return None
    
    def normalize_to_pixel_coords(self, labels, image_shape):
        """
        Convert normalized YOLO coordinates to pixel coordinates
        YOLO format: (x_center, y_center, width, height) - all normalized [0,1]
        Works with ANY image resolution dynamically
        """
        height, width = image_shape[:2]
        pixel_labels = []
        
        print(f"  Image resolution: {width}×{height}")
        
        for label in labels:
            # Convert normalized coordinates to pixels (resolution-agnostic)
            x_center_px = label['x_center'] * width
            y_center_px = label['y_center'] * height
            width_px = label['width'] * width
            height_px = label['height'] * height
            
            # Convert to corner coordinates (x1, y1, x2, y2)
            x1 = max(0, int(x_center_px - width_px / 2))
            y1 = max(0, int(y_center_px - height_px / 2))
            x2 = min(width, int(x_center_px + width_px / 2))
            y2 = min(height, int(y_center_px + height_px / 2))
            
            pixel_labels.append({
                'class_id': label['class_id'],
                'class_name': label['class_name'],
                'x1': x1,
                'y1': y1,
                'x2': x2,
                'y2': y2,
                'x_center': x_center_px,
                'y_center': y_center_px,
                'width': width_px,
                'height': height_px,
                'normalized_coords': (label['x_center'], label['y_center'], label['width'], label['height'])
            })
        
        return pixel_labels
    
    def _copy_image_files(self, image_path, destination_folder):
        """Copy all related files for an image to its folder"""
        import shutil
        
        base_name = os.path.splitext(os.path.basename(image_path))[0]
        
        # Files to copy
        files_to_copy = [
            (image_path, 'original_spectrogram.png'),  # Original PNG
            (image_path.replace('.png', '_marked.png'), 'ground_truth_marked.png'),  # Marked PNG
            (image_path.replace('.png', '.txt'), 'labels.txt'),  # TXT labels
            (image_path.replace('.png', ''), 'raw_data'),  # Raw data file (no extension)
        ]
        
        for source_path, dest_name in files_to_copy:
            if os.path.exists(source_path):
                dest_path = os.path.join(destination_folder, dest_name)
                try:
                    shutil.copy2(source_path, dest_path)
                    print(f"    Copied: {dest_name}")
                except Exception as e:
                    print(f"    Failed to copy {dest_name}: {e}")
            else:
                print(f"    Not found: {dest_name}")
    
    def _create_image_readme(self, image_path, image_folder):
        """Create a README file explaining the contents of this image folder"""
        base_name = os.path.splitext(os.path.basename(image_path))[0]
        
        readme_content = f"""# Image Analysis: {base_name}

## Files in this folder:

### 📁 Original Data
- **`original_spectrogram.png`**: The raw spectrogram image (1024×192 pixels)
- **`raw_data`**: Raw RF signal data (OpenPGP encrypted, ~900KB)

### 📁 Ground Truth
- **`ground_truth_marked.png`**: Human-labeled version with bounding boxes
- **`labels.txt`**: YOLO format labels (bounding box coordinates)

### 📁 Analysis
- **`labeling_comparison.png`**: 3-panel comparison showing:
  - Panel 1: Original spectrogram (clean)
  - Panel 2: Ground truth (marked image)
  - Panel 3: Our generated labels (from .txt file)

## Label Classes:
- **0**: Background/Noise (Gray)
- **1**: WLAN/WiFi (Red)
- **2**: Bluetooth Classic (Blue)
- **3**: BLE - Bluetooth Low Energy (Green)

## Purpose:
This folder contains all files related to one spectrogram sample, allowing you to:
1. Compare the original image with ground truth
2. Verify that our .txt coordinate conversion is accurate
3. Check if all signals are properly labeled
4. Analyze the quality of the labeling system

## Usage:
- Open `labeling_comparison.png` to see the 3-panel analysis
- Compare `original_spectrogram.png` with `ground_truth_marked.png`
- Check `labels.txt` for the exact YOLO coordinates
"""
        
        readme_path = os.path.join(image_folder, 'README.md')
        with open(readme_path, 'w') as f:
            f.write(readme_content)
        print(f"    Created README.md")
    
    def create_visualization(self, image_path, num_samples=10):
        """Create visualization for a single image"""
        # Load original image
        original_image = self.load_spectrogram_image(image_path)
        
        # Load marked image if available
        marked_image = self.load_marked_image(image_path)
        
        # Load labels
        labels = self.load_yolo_labels(image_path)
        
        if not labels:
            print(f"No labels found for {os.path.basename(image_path)}")
            return None
        
        # Convert to pixel coordinates
        pixel_labels = self.normalize_to_pixel_coords(labels, original_image.shape)
        
        # Create figure with 3 panels
        fig, axes = plt.subplots(1, 3, figsize=(24, 8))
        
        # Panel 1: Original spectrogram (clean, no labels)
        axes[0].imshow(original_image)
        axes[0].set_title('Original Spectrogram (Clean)', fontsize=16, fontweight='bold')
        axes[0].axis('off')
        
        # Add image info
        axes[0].text(0.02, 0.98, f'Resolution: {original_image.shape[1]}×{original_image.shape[0]}', 
                    transform=axes[0].transAxes, fontsize=12, fontweight='bold',
                    bbox=dict(boxstyle='round', facecolor='lightblue', alpha=0.8),
                    verticalalignment='top')
        
        # Panel 2: Ground truth (marked image)
        if marked_image is not None:
            axes[1].imshow(marked_image)
            axes[1].set_title('Ground Truth (Marked Image)', fontsize=16, fontweight='bold')
            axes[1].axis('off')
            
            # Add info about ground truth
            axes[1].text(0.02, 0.98, 'Human-labeled bounding boxes', 
                        transform=axes[1].transAxes, fontsize=12, fontweight='bold',
                        bbox=dict(boxstyle='round', facecolor='lightgreen', alpha=0.8),
                        verticalalignment='top')
        else:
            axes[1].text(0.5, 0.5, 'No marked image available', 
                        ha='center', va='center', transform=axes[1].transAxes,
                        fontsize=14, fontweight='bold')
            axes[1].set_title('Ground Truth (Not Available)', fontsize=16, fontweight='bold')
            axes[1].axis('off')
        
        # Panel 3: Our generated labels from .txt file
        axes[2].imshow(original_image)
        axes[2].set_title('Our Generated Labels (from .txt)', fontsize=16, fontweight='bold')
        axes[2].axis('off')
        
        # Draw our bounding boxes
        for i, label in enumerate(pixel_labels):
            x1, y1, x2, y2 = label['x1'], label['y1'], label['x2'], label['y2']
            class_id = label['class_id']
            class_name = label['class_name']
            
            # Create rectangle
            rect = patches.Rectangle(
                (x1, y1), x2-x1, y2-y1,
                linewidth=3,
                edgecolor=self.class_colors[class_id],
                facecolor='none',
                alpha=0.9
            )
            axes[2].add_patch(rect)
            
            # Add label text with better positioning (smaller, lighter, no box)
            text_x = x1
            text_y = max(y1 - 5, 15)  # Ensure text is visible
            
            axes[2].text(
                text_x, text_y, f'{class_name}',
                fontsize=8, fontweight='normal',
                color=self.class_colors[class_id],
                alpha=0.8
            )
        
        # Add summary info to panel 3
        axes[2].text(0.02, 0.98, f'Labels: {len(labels)} | Classes: {set([l["class_id"] for l in pixel_labels])}', 
                    transform=axes[2].transAxes, fontsize=12, fontweight='bold',
                    bbox=dict(boxstyle='round', facecolor='lightyellow', alpha=0.8),
                    verticalalignment='top')
        
        # Add class legend
        legend_text = "Classes:\n"
        for class_id, class_name in enumerate(self.class_names):
            if any(label['class_id'] == class_id for label in pixel_labels):
                legend_text += f"• {class_name} ({class_id})\n"
        
        axes[2].text(0.02, 0.02, legend_text, 
                    transform=axes[2].transAxes, fontsize=10,
                    bbox=dict(boxstyle='round', facecolor='lightgray', alpha=0.8),
                    verticalalignment='bottom')
        
        plt.tight_layout()
        return fig
    
    def analyze_dataset(self, num_samples=10):
        """Analyze the dataset and create visualizations"""
        print(f"Analyzing {num_samples} samples from the dataset...")
        
        # Select random samples
        sample_files = random.sample(self.image_files, min(num_samples, len(self.image_files)))
        
        # Create main output directory
        output_dir = 'labeling_analysis'
        os.makedirs(output_dir, exist_ok=True)
        
        # Statistics
        total_labels = 0
        class_counts = {0: 0, 1: 0, 2: 0, 3: 0}
        images_with_labels = 0
        
        for i, image_path in enumerate(sample_files):
            print(f"\nProcessing {i+1}/{len(sample_files)}: {os.path.basename(image_path)}")
            
            # Create individual folder for this image
            base_name = os.path.splitext(os.path.basename(image_path))[0]
            image_folder = os.path.join(output_dir, f'image_{i+1:02d}_{base_name}')
            os.makedirs(image_folder, exist_ok=True)
            
            # Copy all related files to the folder
            self._copy_image_files(image_path, image_folder)
            
            # Create README for this image folder
            self._create_image_readme(image_path, image_folder)
            
            # Load labels
            labels = self.load_yolo_labels(image_path)
            
            if labels:
                images_with_labels += 1
                total_labels += len(labels)
                
                # Count classes
                for label in labels:
                    class_counts[label['class_id']] += 1
                
                # Create visualization
                fig = self.create_visualization(image_path)
                if fig:
                    # Save visualization in the image folder
                    visualization_path = os.path.join(image_folder, 'labeling_comparison.png')
                    fig.savefig(visualization_path, dpi=300, bbox_inches='tight')
                    plt.close(fig)
                    print(f"  Saved visualization: {visualization_path}")
                else:
                    print(f"  No visualization created")
            else:
                print(f"  No labels found")
        
        # Print statistics
        print(f"\n=== Dataset Analysis Results ===")
        print(f"Total images analyzed: {len(sample_files)}")
        print(f"Images with labels: {images_with_labels}")
        print(f"Label coverage: {images_with_labels/len(sample_files)*100:.1f}%")
        print(f"Total labels: {total_labels}")
        print(f"Average labels per image: {total_labels/images_with_labels if images_with_labels > 0 else 0:.2f}")
        
        print(f"\nClass distribution:")
        for class_id, count in class_counts.items():
            class_name = self.class_names[class_id]
            percentage = count/total_labels*100 if total_labels > 0 else 0
            print(f"  {class_name}: {count} ({percentage:.1f}%)")
        
        print(f"\nVisualizations saved in: {output_dir}/")
        
        return {
            'total_images': len(sample_files),
            'images_with_labels': images_with_labels,
            'total_labels': total_labels,
            'class_counts': class_counts
        }
    
    def create_label_summary(self, image_path):
        """Create a summary of labels for a specific image"""
        labels = self.load_yolo_labels(image_path)
        pixel_labels = self.normalize_to_pixel_coords(labels, self.load_spectrogram_image(image_path).shape)
        
        print(f"\n=== Label Summary for {os.path.basename(image_path)} ===")
        print(f"Number of labels: {len(labels)}")
        
        for i, label in enumerate(pixel_labels):
            print(f"\nLabel {i+1}:")
            print(f"  Class: {label['class_name']} (ID: {label['class_id']})")
            print(f"  Normalized coords: ({label['x_center']/self.load_spectrogram_image(image_path).shape[1]:.3f}, {label['y_center']/self.load_spectrogram_image(image_path).shape[0]:.3f})")
            print(f"  Pixel coords: ({label['x1']}, {label['y1']}) to ({label['x2']}, {label['y2']})")
            print(f"  Size: {label['width']:.1f}×{label['height']:.1f} pixels")

def main():
    """Main function"""
    data_dir = "../spectrogram_training_data_20220711"
    
    print("=== RF Signal Labeling Analysis ===")
    print(f"Data directory: {data_dir}")
    
    # Create analyzer
    analyzer = SpectrogramLabelAnalyzer(data_dir)
    
    # Analyze dataset
    stats = analyzer.analyze_dataset(num_samples=10)
    
    # Show example of label conversion
    if analyzer.image_files:
        example_image = analyzer.image_files[0]
        print(f"\n=== Example Label Conversion ===")
        analyzer.create_label_summary(example_image)
    
    print(f"\nAnalysis complete! Check the 'labeling_analysis' folder for visualizations.")

if __name__ == "__main__":
    main()
