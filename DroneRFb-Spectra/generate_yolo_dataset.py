#!/usr/bin/env python3
"""
Generate YOLO Dataset from DroneRFb Spectrograms
=================================================

Converts .npy spectrograms to proper RGB visualization images
and creates YOLO labels for training.

This creates the dataset first, then you can train with train_yolo_single_drone.py
"""

import numpy as np
import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt
from pathlib import Path
from sklearn.model_selection import train_test_split
import yaml

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


def create_spectrogram_image(spec: np.ndarray, output_path: Path, 
                             img_size: int = 640, colormap: str = 'viridis'):
    """
    Create a proper spectrogram visualization image.
    Similar to visualize_spectrograms.py but without axes/labels for YOLO.
    
    Args:
        spec: Spectrogram array (time x frequency)
        output_path: Where to save the image
        img_size: Output image size (square)
        colormap: Matplotlib colormap
    """
    # Create figure without axes for clean image
    dpi = 100
    fig_width = img_size / dpi
    fig_height = img_size / dpi
    
    fig = plt.figure(figsize=(fig_width, fig_height), dpi=dpi, facecolor='white')
    ax = plt.Axes(fig, [0., 0., 1., 1.])
    ax.set_axis_off()
    fig.add_axes(ax)
    
    # Plot spectrogram (transpose so frequency is on Y-axis)
    ax.imshow(spec.T, aspect='auto', cmap=colormap, origin='lower', 
              interpolation='bilinear', vmin=0, vmax=1)
    
    # Save as RGB image
    plt.savefig(output_path, dpi=dpi, bbox_inches='tight', pad_inches=0, 
                facecolor='white', edgecolor='none')
    plt.close('all')
    
    # Verify size and resize if needed
    try:
        import cv2
        img = cv2.imread(str(output_path))
        if img is not None and (img.shape[0] != img_size or img.shape[1] != img_size):
            img = cv2.resize(img, (img_size, img_size))
            cv2.imwrite(str(output_path), img)
    except ImportError:
        print("Warning: opencv not installed, skipping resize verification")


def generate_yolo_dataset(data_dir: Path, output_dir: Path, 
                          img_size: int = 640, colormap: str = 'viridis'):
    """
    Generate complete YOLO dataset from DroneRFb spectrograms.
    
    Creates:
    - train/images/*.jpg - Training spectrogram images
    - train/labels/*.txt - YOLO format labels
    - val/images/*.jpg - Validation images
    - val/labels/*.txt - Validation labels
    - test/images/*.jpg - Test images
    - test/labels/*.txt - Test labels
    - dataset.yaml - YOLO configuration
    """
    print("=" * 70)
    print("DroneRFb YOLO Dataset Generator")
    print("=" * 70)
    print(f"Input directory: {data_dir}")
    print(f"Output directory: {output_dir}")
    print(f"Image size: {img_size}x{img_size}")
    print(f"Colormap: {colormap}")
    print()
    
    # Create output directories
    train_images = output_dir / 'train' / 'images'
    train_labels = output_dir / 'train' / 'labels'
    val_images = output_dir / 'val' / 'images'
    val_labels = output_dir / 'val' / 'labels'
    test_images = output_dir / 'test' / 'images'
    test_labels = output_dir / 'test' / 'labels'
    
    for d in [train_images, train_labels, val_images, val_labels, test_images, test_labels]:
        d.mkdir(parents=True, exist_ok=True)
    
    # Collect all samples
    print("Scanning dataset...")
    all_samples = []
    class_counts = {}
    
    for class_id in range(24):
        class_dir = data_dir / str(class_id)
        if not class_dir.exists():
            continue
        
        npy_files = list(class_dir.glob("*.npy"))
        class_counts[class_id] = len(npy_files)
        
        print(f"  Class {class_id:2d} ({CLASS_NAMES[class_id]:30s}): {len(npy_files)} samples")
        
        for npy_file in npy_files:
            all_samples.append((npy_file, class_id))
    
    print(f"\nTotal samples: {len(all_samples)}")
    print(f"Classes found: {len(class_counts)}")
    
    # Split into train/val/test (70/15/15)
    print("\nSplitting dataset...")
    train_samples, temp_samples = train_test_split(all_samples, test_size=0.3, 
                                                    random_state=42, shuffle=True)
    val_samples, test_samples = train_test_split(temp_samples, test_size=0.5, 
                                                  random_state=42, shuffle=True)
    
    print(f"  Train: {len(train_samples)} samples ({len(train_samples)/len(all_samples)*100:.1f}%)")
    print(f"  Val:   {len(val_samples)} samples ({len(val_samples)/len(all_samples)*100:.1f}%)")
    print(f"  Test:  {len(test_samples)} samples ({len(test_samples)/len(all_samples)*100:.1f}%)")
    print()
    
    # Process each split
    for split_name, samples, img_dir, lbl_dir in [
        ('train', train_samples, train_images, train_labels),
        ('val', val_samples, val_images, val_labels),
        ('test', test_samples, test_images, test_labels)
    ]:
        print(f"Generating {split_name} split...")
        success_count = 0
        
        for idx, (npy_file, class_id) in enumerate(samples):
            try:
                # Load spectrogram
                spec = np.load(npy_file).astype(np.float32)
                
                # Normalize to [0, 1]
                if spec.max() > spec.min():
                    spec = (spec - spec.min()) / (spec.max() - spec.min())
                else:
                    spec = np.zeros_like(spec)
                
                # Create visualization image
                img_filename = f"{split_name}_{class_id:02d}_{idx:05d}.jpg"
                img_path = img_dir / img_filename
                
                create_spectrogram_image(spec, img_path, img_size, colormap)
                
                # Create YOLO label
                # For single-object classification, one box covering the entire image
                # YOLO format: class_id x_center y_center width height (normalized 0-1)
                label_content = f"{class_id} 0.5 0.5 1.0 1.0\n"
                
                lbl_filename = f"{split_name}_{class_id:02d}_{idx:05d}.txt"
                lbl_path = lbl_dir / lbl_filename
                with open(lbl_path, 'w') as f:
                    f.write(label_content)
                
                success_count += 1
                
                # Progress indicator
                if (idx + 1) % 100 == 0:
                    print(f"  Progress: {idx + 1}/{len(samples)} samples")
                
            except Exception as e:
                print(f"  Error processing {npy_file}: {e}")
        
        print(f"  Successfully generated {success_count}/{len(samples)} {split_name} samples")
        print()
    
    # Create YOLO config file
    print("Creating dataset.yaml...")
    yaml_content = {
        'path': str(output_dir.absolute()),
        'train': 'train/images',
        'val': 'val/images',
        'test': 'test/images',
        'nc': 24,
        'names': CLASS_NAMES
    }
    
    yaml_path = output_dir / 'dataset.yaml'
    with open(yaml_path, 'w') as f:
        yaml.dump(yaml_content, f, default_flow_style=False, sort_keys=False)
    
    print(f"  Created {yaml_path}")
    
    # Create summary file
    summary_path = output_dir / 'dataset_summary.txt'
    with open(summary_path, 'w') as f:
        f.write("DroneRFb-Spectra YOLO Dataset Summary\n")
        f.write("=" * 70 + "\n\n")
        f.write(f"Generated from: {data_dir}\n")
        f.write(f"Output directory: {output_dir}\n")
        f.write(f"Image size: {img_size}x{img_size}\n")
        f.write(f"Colormap: {colormap}\n\n")
        f.write(f"Total samples: {len(all_samples)}\n")
        f.write(f"Training samples: {len(train_samples)}\n")
        f.write(f"Validation samples: {len(val_samples)}\n")
        f.write(f"Test samples: {len(test_samples)}\n\n")
        f.write("Classes:\n")
        for class_id, count in sorted(class_counts.items()):
            f.write(f"  {class_id:2d}. {CLASS_NAMES[class_id]:30s} - {count} samples\n")
    
    print(f"  Created {summary_path}")
    
    print("\n" + "=" * 70)
    print("Dataset generation complete")
    print("=" * 70)
    print(f"Dataset location: {output_dir}")
    print(f"Config file: {yaml_path}")
    print(f"\nNext step: Train YOLO model with:")
    print(f"  python3 train_yolo_single_drone.py")
    print("=" * 70)


def main():
    """Main function."""
    # Configuration
    data_dir = Path("Data")
    output_dir = Path("YOLO_Dataset")
    img_size = 640
    colormap = 'viridis'  # Options: 'viridis', 'plasma', 'inferno', 'magma', 'hot'
    
    # Check if data directory exists
    if not data_dir.exists():
        print(f"ERROR: Data directory not found: {data_dir}")
        print("Please run this script from the DroneRFb-Spectra directory")
        return 1
    
    # Generate dataset
    generate_yolo_dataset(data_dir, output_dir, img_size, colormap)
    
    return 0


if __name__ == "__main__":
    exit(main())

