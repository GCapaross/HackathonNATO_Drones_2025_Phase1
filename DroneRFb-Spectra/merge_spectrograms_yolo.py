#!/usr/bin/env python3
"""
Drone Spectrogram Merger for YOLO Training
===========================================

Merges random drone spectrograms into single images with YOLO format labels.
"""

import numpy as np
import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt
from pathlib import Path
from typing import List, Dict, Tuple
import random

# Class names
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


def load_all_samples(data_dir: Path) -> Dict[int, List[Path]]:
    """Load all .npy file paths organized by class."""
    samples = {}
    
    print("Loading available samples...")
    for class_id in range(24):
        class_dir = data_dir / str(class_id)
        if class_dir.exists():
            npy_files = list(class_dir.glob("*.npy"))
            if npy_files:
                samples[class_id] = npy_files
                print(f"  Class {class_id:2d} ({CLASS_NAMES[class_id]:25s}): {len(npy_files)} samples")
    
    return samples


def resize_spectrogram(spec: np.ndarray, target_width: int, target_height: int) -> np.ndarray:
    """Resize spectrogram to target dimensions."""
    from scipy.ndimage import zoom
    
    # Convert to float32 for scipy compatibility
    spec = spec.astype(np.float32)
    
    scale_x = target_width / spec.shape[0]
    scale_y = target_height / spec.shape[1]
    
    resized = zoom(spec, (scale_x, scale_y), order=1)
    return resized.astype(np.float32)


def merge_spectrograms(samples_dict: Dict[int, List[Path]], 
                      frame_width: int = 1024, 
                      frame_height: int = 512,
                      max_drones: int = 4,
                      min_drones: int = 1) -> Tuple[np.ndarray, List[Dict]]:
    """
    Merge random drone spectrograms - OVERLAPPING in same time-frequency space.
    Like real RF interference where multiple signals exist simultaneously.
    
    Returns:
        merged_frame: Combined spectrogram (frame_width x frame_height)
        labels: List of YOLO format labels with bounding boxes
    """
    # Create empty frame - use same size as original spectrograms
    merged_frame = np.zeros((frame_width, frame_height), dtype=np.float32)
    labels = []
    
    # Decide how many drones to merge
    num_drones = random.randint(min_drones, max_drones)
    
    # Select random classes (exclude background class 0 for now)
    available_classes = [c for c in samples_dict.keys() if c > 0]
    if not available_classes:
        return merged_frame, labels
    
    selected_classes = random.sample(available_classes, min(num_drones, len(available_classes)))
    
    for class_id in selected_classes:
        # Load random sample from this class
        sample_path = random.choice(samples_dict[class_id])
        spec = np.load(sample_path).astype(np.float32)
        
        # Resize to match frame dimensions
        spec_resized = resize_spectrogram(spec, frame_width, frame_height)
        
        # Apply random frequency shift (shift the signal up or down in frequency)
        freq_shift = random.randint(-frame_height // 4, frame_height // 4)
        shifted_spec = np.zeros_like(spec_resized)
        
        if freq_shift > 0:
            # Shift up
            shifted_spec[:, freq_shift:] = spec_resized[:, :-freq_shift]
        elif freq_shift < 0:
            # Shift down
            shifted_spec[:, :freq_shift] = spec_resized[:, -freq_shift:]
        else:
            shifted_spec = spec_resized
        
        # Random time offset (where signal starts)
        time_offset = random.randint(0, frame_width // 4)
        time_length = random.randint(frame_width // 2, frame_width - time_offset)
        
        # Create time-limited signal
        signal_with_time = np.zeros_like(shifted_spec)
        signal_with_time[time_offset:time_offset + time_length, :] = shifted_spec[time_offset:time_offset + time_length, :]
        
        # Add to merged frame with RF superposition (addition, not maximum)
        merged_frame = merged_frame + signal_with_time * 0.5
        
        # Find where the signal actually is (non-zero regions)
        signal_mask = signal_with_time > 0.1
        
        # Find bounding box of the signal
        if np.any(signal_mask):
            time_indices = np.where(np.any(signal_mask, axis=1))[0]
            freq_indices = np.where(np.any(signal_mask, axis=0))[0]
            
            if len(time_indices) > 0 and len(freq_indices) > 0:
                time_start = time_indices[0]
                time_end = time_indices[-1]
                freq_start = freq_indices[0]
                freq_end = freq_indices[-1]
            else:
                time_start, time_end = 0, frame_width
                freq_start, freq_end = 0, frame_height
        else:
            time_start, time_end = 0, frame_width
            freq_start, freq_end = 0, frame_height
        
        # Create YOLO label (normalized to [0, 1])
        x_center = (time_start + time_end) / 2 / frame_width
        y_center = (freq_start + freq_end) / 2 / frame_height
        width = (time_end - time_start) / frame_width
        height = (freq_end - freq_start) / frame_height
        
        labels.append({
            'class_id': class_id,
            'class_name': CLASS_NAMES[class_id],
            'x_center': x_center,
            'y_center': y_center,
            'width': width,
            'height': height
        })
    
    # Add background noise
    noise = np.random.normal(0, 0.05, merged_frame.shape).astype(np.float32)
    merged_frame = np.clip(merged_frame + noise, 0, 1)
    
    return merged_frame, labels


def save_spectrogram_image(spec: np.ndarray, output_path: Path):
    """Save spectrogram as PNG image with viridis colormap."""
    fig, ax = plt.subplots(figsize=(10.24, 5.12))
    
    # Transpose for correct orientation (frequency on Y-axis)
    ax.imshow(spec.T, aspect='auto', cmap='viridis', origin='lower', 
             interpolation='bilinear', vmin=0, vmax=1)
    
    # Clean image (no axes)
    fig.subplots_adjust(left=0, right=1, bottom=0, top=1)
    ax.axis("off")
    
    plt.savefig(output_path, dpi=100, bbox_inches='tight', pad_inches=0)
    plt.close('all')


def save_yolo_labels(labels: List[Dict], output_path: Path):
    """Save labels in YOLO format: class_id x_center y_center width height."""
    with open(output_path, 'w') as f:
        for label in labels:
            f.write(f"{label['class_id']} {label['x_center']:.6f} {label['y_center']:.6f} "
                   f"{label['width']:.6f} {label['height']:.6f}\n")


def save_visualization(spec: np.ndarray, labels: List[Dict], output_path: Path):
    """Save visualization with bounding boxes."""
    from matplotlib.patches import Rectangle
    
    fig, ax = plt.subplots(figsize=(12, 6))
    
    # Plot spectrogram
    ax.imshow(spec.T, aspect='auto', cmap='viridis', origin='lower', 
             interpolation='bilinear', vmin=0, vmax=1)
    
    # Get dimensions
    img_width = spec.shape[0]
    img_height = spec.shape[1]
    
    # Draw bounding boxes
    colors = plt.cm.tab20(np.linspace(0, 1, 24))
    
    for label in labels:
        class_id = label['class_id']
        
        # Convert normalized coordinates to pixels
        x_center_px = label['x_center'] * img_width
        y_center_px = label['y_center'] * img_height
        width_px = label['width'] * img_width
        height_px = label['height'] * img_height
        
        x1 = x_center_px - width_px / 2
        y1 = y_center_px - height_px / 2
        
        # Draw rectangle
        rect = Rectangle(
            (x1, y1), width_px, height_px,
            linewidth=2.5,
            edgecolor=colors[class_id % 20],
            facecolor='none',
            alpha=0.9
        )
        ax.add_patch(rect)
        
        # Add label text
        ax.text(
            x1, y1 - 10,
            label['class_name'],
            color=colors[class_id % 20],
            fontsize=10,
            fontweight='bold',
            bbox=dict(boxstyle="round,pad=0.4", facecolor='white', alpha=0.9)
        )
    
    ax.set_xlabel('Time Bins', fontsize=11)
    ax.set_ylabel('Frequency Bins', fontsize=11)
    ax.set_title(f'Merged Drone Signals - {len(labels)} drones detected', 
                fontsize=13, fontweight='bold')
    
    plt.tight_layout()
    plt.savefig(output_path, dpi=150, bbox_inches='tight')
    plt.close('all')


def generate_dataset(data_dir: Path, output_dir: Path, num_samples: int = 1000,
                    max_drones_per_image: int = 4, visualize_every: int = 20):
    """
    Generate YOLO dataset by merging drone spectrograms.
    
    Args:
        data_dir: Path to DroneRFb-Spectra/Data
        output_dir: Where to save the dataset
        num_samples: Number of merged images to create
        max_drones_per_image: Maximum drones per image
        visualize_every: Create visualization every N samples
    """
    print("=" * 70)
    print("Drone Spectrogram Merger for YOLO Training")
    print("=" * 70)
    print(f"Input directory: {data_dir}")
    print(f"Output directory: {output_dir}")
    print(f"Samples to generate: {num_samples}")
    print(f"Max drones per image: {max_drones_per_image}")
    print(f"Colormap: viridis")
    print()
    
    # Load available samples
    samples_dict = load_all_samples(data_dir)
    
    if not samples_dict:
        print("No samples found!")
        return
    
    # Create output directories
    images_dir = output_dir / "images"
    labels_dir = output_dir / "labels"
    viz_dir = output_dir / "visualizations"
    
    images_dir.mkdir(parents=True, exist_ok=True)
    labels_dir.mkdir(parents=True, exist_ok=True)
    viz_dir.mkdir(parents=True, exist_ok=True)
    
    print("\nGenerating merged spectrograms...")
    
    # Generate samples
    for i in range(num_samples):
        # Merge spectrograms
        merged_spec, labels = merge_spectrograms(
            samples_dict,
            frame_width=1024,
            frame_height=512,
            max_drones=max_drones_per_image,
            min_drones=1
        )
        
        if not labels:
            continue
        
        # Filenames
        base_name = f"drone_merged_{i:06d}"
        
        # Save image
        image_path = images_dir / f"{base_name}.png"
        save_spectrogram_image(merged_spec, image_path)
        
        # Save YOLO labels
        label_path = labels_dir / f"{base_name}.txt"
        save_yolo_labels(labels, label_path)
        
        # Save visualization (every N samples)
        if i % visualize_every == 0:
            viz_path = viz_dir / f"{base_name}_boxes.png"
            save_visualization(merged_spec, labels, viz_path)
        
        # Progress
        if (i + 1) % 50 == 0:
            print(f"  Generated {i + 1}/{num_samples} samples...")
    
    # Create dataset.yaml for YOLO
    yaml_path = output_dir / "dataset.yaml"
    with open(yaml_path, 'w') as f:
        f.write(f"# DroneRFb YOLO Dataset\n")
        f.write(f"path: {output_dir.absolute()}\n")
        f.write(f"train: images\n")
        f.write(f"val: images  # Split into train/val later\n\n")
        f.write(f"# Classes\n")
        f.write(f"nc: 24\n")
        f.write(f"names:\n")
        for i, name in enumerate(CLASS_NAMES):
            f.write(f"  {i}: {name}\n")
    
    print(f"\n{'=' * 70}")
    print("Dataset generation complete!")
    print(f"{'=' * 70}")
    print(f"Images saved to: {images_dir}")
    print(f"Labels saved to: {labels_dir}")
    print(f"Visualizations saved to: {viz_dir}")
    print(f"YOLO config: {yaml_path}")
    print(f"{'=' * 70}\n")


def main():
    """Main function."""
    data_dir = Path("/home/gabriel/Desktop/HackathonNATO_Drones_2025/DroneRFb-Spectra/Data")
    output_dir = Path("/home/gabriel/Desktop/HackathonNATO_Drones_2025/DroneRFb_YOLO_Dataset")
    
    generate_dataset(
        data_dir=data_dir,
        output_dir=output_dir,
        num_samples=1000,
        max_drones_per_image=4,
        visualize_every=20
    )


if __name__ == "__main__":
    main()
