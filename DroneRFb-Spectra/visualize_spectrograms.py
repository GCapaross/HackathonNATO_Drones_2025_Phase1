#!/usr/bin/env python3
"""
DroneRFb-Spectra Visualization Tool
====================================

Visualize pre-computed spectrogram arrays (.npy files) from DroneRFb-Spectra dataset.
Creates PNG images with lighter background for better visibility.
"""

import numpy as np
import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt
from pathlib import Path
from typing import Optional

# Class labels
CLASS_NAMES = [
    "Background (WiFi/BT)",
    "DJI Phantom 3",
    "DJI Phantom 4 Pro",
    "DJI MATRICE 200",
    "DJI MATRICE 100",
    "DJI Air 2S",
    "DJI Mini 3 Pro",
    "DJI Inspire 2",
    "DJI Mavic Pro",
    "DJI Mini 2",
    "DJI Mavic 3",
    "DJI MATRICE 300",
    "DJI Phantom 4 Pro RTK",
    "DJI MATRICE 30T",
    "DJI AVATA",
    "DJI DIY",
    "DJI MATRICE 600 Pro",
    "VBar Controller",
    "FrSky X20 Controller",
    "Futaba T16IZ Controller",
    "Taranis Plus Controller",
    "RadioLink AT9S Controller",
    "Futaba T14SG Controller",
    "Skydroid Controller"
]


def create_spectrogram_visualization(npy_path: Path, output_path: Path, 
                                    class_id: Optional[int] = None,
                                    colormap: str = 'hot',
                                    show_grid: bool = False):
    """
    Create a visualization of a spectrogram with lighter background.
    
    Args:
        npy_path: Path to .npy file
        output_path: Path to save PNG
        class_id: Class ID for title (optional)
        colormap: Matplotlib colormap ('hot', 'viridis', 'plasma', 'inferno', 'magma')
        show_grid: Whether to show grid lines
    """
    try:
        # Load spectrogram
        spec = np.load(npy_path)
        
        # Create figure with white background
        fig, ax = plt.subplots(figsize=(12, 6), facecolor='white')
        
        # Plot spectrogram
        # Transpose so frequency is on Y-axis (more intuitive)
        im = ax.imshow(
            spec.T, 
            aspect='auto', 
            cmap=colormap,
            origin='lower',
            interpolation='bilinear'
        )
        
        # Colorbar
        cbar = plt.colorbar(im, ax=ax, label='Normalized Power')
        cbar.ax.tick_params(labelsize=10)
        
        # Labels
        ax.set_xlabel('Time Bins', fontsize=12, fontweight='bold')
        ax.set_ylabel('Frequency Bins', fontsize=12, fontweight='bold')
        
        # Title
        if class_id is not None and 0 <= class_id < len(CLASS_NAMES):
            title = f"Class {class_id}: {CLASS_NAMES[class_id]}"
        else:
            title = f"Spectrogram: {npy_path.stem}"
        
        ax.set_title(title, fontsize=14, fontweight='bold', pad=15)
        
        # Grid
        if show_grid:
            ax.grid(True, alpha=0.3, linestyle='--', linewidth=0.5)
        
        # Info text
        info_text = f"Shape: {spec.shape[0]} × {spec.shape[1]} | Range: [{spec.min():.3f}, {spec.max():.3f}]"
        ax.text(0.02, 0.98, info_text, transform=ax.transAxes,
                fontsize=9, verticalalignment='top',
                bbox=dict(boxstyle='round', facecolor='white', alpha=0.8))
        
        # Style improvements
        ax.spines['top'].set_visible(False)
        ax.spines['right'].set_visible(False)
        ax.tick_params(labelsize=10)
        
        # Save with white background
        plt.tight_layout()
        plt.savefig(output_path, dpi=150, bbox_inches='tight', 
                   facecolor='white', edgecolor='none')
        plt.close('all')
        
        return True
        
    except Exception as e:
        print(f"Error processing {npy_path}: {e}")
        return False


def visualize_samples_per_class(data_dir: Path, output_dir: Path, 
                                samples_per_class: int = 3,
                                colormap: str = 'hot'):
    """
    Visualize a few sample spectrograms from each class.
    
    Args:
        data_dir: Path to Data directory
        output_dir: Path to save visualizations
        samples_per_class: Number of samples to visualize per class
        colormap: Matplotlib colormap
    """
    output_dir.mkdir(parents=True, exist_ok=True)
    
    print("=" * 70)
    print("DroneRFb-Spectra Visualization Tool")
    print("=" * 70)
    print(f"Input directory: {data_dir}")
    print(f"Output directory: {output_dir}")
    print(f"Colormap: {colormap}")
    print(f"Samples per class: {samples_per_class}")
    print()
    
    total_visualized = 0
    
    for class_id in range(24):
        class_dir = data_dir / str(class_id)
        
        if not class_dir.exists():
            print(f"Class {class_id}: Directory not found, skipping")
            continue
        
        # Get .npy files
        npy_files = list(class_dir.glob("*.npy"))
        
        if not npy_files:
            print(f"Class {class_id}: No .npy files found")
            continue
        
        # Create class output directory
        class_output_dir = output_dir / f"class_{class_id}"
        class_output_dir.mkdir(parents=True, exist_ok=True)
        
        # Visualize samples
        num_samples = min(samples_per_class, len(npy_files))
        print(f"Class {class_id:2d} ({CLASS_NAMES[class_id]:30s}): "
              f"Visualizing {num_samples}/{len(npy_files)} samples...")
        
        for i, npy_file in enumerate(npy_files[:num_samples]):
            output_file = class_output_dir / f"{npy_file.stem}_visualization.png"
            
            success = create_spectrogram_visualization(
                npy_file, 
                output_file, 
                class_id=class_id,
                colormap=colormap
            )
            
            if success:
                total_visualized += 1
    
    print()
    print("=" * 70)
    print(f"Visualization complete! Created {total_visualized} images")
    print(f"Output directory: {output_dir}")
    print("=" * 70)


def create_comparison_grid(data_dir: Path, output_path: Path, 
                          samples_per_class: int = 1,
                          colormap: str = 'hot'):
    """
    Create a grid comparing one sample from each class.
    
    Args:
        data_dir: Path to Data directory
        output_path: Path to save comparison grid
        samples_per_class: Number of samples to show per class
        colormap: Matplotlib colormap
    """
    print("\nCreating comparison grid...")
    
    # Calculate grid layout
    num_classes = 24
    cols = 4
    rows = (num_classes + cols - 1) // cols
    
    fig, axes = plt.subplots(rows, cols, figsize=(20, 24), facecolor='white')
    axes = axes.flatten()
    
    class_idx = 0
    for class_id in range(24):
        class_dir = data_dir / str(class_id)
        
        if not class_dir.exists():
            continue
        
        npy_files = list(class_dir.glob("*.npy"))
        
        if not npy_files:
            continue
        
        # Load first sample
        spec = np.load(npy_files[0])
        
        # Plot
        ax = axes[class_idx]
        im = ax.imshow(spec.T, aspect='auto', cmap=colormap, origin='lower')
        ax.set_title(f"Class {class_id}\n{CLASS_NAMES[class_id]}", 
                    fontsize=10, fontweight='bold')
        ax.set_xlabel('Time', fontsize=8)
        ax.set_ylabel('Freq', fontsize=8)
        ax.tick_params(labelsize=7)
        
        class_idx += 1
    
    # Hide unused subplots
    for i in range(class_idx, len(axes)):
        axes[i].axis('off')
    
    plt.suptitle('DroneRFb-Spectra: All 24 Classes Comparison', 
                fontsize=16, fontweight='bold', y=0.995)
    plt.tight_layout()
    plt.savefig(output_path, dpi=150, bbox_inches='tight',
               facecolor='white', edgecolor='none')
    plt.close('all')
    
    print(f"Saved comparison grid to: {output_path}")


def main():
    """Main function."""
    # Paths
    data_dir = Path("/home/gabriel/Desktop/HackathonNATO_Drones_2025/DroneRFb-Spectra/Data")
    output_dir = Path("/home/gabriel/Desktop/HackathonNATO_Drones_2025/DroneRFb-Spectra/Visualizations")
    
    # Colormap options: 'hot', 'viridis', 'plasma', 'inferno', 'magma', 'cividis'
    # 'viridis' is nice blue-green-yellow, perceptually uniform
    colormap = 'viridis'
    
    # Create visualizations
    visualize_samples_per_class(
        data_dir=data_dir,
        output_dir=output_dir,
        samples_per_class=3,  # Adjust this to visualize more/less samples
        colormap=colormap
    )
    
    # Create comparison grid
    comparison_path = output_dir / "all_classes_comparison.png"
    create_comparison_grid(
        data_dir=data_dir,
        output_path=comparison_path,
        colormap=colormap
    )


if __name__ == "__main__":
    main()
