"""Spectrogram image generator with YOLO labels

This script creates spectrograms with YOLO bounding box labels overlaid.
It follows the exact same process as generator.py but adds bounding box visualization.
"""

import multiprocessing
import warnings
from dataclasses import dataclass
from pathlib import Path

import matplotlib
import matplotlib.pyplot as plt
import matplotlib.patches as patches
import numpy as np
from joblib import Parallel, delayed

from spectrogram_images.sample_files_list import SampleFileList

# Switch to non-interactive backend for faster saving the spectrogram image to file
matplotlib.use("Agg")


@dataclass
class Resolution:
    """Container for image resolution - number of pixels in x and y."""

    x: int
    y: int

    def __post_init__(self):
        if (self.x % 32 != 0) or (self.y % 32 != 0):
            warnings.warn(
                "Resolution must be a multiple of 32 when used for darknet yolo"
                " algorithm!"
            )


def parse_yolo_labels(label_file: Path):
    """Parse YOLO format labels from text file"""
    labels = []
    if not label_file.exists():
        return labels
    
    with open(label_file, 'r') as f:
        for line in f:
            line = line.strip()
            if line:
                parts = line.split()
                if len(parts) >= 5:
                    labels.append({
                        'class_id': int(parts[0]),
                        'x_center': float(parts[1]),
                        'y_center': float(parts[2]),
                        'width': float(parts[3]),
                        'height': float(parts[4])
                    })
    return labels


def get_class_color(class_id: int) -> str:
    """Get color for bounding box based on class"""
    colors = {
        0: "red",        # WLAN
        1: "yellow",     # collision
        2: "blue",       # BT_classic
        3: "green",      # BLE_1MHz
        4: "purple"      # BLE_2MHz
    }
    return colors.get(class_id, "orange")


def get_class_name(class_id: int) -> str:
    """Get class name for bounding box label"""
    class_names = {
        0: "WLAN",
        1: "collision", 
        2: "BT_classic",
        3: "BLE_1MHz",
        4: "BLE_2MHz"
    }
    return class_names.get(class_id, f"class_{class_id}")


def make_spectrograms_with_labels(
    sample_files: SampleFileList,
    png_resolution: Resolution,
    colormap: matplotlib.colors.Colormap = "viridis",
    auto_normalization=False,
    results2_dir: str = "results2"
):
    """
    It takes a list of sample files, creates a spectrogram for each file with bounding boxes, and saves the spectrogram as a PNG file
    
    This is the EXACT same code as generator.py but with added bounding box visualization.
    """
    print("creating spectrograms with labels...")
    
    # Create results2 directory
    results2_path = Path(results2_dir)
    results2_path.mkdir(exist_ok=True)
    
    if auto_normalization:
        norm_vmin = None
        norm_vmax = None
    else:
        norm_vmin = -150
        norm_vmax = -50

    def __spectrogram_with_labels(
        file_path: Path,
        png_resolution: Resolution,
        fft_size,
        norm_vmin=None,
        norm_vmax=None,
    ):
        # Read samples from file (EXACT same as original)
        result_samples = np.fromfile(file_path, dtype=np.complex64)
        
        # generate spectrogram (EXACT same as original)
        fig, ax = plt.subplots(figsize=(png_resolution.x / 100, png_resolution.y / 100))
        ax.specgram(
            result_samples,
            NFFT=fft_size,
            Fs=42e6,  # Dummy sample rate
            noverlap=0,  # int(FFT_size / 4),
            mode="default",
            sides="default",
            vmin=norm_vmin,
            vmax=norm_vmax,
            window=np.hanning(fft_size),
            cmap=colormap,
        )
        
        # ADD: Read and draw bounding boxes BEFORE turning off axis
        label_file = file_path.with_suffix('.txt')
        yolo_labels = parse_yolo_labels(label_file)
        
        print(f"Processing {file_path.name}: Found {len(yolo_labels)} labels")
        
        for label in yolo_labels:
            # Convert YOLO coordinates to pixel coordinates
            x_center = label['x_center'] * png_resolution.x
            y_center = label['y_center'] * png_resolution.y
            width = label['width'] * png_resolution.x
            height = label['height'] * png_resolution.y
            
            x_min = x_center - (width / 2)
            y_min = y_center - (height / 2)
            
            print(f"  Drawing box: class={label['class_id']}, x={x_min:.1f}, y={y_min:.1f}, w={width:.1f}, h={height:.1f}")
            
            # Draw rectangle
            rect = patches.Rectangle(
                (x_min, y_min), width, height,
                linewidth=3,
                edgecolor=get_class_color(label['class_id']),
                facecolor='none'
            )
            ax.add_patch(rect)
            
            # Add class label text
            class_name = get_class_name(label['class_id'])
            ax.text(x_min, y_min - 5, class_name, 
                   color=get_class_color(label['class_id']),
                   fontsize=10, fontweight='bold',
                   bbox=dict(boxstyle="round,pad=0.3", 
                           facecolor='white', 
                           edgecolor=get_class_color(label['class_id']),
                           alpha=0.8))
        
        # turn axis off (EXACT same as original) - AFTER drawing bounding boxes
        fig.subplots_adjust(left=0, right=1, bottom=0, top=1)
        ax.axis("tight")
        ax.axis("off")
        
        # Save to results2 directory (ONLY change from original)
        output_file = results2_path / f"{file_path.stem}_marked.png"
        plt.savefig(output_file, dpi=100)
        plt.close("all")

    # Perform multithreaded spectrogram generation (EXACT same as original)
    num_cores = multiprocessing.cpu_count()
    Parallel(n_jobs=num_cores)(
        delayed(__spectrogram_with_labels)(
            file_path=sample_file.path,
            png_resolution=png_resolution,
            fft_size=256 if sample_file.sample_rate >= 40 else 128,
            norm_vmin=norm_vmin,
            norm_vmax=norm_vmax,
        )
        for sample_file in sample_files
    )