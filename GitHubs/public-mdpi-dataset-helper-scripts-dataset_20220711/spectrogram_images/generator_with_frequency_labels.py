"""Enhanced spectrogram generator with frequency labels and YOLO bounding boxes

This script creates spectrograms with:
1. Frequency axis labels (Hz)
2. YOLO bounding box labels overlaid
3. Proper coordinate system handling
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


def make_spectrograms_with_frequency_labels(
    sample_files: SampleFileList,
    png_resolution: Resolution,
    colormap: matplotlib.colors.Colormap = "viridis",
    auto_normalization=False,
    results2_dir: str = "results2",
    show_frequency_axis=True,
    center_freq=2.412e9  # Default center frequency in Hz
):
    """
    Enhanced spectrogram generator with frequency labels and YOLO bounding boxes
    
    Args:
        sample_files: List of sample files to process
        png_resolution: Output image resolution
        colormap: Matplotlib colormap
        auto_normalization: Whether to auto-normalize
        results2_dir: Output directory
        show_frequency_axis: Whether to show frequency axis labels
        center_freq: Center frequency in Hz for frequency calculations
    """
    print("creating spectrograms with frequency labels and YOLO bounding boxes...")
    
    # Create output directory
    results2_path = Path(results2_dir)
    results2_path.mkdir(exist_ok=True)
    
    if auto_normalization:
        norm_vmin = None
        norm_vmax = None
    else:
        norm_vmin = -150
        norm_vmax = -50

    def __spectrogram_with_frequency_labels(
        file_path: Path,
        png_resolution: Resolution,
        fft_size,
        norm_vmin=None,
        norm_vmax=None,
    ):
        # Read samples from file
        result_samples = np.fromfile(file_path, dtype=np.complex64)
        print(f"Loaded {len(result_samples)} samples from {file_path.name}")
        
        # Calculate actual sample rate from filename
        sample_rate = 42e6  # Default, will be updated from filename
        if 'bw_25E+6' in file_path.name:
            sample_rate = 25e6
        elif 'bw_45E+6' in file_path.name:
            sample_rate = 45e6
        elif 'bw_60E+6' in file_path.name:
            sample_rate = 60e6
        elif 'bw_125E+6' in file_path.name:
            sample_rate = 125e6
        
        # Generate spectrogram with proper frequency axis
        fig, ax = plt.subplots(figsize=(png_resolution.x / 100, png_resolution.y / 100))
        
        # Create spectrogram - store the return values to ensure it's generated
        Pxx, freqs, bins, im = ax.specgram(
            result_samples,
            NFFT=fft_size,
            Fs=sample_rate,  # Use actual sample rate
            noverlap=0,
            mode="default",
            sides="default",
            vmin=norm_vmin,
            vmax=norm_vmax,
            window=np.hanning(fft_size),
            cmap=colormap,
        )
        
        # Ensure the spectrogram is visible and properly scaled
        ax.set_ylim(0, len(freqs)-1)
        ax.set_xlim(0, len(bins)-1)
        
        # Force the image to be displayed with proper scaling
        if im is not None:
            im.set_clim(vmin=norm_vmin, vmax=norm_vmax)
        
        print(f"Spectrogram generated: {len(freqs)} frequency bins, {len(bins)} time bins")
        print(f"Pxx shape: {Pxx.shape}, min: {np.min(Pxx):.2f}, max: {np.max(Pxx):.2f}")
        print(f"Image object: {im is not None}, vmin: {norm_vmin}, vmax: {norm_vmax}")
        
        # Add frequency axis labels if requested
        if show_frequency_axis:
            # Calculate frequency range
            freq_min = center_freq - sample_rate/2
            freq_max = center_freq + sample_rate/2
            
            # Set frequency axis labels
            ax.set_ylabel('Frequency (MHz)', fontsize=8)
            ax.set_xlabel('Time (s)', fontsize=8)
            
            # Set frequency ticks
            freq_ticks = np.linspace(0, len(freqs)-1, 5)
            freq_labels = [f"{(freq_min + freqs[int(tick)]*1e6)/1e6:.1f}" for tick in freq_ticks]
            ax.set_yticks(freq_ticks)
            ax.set_yticklabels(freq_labels, fontsize=6)
            
            # Set time ticks
            time_ticks = np.linspace(0, len(bins)-1, 5)
            time_labels = [f"{bins[int(tick)]:.3f}" for tick in time_ticks]
            ax.set_xticks(time_ticks)
            ax.set_xticklabels(time_labels, fontsize=6)
        
        # Read and draw YOLO bounding boxes
        label_file = file_path.with_suffix('.txt')
        yolo_labels = parse_yolo_labels(label_file)
        
        print(f"Processing {file_path.name}: Found {len(yolo_labels)} labels, Sample rate: {sample_rate/1e6:.1f} MHz")
        
        for label in yolo_labels:
            # Convert YOLO coordinates to spectrogram coordinates
            # YOLO coordinates are normalized (0-1), convert to spectrogram data coordinates
            x_center = label['x_center'] * len(bins)
            y_center = label['y_center'] * len(freqs)
            width = label['width'] * len(bins)
            height = label['height'] * len(freqs)
            
            x_min = x_center - (width / 2)
            y_min = y_center - (height / 2)
            
            print(f"  Drawing box: class={label['class_id']}, x={x_min:.1f}, y={y_min:.1f}, w={width:.1f}, h={height:.1f}")
            
            # Draw rectangle
            rect = patches.Rectangle(
                (x_min, y_min), width, height,
                linewidth=2,
                edgecolor=get_class_color(label['class_id']),
                facecolor='none'
            )
            ax.add_patch(rect)
            
            # Add class label text
            class_name = get_class_name(label['class_id'])
            ax.text(x_min, y_min - 5, class_name, 
                   color=get_class_color(label['class_id']),
                   fontsize=8, fontweight='bold',
                   bbox=dict(boxstyle="round,pad=0.2", 
                           facecolor='white', 
                           edgecolor=get_class_color(label['class_id']),
                           alpha=0.8))
        
        # Adjust layout and save
        if show_frequency_axis:
            fig.subplots_adjust(left=0.1, right=0.95, bottom=0.1, top=0.95)
        else:
            # Don't turn off axis completely - just make it tight
            fig.subplots_adjust(left=0, right=1, bottom=0, top=1)
            ax.axis("tight")
            # Don't turn off axis - this might be causing the white image
        
        # Save to output directory
        output_file = results2_path / f"{file_path.stem}_marked.png"
        plt.savefig(output_file, dpi=100, bbox_inches='tight')
        plt.close("all")

    # Perform multithreaded spectrogram generation
    num_cores = multiprocessing.cpu_count()
    Parallel(n_jobs=num_cores)(
        delayed(__spectrogram_with_frequency_labels)(
            file_path=sample_file.path,
            png_resolution=png_resolution,
            fft_size=256 if sample_file.sample_rate >= 40 else 128,
            norm_vmin=norm_vmin,
            norm_vmax=norm_vmax,
        )
        for sample_file in sample_files
    )
