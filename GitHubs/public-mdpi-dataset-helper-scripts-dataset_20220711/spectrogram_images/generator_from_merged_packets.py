"""Spectrogram generator from merged packets with CSV labels

This script reads mixed signals from merged_packets directory and creates spectrograms
with YOLO format labels derived from CSV annotations.
"""

import multiprocessing
import warnings
from dataclasses import dataclass
from pathlib import Path
from typing import List, Dict, Any

import matplotlib
import matplotlib.pyplot as plt
import matplotlib.patches as patches
import numpy as np
import pandas as pd
from joblib import Parallel, delayed

# Switch to non-interactive backend for faster saving
matplotlib.use("Agg")


@dataclass
class Resolution:
    """Container for image resolution - number of pixels in x and y."""
    x: int
    y: int

    def __post_init__(self):
        if (self.x % 32 != 0) or (self.y % 32 != 0):
            warnings.warn(
                "Resolution must be a multiple of 32 when used for darknet yolo algorithm!"
            )


def read_mixed_signal(file_path: Path) -> np.ndarray:
    """Read mixed signal from merged_packets directory"""
    try:
        signal_data = np.fromfile(file_path, dtype=np.complex64)
        print(f"Loaded {len(signal_data)} samples from {file_path.name}")
        return signal_data
    except Exception as e:
        print(f"Error reading {file_path.name}: {e}")
        return np.array([])


def read_csv_labels(csv_file: Path) -> List[Dict[str, Any]]:
    """Read CSV labels from merged_packets"""
    try:
        df = pd.read_csv(csv_file)
        labels = []
        
        for _, row in df.iterrows():
            labels.append({
                'id': row['id'],
                'class': row['class'],
                'start_sample': row['sample_position_start'],
                'end_sample': row['sample_position_end'],
                'bandwidth': row['bandwidth'],
                'freq_offset': row['freq_offset'],
                'level': row['level'],
                'rf_std': row['rf_std'],
                'sample_rate': row['sample_rate']
            })
        
        print(f"Loaded {len(labels)} labels from {csv_file.name}")
        return labels
    except Exception as e:
        print(f"Error reading {csv_file.name}: {e}")
        return []


def csv_to_yolo_labels(csv_labels: List[Dict[str, Any]], signal_length: int, sample_rate: float) -> List[Dict[str, Any]]:
    """Convert CSV labels to YOLO format"""
    yolo_labels = []
    
    # Class mapping
    class_map = {
        'WLAN': 0,
        'BT_classic': 2,
        'BLE_1MHz': 3,
        'BLE_2MHz': 4,
        'collision': 1
    }
    
    for label in csv_labels:
        try:
            # Calculate time positions (normalized to 0-1)
            start_time = label['start_sample'] / signal_length
            end_time = label['end_sample'] / signal_length
            duration = end_time - start_time
            
            # Calculate frequency position (normalized to 0-1)
            # Center frequency is 0.5, frequency offset affects y position
            freq_offset = label['freq_offset']
            bandwidth = label['bandwidth']
            
            # Convert frequency offset to normalized position
            # Assuming signal spans from -sample_rate/2 to +sample_rate/2
            freq_center = 0.5 + (freq_offset / sample_rate)
            freq_height = bandwidth / sample_rate
            
            # Ensure coordinates are within bounds
            x_center = max(0, min(1, start_time + duration/2))
            y_center = max(0, min(1, freq_center))
            width = max(0.001, min(1, duration))  # Minimum width for visibility
            height = max(0.001, min(1, freq_height))  # Minimum height for visibility
            
            class_id = class_map.get(label['class'], 0)
            
            yolo_labels.append({
                'class_id': class_id,
                'x_center': x_center,
                'y_center': y_center,
                'width': width,
                'height': height,
                'original_class': label['class'],
                'level': label['level']
            })
            
        except Exception as e:
            print(f"Error converting label {label.get('id', 'unknown')}: {e}")
            continue
    
    return yolo_labels


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


def generate_spectrogram_with_labels(
    mixed_signal_file: Path,
    csv_labels_file: Path,
    output_dir: Path,
    sample_rate: float,
    resolution: Resolution,
    colormap: str = "viridis"
):
    """Generate spectrogram from mixed signal with labels"""
    
    # Read mixed signal
    signal_data = read_mixed_signal(mixed_signal_file)
    if len(signal_data) == 0:
        return None, None
    
    # Read CSV labels
    csv_labels = read_csv_labels(csv_labels_file)
    if not csv_labels:
        return None, None
    
    # Convert to YOLO format
    yolo_labels = csv_to_yolo_labels(csv_labels, len(signal_data), sample_rate)
    
    # Generate spectrogram
    fig, ax = plt.subplots(figsize=(resolution.x / 100, resolution.y / 100))
    
    # Create spectrogram
    Pxx, freqs, bins, im = ax.specgram(
        signal_data,
        NFFT=256,
        Fs=sample_rate,
        noverlap=0,
        mode="default",
        sides="default",
        vmin=-150,
        vmax=-50,
        window=np.hanning(256),
        cmap=colormap,
    )
    
    # Ensure spectrogram is visible
    ax.set_ylim(0, len(freqs)-1)
    ax.set_xlim(0, len(bins)-1)
    
    # Draw bounding boxes
    for label in yolo_labels:
        # Convert YOLO coordinates to spectrogram coordinates
        x_center = label['x_center'] * len(bins)
        y_center = label['y_center'] * len(freqs)
        width = label['width'] * len(bins)
        height = label['height'] * len(freqs)
        
        x_min = x_center - (width / 2)
        y_min = y_center - (height / 2)
        
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
    
    # Adjust layout
    fig.subplots_adjust(left=0, right=1, bottom=0, top=1)
    ax.axis("tight")
    
    # Save spectrogram
    output_file = output_dir / f"{mixed_signal_file.stem}_spectrogram.png"
    plt.savefig(output_file, dpi=100, bbox_inches='tight')
    plt.close()
    
    # Save YOLO labels
    label_file = output_dir / f"{mixed_signal_file.stem}.txt"
    with open(label_file, 'w') as f:
        for label in yolo_labels:
            f.write(f"{label['class_id']} {label['x_center']:.6f} {label['y_center']:.6f} {label['width']:.6f} {label['height']:.6f}\n")
    
    print(f"Generated: {output_file.name} with {len(yolo_labels)} labels")
    return output_file, label_file


def process_merged_packets(
    merged_packets_dir: Path,
    output_dir: Path,
    resolution: Resolution = Resolution(1024, 192),
    colormap: str = "viridis",
    max_samples: int = None
):
    """Process all mixed signals from merged_packets"""
    
    # Create output directory
    output_dir.mkdir(exist_ok=True)
    
    all_files = []
    
    # Process each bandwidth directory
    for bw_dir in merged_packets_dir.iterdir():
        if bw_dir.is_dir() and bw_dir.name.startswith('bw_'):
            # Extract sample rate from directory name
            try:
                sample_rate_str = bw_dir.name.split('_')[1].replace('e6', 'e6')
                sample_rate = float(sample_rate_str)
                print(f"Processing {bw_dir.name} with sample rate {sample_rate/1e6:.1f} MHz")
                
                # Find all signal files
                signal_files = [f for f in bw_dir.iterdir() if f.is_file() and not f.name.endswith('.csv')]
                
                for signal_file in signal_files:
                    # Find corresponding CSV file
                    csv_file = signal_file.with_suffix('.csv')
                    if csv_file.exists():
                        all_files.append((signal_file, csv_file, sample_rate))
                        
            except Exception as e:
                print(f"Error processing {bw_dir.name}: {e}")
                continue
    
    # Limit samples if specified
    if max_samples:
        all_files = all_files[:max_samples]
        print(f"Processing only {len(all_files)} samples for testing...")
    
    print(f"Found {len(all_files)} signal files to process")
    
    # Process files in parallel
    def process_single_file(file_info):
        signal_file, csv_file, sample_rate = file_info
        try:
            return generate_spectrogram_with_labels(
                signal_file, csv_file, output_dir, sample_rate, resolution, colormap
            )
        except Exception as e:
            print(f"Error processing {signal_file.name}: {e}")
            return None, None
    
    # Use multiprocessing
    num_cores = multiprocessing.cpu_count()
    results = Parallel(n_jobs=num_cores)(
        delayed(process_single_file)(file_info) for file_info in all_files
    )
    
    # Count successful results
    successful = sum(1 for result in results if result[0] is not None)
    print(f"Successfully processed {successful}/{len(all_files)} files")


if __name__ == "__main__":
    import argparse
    
    parser = argparse.ArgumentParser(
        description="Generate spectrograms from merged packets with CSV labels",
        formatter_class=argparse.ArgumentDefaultsHelpFormatter,
    )
    parser.add_argument(
        "-p",
        "--path",
        type=Path,
        required=True,
        help="path to merged_packets directory",
    )
    parser.add_argument(
        "-o",
        "--output-dir",
        type=Path,
        default="generated_spectrograms",
        help="output directory for generated spectrograms",
    )
    parser.add_argument(
        "-r",
        "--resolution",
        nargs=2,
        default=[1024, 192],
        type=int,
        metavar=("x", "y"),
        help="Target resolution of the generated spectrogram images",
    )
    parser.add_argument(
        "-c",
        "--colormap",
        type=str,
        default="viridis",
        help="matplotlib colormap used for spectrogram generation",
    )
    parser.add_argument(
        "--max-samples",
        type=int,
        default=None,
        help="Maximum number of samples to process (for testing)",
    )
    
    args = parser.parse_args()
    
    print("Generating spectrograms from merged packets...")
    print(f"Input directory: {args.path}")
    print(f"Output directory: {args.output_dir}")
    print(f"Resolution: {args.resolution[0]}x{args.resolution[1]}")
    print(f"Colormap: {args.colormap}")
    
    process_merged_packets(
        merged_packets_dir=args.path,
        output_dir=args.output_dir,
        resolution=Resolution(x=args.resolution[0], y=args.resolution[1]),
        colormap=args.colormap,
        max_samples=args.max_samples
    )
