#!/usr/bin/env python3
"""
Complete Pipeline Generator - FIXED v2
==========================

This script implements the complete pipeline:
1. Load single packets from single_packet_samples/
2. Merge packets into frames based on config
3. Generate PNG spectrograms from merged frames

Fixed: Proper signal positioning and collision detection
"""

import os
import sys
import time
import random
import struct
from pathlib import Path
from typing import List, Dict, Tuple
import math
import numpy as np
import matplotlib.pyplot as plt

def load_single_packets(single_packet_dir: Path):
    """Load individual packet samples from single_packet_samples directory."""
    print("Loading single packet samples...")
    
    packets = []
    
    for protocol_dir in single_packet_dir.iterdir():
        if not protocol_dir.is_dir():
            continue
            
        protocol = protocol_dir.name
        print(f"  Loading {protocol} packets...")
        
        for packet_file in protocol_dir.glob("*.packet"):
            try:
                # Load complex64 data
                data = np.fromfile(packet_file, dtype=np.complex64)
                
                # Parse filename for metadata
                metadata = parse_packet_filename(packet_file.name)
                
                packet = {
                    'path': packet_file,
                    'protocol': protocol,
                    'data': data,
                    'sample_rate': metadata.get('sample_rate', 125e6),
                    'payload_size': metadata.get('payload', 0),
                    'duration': len(data) / metadata.get('sample_rate', 125e6)
                }
                
                packets.append(packet)
                
            except Exception as e:
                print(f"    Warning: Could not load {packet_file}: {e}")
    
    print(f"  Loaded {len(packets)} packets total")
    return packets

def parse_packet_filename(filename: str) -> Dict:
    """Parse packet filename to extract metadata."""
    metadata = {}
    
    # Extract sample rate
    if 'sampRate_' in filename:
        try:
            rate_str = filename.split('sampRate_')[1].split('_')[0]
            metadata['sample_rate'] = float(rate_str)
        except:
            metadata['sample_rate'] = 125e6
    
    # Extract payload size
    if 'payload_' in filename:
        try:
            payload_str = filename.split('payload_')[1].split('.')[0]
            metadata['payload'] = int(payload_str)
        except:
            metadata['payload'] = 0
    
    return metadata

def merge_packets_into_frame(packets: List[Dict], target_bandwidth: float, 
                           frame_duration: float = 45e-4) -> Tuple[np.ndarray, List[Dict]]:
    """Merge multiple packets into a single frame."""
    
    # Calculate frame length in samples
    frame_samples = int(frame_duration * target_bandwidth)
    frame_data = np.zeros(frame_samples, dtype=np.complex64)
    
    # Generate labels for this frame
    labels = []
    
    # Select random packets for this frame (based on config ratios)
    num_packets = random.randint(18, 25)  # From config: min=18, max=25
    selected_packets = random.sample(packets, min(num_packets, len(packets)))
    
    for packet in selected_packets:
        # Random timing within frame
        start_time = random.uniform(0, frame_duration - packet['duration'])
        start_sample = int(start_time * target_bandwidth)
        
        # Check for collisions
        if check_collision(frame_data, start_sample, len(packet['data'])):
            continue
        
        # Add packet to frame
        end_sample = min(start_sample + len(packet['data']), frame_samples)
        packet_length = end_sample - start_sample
        
        if packet_length > 0:
            # Apply channel effects
            processed_packet = apply_channel_effects(
                packet['data'][:packet_length], packet['protocol']
            )
            
            frame_data[start_sample:end_sample] += processed_packet
            
            # Create label
            label = {
                'class': packet['protocol'],
                'start_sample': start_sample,
                'end_sample': end_sample,
                'payload_size': packet['payload_size']
            }
            labels.append(label)
    
    # Add noise (from config: 0.0055 to 0.0065)
    noise_amplitude = random.uniform(0.0055, 0.0065)
    noise = (np.random.randn(frame_samples) + 1j * np.random.randn(frame_samples)) * noise_amplitude
    frame_data += noise
    
    return frame_data, labels

def check_collision(frame_data: np.ndarray, start: int, length: int) -> bool:
    """Check if adding packet would cause collision."""
    end = min(start + length, len(frame_data))
    return np.any(np.abs(frame_data[start:end]) > 0.1)

def apply_channel_effects(packet_data: np.ndarray, protocol: str) -> np.ndarray:
    """Apply channel effects to packet data."""
    # Simple channel model
    phase_shift = random.uniform(0, 2 * math.pi)
    packet_data *= np.exp(1j * phase_shift)
    
    amplitude = random.uniform(0.5, 1.5)
    packet_data *= amplitude
    
    return packet_data

def create_spectrogram(signal_data: np.ndarray, sample_rate: float, 
                      output_path: Path) -> bool:
    """Create spectrogram from signal data using the same approach as generator.py."""
    try:
        import matplotlib
        matplotlib.use('Agg')
        import matplotlib.pyplot as plt
        
        # Use the same approach as generator.py
        # Determine FFT size based on sample rate (same logic as generator.py)
        fft_size = 256 if sample_rate >= 40e6 else 128
        
        # Create figure with proper resolution (like generator.py)
        png_resolution_x = 1024
        png_resolution_y = 192
        fig, ax = plt.subplots(figsize=(png_resolution_x / 100, png_resolution_y / 100))
        
        # Generate spectrogram with same parameters as generator.py
        ax.specgram(
            signal_data,
            NFFT=fft_size,
            Fs=42e6,  # Dummy sample rate (same as generator.py)
            noverlap=0,  # Same as generator.py
            mode="default",
            sides="default",
            vmin=-150,  # Same normalization as generator.py
            vmax=-50,
            window=np.hanning(fft_size),
            cmap="viridis",
        )
        
        # Format and save exactly like generator.py
        fig.subplots_adjust(left=0, right=1, bottom=0, top=1)
        ax.axis("tight")
        ax.axis("off")
        plt.savefig(output_path)  # Same as generator.py
        plt.close("all")
        
        return True
        
    except Exception as e:
        print(f"Error creating spectrogram: {e}")
        return False

def detect_collisions(labels: List[Dict]) -> List[Dict]:
    """Detect signal collisions considering BOTH time AND frequency overlap."""
    collision_boxes = []
    n = len(labels)
    
    for i in range(n):
        for j in range(i + 1, n):
            # Check if signals overlap in time
            signal1_start = labels[i]['start_sample']
            signal1_end = labels[i]['end_sample']
            signal2_start = labels[j]['start_sample']
            signal2_end = labels[j]['end_sample']
            
            # Check for time overlap
            if not (signal1_end < signal2_start or signal2_end < signal1_start):
                # Calculate time collision area
                collision_start = max(signal1_start, signal2_start)
                collision_end = min(signal1_end, signal2_end)
                
                # Get frequency bounds for both signals
                signal1_y_min = labels[i].get('y_min', 0.3)
                signal1_y_max = labels[i].get('y_max', 0.7)
                signal2_y_min = labels[j].get('y_min', 0.3)
                signal2_y_max = labels[j].get('y_max', 0.7)
                
                # Check for frequency overlap
                if not (signal1_y_max < signal2_y_min or signal2_y_max < signal1_y_min):
                    # Calculate frequency overlap
                    collision_y_min = max(signal1_y_min, signal2_y_min)
                    collision_y_max = min(signal1_y_max, signal2_y_max)
                    collision_y_center = (collision_y_min + collision_y_max) / 2
                    collision_y_height = collision_y_max - collision_y_min
                    
                    collision_box = {
                        'class': 'COLLISION',
                        'start_sample': collision_start,
                        'end_sample': collision_end,
                        'y_center': collision_y_center,
                        'y_height': collision_y_height,
                        'y_min': collision_y_min,
                        'y_max': collision_y_max
                    }
                    collision_boxes.append(collision_box)
                    print(f"    Collision: {labels[i]['class']} vs {labels[j]['class']}")
                    print(f"      Time: samples {collision_start}-{collision_end}")
                    print(f"      Freq: y={collision_y_center:.3f}, height={collision_y_height:.3f}")
    
    return collision_boxes

def analyze_signal_characteristics(signal_data: np.ndarray, start_sample: int, end_sample: int, sample_rate: float) -> Dict:
    """Analyze signal characteristics for a specific segment - MORE ACCURATE VERSION."""
    try:
        # Extract signal segment
        segment = signal_data[start_sample:end_sample]
        
        if len(segment) == 0:
            return {'amplitude': 0, 'max_freq': 0, 'energy': 0, 'freq_center': 0.5, 'freq_bandwidth': 0.3, 'y_min': 0.35, 'y_max': 0.65}
        
        # Calculate amplitude characteristics
        amplitude = np.max(np.abs(segment))
        rms_amplitude = np.sqrt(np.mean(np.abs(segment)**2))
        
        # Calculate frequency characteristics using FFT
        fft_data = np.fft.fftshift(np.fft.fft(segment))
        freqs = np.fft.fftshift(np.fft.fftfreq(len(segment), 1/sample_rate))
        power_spectrum = np.abs(fft_data)**2
        
        # Find peak frequency
        max_freq_idx = np.argmax(power_spectrum)
        max_freq = freqs[max_freq_idx]
        
        # Calculate frequency bandwidth using 20% threshold for tighter bounds
        peak_power = np.max(power_spectrum)
        threshold = peak_power * 0.2  # Increased threshold for tighter fit
        
        # Find frequency range
        above_threshold = power_spectrum > threshold
        if np.any(above_threshold):
            freq_indices = np.where(above_threshold)[0]
            min_freq = freqs[freq_indices[0]]
            max_freq_range = freqs[freq_indices[-1]]
            freq_bandwidth = abs(max_freq_range - min_freq)
            freq_center = (min_freq + max_freq_range) / 2
            
            # Normalize to image coordinates (0 = bottom, 1 = top)
            freq_center_norm = (freq_center + sample_rate/2) / sample_rate
            freq_bandwidth_norm = freq_bandwidth / sample_rate
            
            # Calculate y_min and y_max for collision detection
            y_min = (min_freq + sample_rate/2) / sample_rate
            y_max = (max_freq_range + sample_rate/2) / sample_rate
        else:
            freq_center = max_freq
            freq_bandwidth = 0.15 * sample_rate
            
            freq_center_norm = (freq_center + sample_rate/2) / sample_rate
            freq_bandwidth_norm = freq_bandwidth / sample_rate
            
            # Calculate bounds
            y_min = freq_center_norm - freq_bandwidth_norm / 2
            y_max = freq_center_norm + freq_bandwidth_norm / 2
        
        # Clamp to valid range
        freq_center_norm = max(0.0, min(1.0, freq_center_norm))
        freq_bandwidth_norm = max(0.05, min(0.8, freq_bandwidth_norm))  # Reduced min from 0.1 to 0.05
        y_min = max(0.0, min(1.0, y_min))
        y_max = max(0.0, min(1.0, y_max))
        
        # Calculate energy
        energy = np.sum(power_spectrum)
        
        print(f"      Freq: {freq_center/1e6:.1f} MHz, BW: {freq_bandwidth/1e6:.1f} MHz")
        print(f"      Norm: y_center={freq_center_norm:.3f}, height={freq_bandwidth_norm:.3f}")
        print(f"      Bounds: y_min={y_min:.3f}, y_max={y_max:.3f}")
        
        return {
            'amplitude': amplitude,
            'rms_amplitude': rms_amplitude,
            'max_freq': abs(max_freq),
            'energy': energy,
            'freq_center': freq_center_norm,
            'freq_bandwidth': freq_bandwidth_norm,
            'y_min': y_min,
            'y_max': y_max
        }
    except Exception as e:
        print(f"Error analyzing signal: {e}")
        return {'amplitude': 0, 'max_freq': 0, 'energy': 0, 'freq_center': 0.5, 'freq_bandwidth': 0.3, 'y_min': 0.35, 'y_max': 0.65}

def create_yolo_labels(labels: List[Dict], frame_samples: int, output_path: Path, signal_data: np.ndarray = None, sample_rate: float = None) -> bool:
    """Create YOLO format labels - IMPROVED VERSION."""
    try:
        # First, analyze all signals and store their frequency bounds
        # Make a copy to avoid modifying original labels during collision detection
        labels_with_bounds = []
        for label in labels:
            # Create a copy of the label
            label_copy = dict(label)
            
            if signal_data is not None and sample_rate is not None:
                characteristics = analyze_signal_characteristics(
                    signal_data, label['start_sample'], label['end_sample'], sample_rate
                )
                label_copy['y_center'] = characteristics['freq_center']
                label_copy['y_height'] = characteristics['freq_bandwidth']
                label_copy['y_min'] = characteristics['y_min']
                label_copy['y_max'] = characteristics['y_max']
            else:
                label_copy['y_center'] = 0.5
                label_copy['y_height'] = 0.3
                label_copy['y_min'] = 0.35
                label_copy['y_max'] = 0.65
            
            labels_with_bounds.append(label_copy)
        
        # Now detect collisions with frequency information
        collision_boxes = detect_collisions(labels_with_bounds)
        
        with open(output_path, 'w') as f:
            # Write signal labels (use labels_with_bounds for frequency info)
            for i, label in enumerate(labels):
                class_id = get_class_id(label['class'])
                
                # Calculate normalized coordinates (x-axis = time)
                start_norm = label['start_sample'] / frame_samples
                end_norm = label['end_sample'] / frame_samples
                width_norm = end_norm - start_norm
                center_norm = start_norm + width_norm / 2
                
                # Get frequency info from the bounds version
                y_center = labels_with_bounds[i]['y_center']
                height_norm = labels_with_bounds[i]['y_height']
                
                # Ensure reasonable bounds
                y_center = max(0.0, min(1.0, y_center))
                height_norm = max(0.05, min(0.9, height_norm))
                
                print(f"    Label {label['class']}: x={center_norm:.3f}, y={y_center:.3f}, w={width_norm:.3f}, h={height_norm:.3f}")
                
                # YOLO format: class_id x_center y_center width height
                f.write(f"{class_id} {center_norm:.6f} {y_center:.6f} {width_norm:.6f} {height_norm:.6f}\n")
            
            # Write collision labels (class_id = 4)
            for collision in collision_boxes:
                class_id = 4  # Collision class
                
                # Calculate normalized coordinates
                start_norm = collision['start_sample'] / frame_samples
                end_norm = collision['end_sample'] / frame_samples
                width_norm = end_norm - start_norm
                center_norm = start_norm + width_norm / 2
                
                # Use the calculated collision bounds
                y_center = collision['y_center']
                height_norm = collision['y_height']
                
                # Ensure reasonable bounds
                y_center = max(0.0, min(1.0, y_center))
                height_norm = max(0.05, min(0.9, height_norm))
                
                print(f"    Collision label: x={center_norm:.3f}, y={y_center:.3f}, w={width_norm:.3f}, h={height_norm:.3f}")
                
                # YOLO format: class_id x_center y_center width height
                f.write(f"{class_id} {center_norm:.6f} {y_center:.6f} {width_norm:.6f} {height_norm:.6f}\n")
        
        return True
        
    except Exception as e:
        print(f"Error creating YOLO labels: {e}")
        return False

def get_class_id(protocol: str) -> int:
    """Get class ID for protocol."""
    class_mapping = {
        'WLAN': 0,
        'BT_classic': 1,
        'BLE_1MHz': 2,
        'BLE_2MHz': 3,
        'wlan': 0
    }
    return class_mapping.get(protocol, 0)

def get_class_name(class_id: int) -> str:
    """Get class name from class ID."""
    class_names = {
        0: 'WLAN',
        1: 'BT',
        2: 'BLE',
        3: 'BLE2',
        4: 'COLLISION'
    }
    return class_names.get(class_id, 'UNKNOWN')

def create_marked_image(spectrogram_path: Path, label_path: Path, output_path: Path) -> bool:
    """Create marked image with bounding boxes drawn on spectrogram."""
    try:
        import matplotlib
        matplotlib.use('Agg')
        import matplotlib.pyplot as plt
        from matplotlib.patches import Rectangle
        from PIL import Image
        import numpy as np
        
        # Load the spectrogram image
        img = Image.open(spectrogram_path)
        img_array = np.array(img)
        
        # Create figure with same resolution as generator.py
        png_resolution_x = 1024
        png_resolution_y = 192
        fig, ax = plt.subplots(figsize=(png_resolution_x / 100, png_resolution_y / 100))
        ax.imshow(img_array, aspect='auto')
        
        # Define colors for different classes
        colors = ['red', 'blue', 'green', 'yellow', 'orange']
        class_names = ['WLAN', 'BT', 'BLE', 'BLE2', 'COLLISION']
        
        # Read YOLO labels
        labels = []
        if label_path.exists():
            with open(label_path, 'r') as f:
                for line in f:
                    line = line.strip()
                    if line and not line.startswith('#'):
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
        
        # Get image dimensions
        img_height, img_width = img_array.shape[:2]
        
        # Draw bounding boxes
        for label in labels:
            class_id = label['class_id']
            class_name = get_class_name(class_id)
            
            # Convert normalized coordinates to pixel coordinates
            x_center_pixel = label['x_center'] * img_width
            y_center_pixel = label['y_center'] * img_height
            width_pixel = label['width'] * img_width
            height_pixel = label['height'] * img_height
            
            # Calculate box corners
            x1 = x_center_pixel - width_pixel / 2
            y1 = y_center_pixel - height_pixel / 2
            
            # Choose color based on class
            color = colors[class_id % len(colors)]
            
            # Create rectangle
            rect = Rectangle(
                (x1, y1), width_pixel, height_pixel,
                linewidth=2,
                edgecolor=color,
                facecolor='none',
                alpha=0.8
            )
            ax.add_patch(rect)
            
            # Add label text
            ax.text(
                x1, y1 - 5,
                f"{class_name}",
                color=color,
                fontsize=8,
                fontweight='bold',
                bbox=dict(boxstyle="round,pad=0.3", facecolor='white', alpha=0.8)
            )
        
        # Format and save
        ax.set_title(f"Signal Detection: {len(labels)} objects found", fontsize=10, pad=10)
        ax.axis('off')
        plt.tight_layout()
        plt.savefig(output_path, dpi=100, bbox_inches='tight')
        plt.close("all")
        
        return True
        
    except Exception as e:
        print(f"Error creating marked image: {e}")
        return False

def main():
    """Main function to run the complete pipeline."""
    print("=" * 60)
    print("Complete Pipeline Generator - FIXED v2")
    print("=" * 60)
    
    # Define paths based on config
    base_dir = Path("../spectrogram_training_data_20220711")
    single_packet_dir = base_dir / "single_packet_samples"
    output_dir = Path("generated_output")
    
    # Create output directory
    output_dir.mkdir(exist_ok=True)
    (output_dir / "results").mkdir(exist_ok=True)
    
    # Load single packets
    packets = load_single_packets(single_packet_dir)
    if not packets:
        print("No packets loaded")
        return
    
    # Generate frames based on config
    sample_rates = [25e6, 45e6, 60e6, 125e6]  # From config
    frames_per_rate = 10  # Reduced for testing
    
    total_frames = 0
    
    for sample_rate in sample_rates:
        print(f"\nProcessing {sample_rate/1e6:.0f}MHz bandwidth...")
        
        for frame_idx in range(frames_per_rate):
            # Generate frame
            frame_data, labels = merge_packets_into_frame(packets, sample_rate)
            
            # Create filename
            timestamp = int(time.time() * 1e15) + frame_idx
            frame_id = f"result_frame_{timestamp}_bw_{sample_rate/1e6:.0f}E+6"
            
            # Save raw signal data
            signal_file = output_dir / "results" / frame_id
            frame_data.astype(np.complex64).tofile(signal_file)
            
            # Create spectrogram
            spectrogram_file = output_dir / "results" / f"{frame_id}.png"
            success = create_spectrogram(frame_data, sample_rate, spectrogram_file)
            
            # Create labels
            if success and labels:
                label_file = output_dir / "results" / f"{frame_id}.txt"
                create_yolo_labels(labels, len(frame_data), label_file, frame_data, sample_rate)
                
                # Create marked image with bounding boxes
                marked_file = output_dir / "results" / f"{frame_id}_marked.png"
                marked_success = create_marked_image(spectrogram_file, label_file, marked_file)
                if marked_success:
                    print(f"    Created marked image: {marked_file.name}")
            
            if success:
                total_frames += 1
                print(f"  Created frame {frame_idx + 1}/{frames_per_rate}")
    
    print(f"\nCompleted: {total_frames} frames generated")
    print(f"Output directory: {output_dir.absolute()}")

if __name__ == "__main__":
    # Import numpy here to avoid issues if not available
    try:
        import numpy as np
        main()
    except ImportError:
        print("NumPy not available. Please install numpy and matplotlib.")
        print("Commands to install:")
        print("  pip install numpy matplotlib")
