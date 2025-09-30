#!/usr/bin/env python3
"""
Complete Pipeline Generator
==========================

This script implements the complete pipeline:
1. Load single packets from single_packet_samples/
2. Merge packets into frames based on config
3. Generate PNG spectrograms from merged frames

Based on config_training_data.toml and config_packet_capture.toml
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
    """Create spectrogram from signal data."""
    try:
        import matplotlib
        matplotlib.use('Agg')
        import matplotlib.pyplot as plt
        
        # Create figure
        fig, ax = plt.subplots(figsize=(10.24, 1.92))
        
        # Determine FFT size based on sample rate
        fft_size = 256 if sample_rate >= 40e6 else 128
        
        # Generate spectrogram
        ax.specgram(
            signal_data,
            NFFT=fft_size,
            Fs=sample_rate,
            noverlap=0,
            mode="default",
            sides="default",
            vmin=-150,
            vmax=-50,
            window=np.hanning(fft_size),
            cmap="viridis",
        )
        
        # Format and save
        fig.subplots_adjust(left=0, right=1, bottom=0, top=1)
        ax.axis("tight")
        ax.axis("off")
        plt.savefig(output_path, dpi=100, bbox_inches='tight')
        plt.close("all")
        
        return True
        
    except Exception as e:
        print(f"Error creating spectrogram: {e}")
        return False

def detect_collisions(labels: List[Dict]) -> List[Dict]:
    """Detect signal collisions and return collision bounding boxes."""
    collision_boxes = []
    n = len(labels)
    
    for i in range(n):
        for j in range(i + 1, n):
            # Check if signals overlap in time
            signal1_start = labels[i]['start_sample']
            signal1_end = labels[i]['end_sample']
            signal2_start = labels[j]['start_sample']
            signal2_end = labels[j]['end_sample']
            
            # Check for overlap
            if not (signal1_end < signal2_start or signal2_end < signal1_start):
                # Calculate collision area
                collision_start = max(signal1_start, signal2_start)
                collision_end = min(signal1_end, signal2_end)
                
                collision_box = {
                    'class': 'COLLISION',
                    'start_sample': collision_start,
                    'end_sample': collision_end
                }
                collision_boxes.append(collision_box)
                print(f"    Collision: {labels[i]['class']} vs {labels[j]['class']} at samples {collision_start}-{collision_end}")
    
    return collision_boxes

def create_yolo_labels(labels: List[Dict], frame_samples: int, output_path: Path) -> bool:
    """Create YOLO format labels."""
    try:
        # Detect collisions
        collision_boxes = detect_collisions(labels)
        
        with open(output_path, 'w') as f:
            # Write signal labels
            for label in labels:
                # Convert to YOLO format
                class_id = get_class_id(label['class'])
                
                # Calculate normalized coordinates
                start_norm = label['start_sample'] / frame_samples
                end_norm = label['end_sample'] / frame_samples
                width_norm = end_norm - start_norm
                center_norm = start_norm + width_norm / 2
                
                # YOLO format: class_id x_center y_center width height
                f.write(f"{class_id} {center_norm:.6f} 0.5 {width_norm:.6f} 0.5\n")
            
            # Write collision labels (class_id = 4)
            for collision in collision_boxes:
                class_id = 4  # Collision class
                
                # Calculate normalized coordinates
                start_norm = collision['start_sample'] / frame_samples
                end_norm = collision['end_sample'] / frame_samples
                width_norm = end_norm - start_norm
                center_norm = start_norm + width_norm / 2
                
                # YOLO format: class_id x_center y_center width height
                f.write(f"{class_id} {center_norm:.6f} 0.5 {width_norm:.6f} 0.5\n")
        
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

def main():
    """Main function to run the complete pipeline."""
    print("=" * 60)
    print("Complete Pipeline Generator")
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
                create_yolo_labels(labels, len(frame_data), label_file)
                
                # Report collision info
                collision_boxes = detect_collisions(labels)
                if collision_boxes:
                    print(f"    Found {len(collision_boxes)} collision areas")
            
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
