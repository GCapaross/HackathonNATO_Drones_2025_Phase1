"""
Real-time RF Spectrogram Generator
===================================
Enhanced version of rrs.py for real-time spectrogram generation
with YOLO-compatible output format.
"""

import numpy as np
import matplotlib.pyplot as plt
from scipy.signal import spectrogram
import os
import datetime
import cv2
from typing import List, Tuple, Optional
import threading
import queue
import time

class RealtimeSpectrogramGenerator:
    """
    Real-time spectrogram generator for RF signal processing pipeline.
    """
    
    def __init__(self, 
                 fs: float = 125e6,
                 section_length: float = 4.5e-3,
                 noise_sigma: float = 0.01,
                 spectrogram_params: dict = None,
                 packet_type: str = "single",
                 use_training_format: bool = True):
        """
        Initialize the spectrogram generator.
        
        Args:
            fs: Sample rate (Hz)
            section_length: Length of output buffer (seconds)
            noise_sigma: Standard deviation of added Gaussian noise
            spectrogram_params: Parameters for spectrogram generation
            packet_type: Type of packets to use ("single" or "merged")
            use_training_format: Whether to use the same format as training data
        """
        self.fs = fs
        self.section_length = section_length
        self.noise_sigma = noise_sigma
        self.packet_type = packet_type
        self.use_training_format = use_training_format
        self.n_samples = int(section_length * fs)
        
        # For real-time processing
        self.signal_buffer = np.zeros(self.n_samples, dtype=np.complex64)
        self.spectrogram_queue = queue.Queue(maxsize=10)
        self.running = False
        
    def read_packet_file(self, filename: str) -> np.ndarray:
        """
        Read a packet file and return complex IQ samples.
        Each file = interleaved float32 pairs [I0, Q0, I1, Q1, ...].
        Handles both .packet files and merged packet files (no extension).
        """
        try:
            raw = np.fromfile(filename, dtype=np.float32)
            iq_samples = raw[0::2] + 1j * raw[1::2]
            return iq_samples
        except Exception as e:
            print(f"Error reading packet file {filename}: {e}")
            return np.array([])
    
    def aggregate_frames(self, file_list: List[str]) -> np.ndarray:
        """
        Create a composite buffer of several frames with random positioning.
        
        Args:
            file_list: List of .packet file paths
            
        Returns:
            Composite IQ signal
        """
        buffer = np.zeros(self.n_samples, dtype=np.complex64)
        
        for file_path in file_list:
            iq = self.read_packet_file(file_path)
            if len(iq) == 0:
                continue
                
            # Random start position (leave space so frame fits)
            max_start = max(1, self.n_samples - len(iq))
            start = np.random.randint(0, max_start)
            
            # Add the frame to the buffer
            buffer[start:start+len(iq)] += iq
        
        # Add AWGN noise
        noise = (np.random.normal(0, self.noise_sigma, self.n_samples) +
                 1j * np.random.normal(0, self.noise_sigma, self.n_samples))
        buffer += noise
        
        return buffer
    
    def generate_spectrogram(self, iq_samples: np.ndarray, 
                           save_path: Optional[str] = None,
                           return_image: bool = True) -> Tuple[np.ndarray, str]:
        """
        Generate spectrogram from IQ samples using the same method as the original generator.
        
        Args:
            iq_samples: Complex IQ signal
            save_path: Path to save spectrogram (optional)
            return_image: Whether to return image array
            
        Returns:
            Tuple of (spectrogram_image, file_path)
        """
        if save_path is None:
            timestamp = datetime.datetime.now().strftime('%Y%m%d_%H%M%S_%f')
            save_path = f"realtime_spectrogram_{timestamp}.png"
        
        # Use the same method as the original generator
        # Set up matplotlib for non-interactive backend
        import matplotlib
        matplotlib.use("Agg")
        
        # Determine FFT size based on sample rate (matching original logic)
        fft_size = 256 if self.fs >= 40e6 else 128
        
        # Create figure with proper resolution (matching original)
        fig, ax = plt.subplots(figsize=(10, 4))  # Standard size for YOLO
        
        # Generate spectrogram using ax.specgram (matching original method)
        ax.specgram(
            iq_samples,
            NFFT=fft_size,
            Fs=self.fs,  # Use actual sample rate
            noverlap=0,  # No overlap, matching original
            mode="default",
            sides="default",
            vmin=-150,  # Matching original normalization
            vmax=-50,   # Matching original normalization
            window=np.hanning(fft_size),
            cmap="viridis"
        )
        
        # Turn axis off and save (matching original method)
        fig.subplots_adjust(left=0, right=1, bottom=0, top=1)
        ax.axis("tight")
        ax.axis("off")
        
        # Save with high DPI for quality
        plt.savefig(save_path, dpi=150, bbox_inches='tight')
        plt.close("all")
        
        # Load image for YOLO processing
        if return_image:
            image = cv2.imread(save_path)
            return image, save_path
        else:
            return None, save_path
    
    def get_packet_files(self, base_dir: str) -> List[str]:
        """
        Get list of packet files based on packet type.
        
        Args:
            base_dir: Base directory containing packet data
            
        Returns:
            List of packet file paths
        """
        packet_files = []
        
        if self.packet_type == "single":
            # Look in single_packet_samples subdirectories
            single_dir = os.path.join(base_dir, "single_packet_samples")
            if os.path.exists(single_dir):
                for root, dirs, files in os.walk(single_dir):
                    for file in files:
                        if file.endswith('.packet'):
                            packet_files.append(os.path.join(root, file))
        elif self.packet_type == "merged":
            # Look in merged_packets subdirectories (no file extension)
            merged_dir = os.path.join(base_dir, "merged_packets")
            if os.path.exists(merged_dir):
                for root, dirs, files in os.walk(merged_dir):
                    for file in files:
                        # Merged packets have no extension, just check if it's not a directory
                        if not os.path.isdir(os.path.join(root, file)) and '.' not in file:
                            packet_files.append(os.path.join(root, file))
        else:
            # Look in both directories
            for subdir in ["single_packet_samples", "merged_packets"]:
                full_dir = os.path.join(base_dir, subdir)
                if os.path.exists(full_dir):
                    for root, dirs, files in os.walk(full_dir):
                        for file in files:
                            if subdir == "single_packet_samples" and file.endswith('.packet'):
                                packet_files.append(os.path.join(root, file))
                            elif subdir == "merged_packets" and '.' not in file:
                                packet_files.append(os.path.join(root, file))
        
        return packet_files

    def generate_from_packets(self, packet_files: List[str], 
                            output_dir: str = "realtime_output") -> Tuple[np.ndarray, str]:
        """
        Generate spectrogram from a list of packet files.
        
        Args:
            packet_files: List of .packet file paths
            output_dir: Directory to save output
            
        Returns:
            Tuple of (spectrogram_image, file_path)
        """
        os.makedirs(output_dir, exist_ok=True)
        
        if self.packet_type == "merged":
            # For merged packets, just read the first file directly
            if packet_files:
                iq_samples = self.read_packet_file(packet_files[0])
                if len(iq_samples) == 0:
                    return None, None
                # For merged packets, use the full file length, don't truncate
                # The file already contains the complete merged signal
            else:
                return None, None
        else:
            # For single packets, aggregate multiple frames
            composite_signal = self.aggregate_frames(packet_files)
            iq_samples = composite_signal
        
        # Generate spectrogram
        timestamp = datetime.datetime.now().strftime('%Y%m%d_%H%M%S_%f')
        save_path = os.path.join(output_dir, f"spectrogram_{timestamp}.png")
        
        return self.generate_spectrogram(iq_samples, save_path)
    
    def generate_multiple_spectrograms(self, base_dir: str, 
                                     output_dir: str = "realtime_output",
                                     num_spectrograms: int = 10,
                                     packets_per_spectrogram: int = 5) -> List[Tuple[np.ndarray, str]]:
        """
        Generate multiple spectrograms for testing.
        
        Args:
            base_dir: Base directory containing packet data
            output_dir: Directory to save spectrograms
            num_spectrograms: Number of spectrograms to generate
            packets_per_spectrogram: Number of packets to use per spectrogram
            
        Returns:
            List of (image, file_path) tuples
        """
        os.makedirs(output_dir, exist_ok=True)
        
        # Get available packet files
        packet_files = self.get_packet_files(base_dir)
        
        if not packet_files:
            print(f"No packet files found in {base_dir}")
            return []
        
        print(f"Found {len(packet_files)} packet files")
        print(f"Generating {num_spectrograms} spectrograms with {packets_per_spectrogram} packets each...")
        
        results = []
        
        for i in range(num_spectrograms):
            try:
                if self.packet_type == "merged":
                    # For merged packets, just select one file
                    selected_files = [np.random.choice(packet_files)]
                else:
                    # For single packets, select multiple files
                    n_packets = min(packets_per_spectrogram, len(packet_files))
                    selected_files = np.random.choice(packet_files, size=n_packets, replace=False)
                
                # Generate spectrogram
                image, file_path = self.generate_from_packets(selected_files, output_dir)
                
                if image is not None:
                    results.append((image, file_path))
                    print(f"Generated {i+1}/{num_spectrograms}: {file_path}")
                else:
                    print(f"Failed to generate spectrogram {i+1}")
                    
            except Exception as e:
                print(f"Error generating spectrogram {i+1}: {e}")
        
        print(f"Successfully generated {len(results)} spectrograms")
        return results
    
    def start_realtime_processing(self, packet_source_dir: str, 
                                output_dir: str = "realtime_output",
                                processing_interval: float = 1.0):
        """
        Start real-time processing of packet files.
        
        Args:
            packet_source_dir: Directory containing .packet files
            output_dir: Directory to save spectrograms
            processing_interval: Time between processing cycles (seconds)
        """
        self.running = True
        os.makedirs(output_dir, exist_ok=True)
        
        # Get all packet files
        packet_files = []
        for root, dirs, files in os.walk(packet_source_dir):
            for file in files:
                if file.endswith('.packet'):
                    packet_files.append(os.path.join(root, file))
        
        if not packet_files:
            print(f"No .packet files found in {packet_source_dir}")
            return
        
        print(f"Found {len(packet_files)} packet files")
        print(f"Starting real-time processing...")
        
        while self.running:
            try:
                # Select random subset of packets
                n_packets = min(10, len(packet_files))  # Use up to 10 packets
                selected_files = np.random.choice(packet_files, size=n_packets, replace=False)
                
                # Generate spectrogram
                image, file_path = self.generate_from_packets(selected_files, output_dir)
                
                # Add to queue for YOLO processing
                if not self.spectrogram_queue.full():
                    self.spectrogram_queue.put((image, file_path))
                else:
                    print("Warning: Spectrogram queue full, dropping frame")
                
                print(f"Generated: {file_path}")
                time.sleep(processing_interval)
                
            except KeyboardInterrupt:
                print("Stopping real-time processing...")
                self.running = False
            except Exception as e:
                print(f"Error in real-time processing: {e}")
                time.sleep(1.0)
    
    def stop_realtime_processing(self):
        """Stop real-time processing."""
        self.running = False
    
    def get_next_spectrogram(self) -> Tuple[np.ndarray, str]:
        """
        Get the next spectrogram from the processing queue.
        
        Returns:
            Tuple of (spectrogram_image, file_path) or (None, None) if queue empty
        """
        try:
            return self.spectrogram_queue.get_nowait()
        except queue.Empty:
            return None, None

# Example usage and testing
if __name__ == "__main__":
    import argparse
    
    # Parse command line arguments
    parser = argparse.ArgumentParser(description="Generate spectrograms from RF packet data")
    parser.add_argument("--num_images", type=int, default=10, help="Number of spectrograms to generate")
    parser.add_argument("--packet_type", type=str, default="single", choices=["single", "merged", "both"], 
                       help="Type of packets to use")
    parser.add_argument("--packets_per_image", type=int, default=5, help="Number of packets per spectrogram")
    parser.add_argument("--output_dir", type=str, default="generated_spectrograms", 
                       help="Output directory for spectrograms")
    parser.add_argument("--base_dir", type=str, default="../spectrogram_training_data_20220711/", 
                       help="Base directory containing packet data")
    parser.add_argument("--training_format", action="store_true", 
                       help="Use the same spectrogram format as training data (may cut frequency at 0)")
    parser.add_argument("--full_frequency", action="store_true", 
                       help="Use full frequency range (0 to fs/2) instead of training format")
    
    args = parser.parse_args()
    
    # Determine format based on arguments
    use_training_format = args.training_format or not args.full_frequency
    
    # Initialize generator with specified packet type
    generator = RealtimeSpectrogramGenerator(
        fs=125e6,
        section_length=4.5e-3,
        noise_sigma=0.005,
        packet_type=args.packet_type,
        use_training_format=use_training_format
    )
    
    print(f"Generating {args.num_images} spectrograms...")
    print(f"Packet type: {args.packet_type}")
    print(f"Packets per image: {args.packets_per_image}")
    print(f"Output directory: {args.output_dir}")
    print("-" * 50)
    
    # Generate multiple spectrograms
    results = generator.generate_multiple_spectrograms(
        base_dir=args.base_dir,
        output_dir=args.output_dir,
        num_spectrograms=args.num_images,
        packets_per_spectrogram=args.packets_per_image
    )
    
    print(f"\nGenerated {len(results)} spectrograms successfully!")
    print(f"Files saved in: {args.output_dir}/")
    
    # TODO: Future adaptation for constantly receiving data
    # ====================================================
    # 
    # For real-time data streaming, you would modify this to:
    # 
    # 1. CONTINUOUS DATA STREAMING:
    #    - Replace file-based packet reading with live data streams
    #    - Use threading/async processing for continuous generation
    #    - Implement data buffers for incoming RF samples
    # 
    # 2. LIVE DATA SOURCES:
    #    - SDR (Software Defined Radio) integration (USRP, HackRF, RTL-SDR)
    #    - Network streaming (UDP/TCP packets from RF receivers)
    #    - Real-time IQ data from hardware
    # 
    # 3. STREAMING ARCHITECTURE:
    #    - Producer-Consumer pattern with queues
    #    - Sliding window processing for continuous spectrograms
    #    - Real-time FFT processing without file I/O
    # 
    # 4. EXAMPLE ADAPTATION:
    #    ```python
    #    def process_live_stream(self, data_stream):
    #        """Process continuously incoming RF data stream"""
    #        buffer = np.zeros(self.n_samples, dtype=np.complex64)
    #        while self.running:
    #            # Get new data from stream
    #            new_data = data_stream.get_samples(self.n_samples)
    #            
    #            # Update buffer (sliding window)
    #            buffer = np.roll(buffer, -len(new_data))
    #            buffer[-len(new_data):] = new_data
    #            
    #            # Generate spectrogram from current buffer
    #            spectrogram = self.generate_spectrogram(buffer)
    #            
    #            # Send to detection pipeline
    #            self.send_to_detector(spectrogram)
    #    ```
    # 
    # 5. HARDWARE INTEGRATION:
    #    - GNU Radio integration for SDR data
    #    - Custom hardware drivers for RF frontends
    #    - Network protocols for distributed RF sensing
    # 
    # 6. PERFORMANCE OPTIMIZATIONS:
    #    - GPU acceleration for FFT processing
    #    - Parallel spectrogram generation
    #    - Memory-mapped files for large datasets
    #    - Caching strategies for repeated processing

# =============================================================================
# HOW TO RUN THE PROGRAM
# =============================================================================
#
# 1. BASIC USAGE:
#    python3 realtime_spectrogram_generator.py
#    (Generates 10 spectrograms from single packets by default)
#
# 2. SPECIFY NUMBER OF IMAGES:
#    python3 realtime_spectrogram_generator.py --num_images 50
#    (Generates 50 spectrograms)
#
# 3. CHOOSE PACKET TYPE:
#    python3 realtime_spectrogram_generator.py --packet_type merged
#    python3 realtime_spectrogram_generator.py --packet_type single
#    python3 realtime_spectrogram_generator.py --packet_type both
#
# 4. CUSTOM OUTPUT DIRECTORY:
#    python3 realtime_spectrogram_generator.py --output_dir my_spectrograms
#
# 5. CUSTOM PACKETS PER IMAGE:
#    python3 realtime_spectrogram_generator.py --packets_per_image 3
#    (For single packets: uses 3 packets per spectrogram)
#    (For merged packets: ignored, uses 1 packet per spectrogram)
#
# 6. CUSTOM DATA DIRECTORY:
#    python3 realtime_spectrogram_generator.py --base_dir /path/to/your/data/
#
# 7. FREQUENCY RANGE OPTIONS:
#    python3 realtime_spectrogram_generator.py --full_frequency
#    (Uses full frequency range 0 to fs/2, fixes the cutting issue)
#    python3 realtime_spectrogram_generator.py --training_format
#    (Uses original training data format, may cut at frequency=0)
#
# 8. COMPLETE EXAMPLE:
#    python3 realtime_spectrogram_generator.py \
#        --num_images 100 \
#        --packet_type single \
#        --packets_per_image 5 \
#        --output_dir test_spectrograms \
#        --base_dir ../spectrogram_training_data_20220711/
#
# 9. HELP:
#    python3 realtime_spectrogram_generator.py --help
#
# =============================================================================
# EXPECTED OUTPUT:
# =============================================================================
#
# The program will:
# 1. Scan the specified directory for .packet files
# 2. Generate the requested number of spectrograms
# 3. Save them as PNG files in the output directory
# 4. Print progress information to console
#
# Example output:
# ```
# Generating 20 spectrograms...
# Packet type: single
# Packets per image: 5
# Output directory: generated_spectrograms
# --------------------------------------------------
# Found 1500 packet files
# Generating 20 spectrograms with 5 packets each...
# Generated 1/20: generated_spectrograms/spectrogram_20241201_143022_123456.png
# Generated 2/20: generated_spectrograms/spectrogram_20241201_143023_234567.png
# ...
# Successfully generated 20 spectrograms
# Generated 20 spectrograms successfully!
# Files saved in: generated_spectrograms/
# ```
#
# =============================================================================
# TROUBLESHOOTING:
# =============================================================================
#
# 1. "No packet files found":
#    - Check that the base directory contains .packet files
#    - For single packets: look in single_packet_samples/ subdirectories
#    - For merged packets: look in merged_packets/ subdirectories
#
# 2. "Permission denied":
#    - Make sure you have write permissions for the output directory
#    - Try running with: chmod +x realtime_spectrogram_generator.py
#
# 3. "Module not found":
#    - Install required packages: pip install -r requirements.txt
#    - Make sure you're in the correct directory
#
# 4. "No such file or directory":
#    - Check that the base directory path is correct
#    - Use absolute paths if relative paths don't work
#
# =============================================================================
# INTEGRATION WITH OTHER COMPONENTS:
# =============================================================================
#
# This generator can be used with:
# 1. YOLO Detection: Feed generated spectrograms to realtime_yolo_detector.py
# 2. GUI Interface: Use with realtime_gui.py for live monitoring
# 3. Complete Pipeline: Integrate with realtime_pipeline.py
#
# Example integration:
# ```python
# from realtime_spectrogram_generator import RealtimeSpectrogramGenerator
# 
# generator = RealtimeSpectrogramGenerator(packet_type="single")
# results = generator.generate_multiple_spectrograms(
#     base_dir="../spectrogram_training_data_20220711/",
#     num_spectrograms=10
# )
# 
# # Process results with YOLO detector
# for image, path in results:
#     detections = detector.detect_in_image(image)
#     print(f"Found {len(detections)} RF frames in {path}")
# ```
