"""
Real-time RF Detection Pipeline
==============================
Complete pipeline for real-time RF frame detection:
Raw Signals -> Merge Packets -> Generate Spectrograms -> YOLO Detection -> Live GUI

Usage:
    python3 realtime_pipeline.py [--gui] [--model MODEL_PATH] [--packets PACKET_DIR]
"""

import argparse
import os
import sys
import time
import threading
import queue
from typing import Optional

# Add current directory to path for imports
sys.path.append(os.path.dirname(os.path.abspath(__file__)))

from realtime_spectrogram_generator import RealtimeSpectrogramGenerator
from realtime_yolo_detector import RealtimeYOLODetector
from realtime_gui import RealtimeRFDetectionGUI

class RealtimeRFPipeline:
    """
    Complete real-time RF detection pipeline.
    """
    
    def __init__(self, 
                 model_path: str,
                 packet_source: str,
                 output_dir: str = "realtime_output",
                 processing_interval: float = 2.0):
        """
        Initialize the pipeline.
        
        Args:
            model_path: Path to trained YOLO model
            packet_source: Directory containing .packet files
            output_dir: Directory to save outputs
            processing_interval: Time between processing cycles (seconds)
        """
        self.model_path = model_path
        self.packet_source = packet_source
        self.output_dir = output_dir
        self.processing_interval = processing_interval
        
        # Initialize components
        self.generator = RealtimeSpectrogramGenerator()
        self.detector = RealtimeYOLODetector(model_path)
        
        # Communication queues
        self.spectrogram_queue = queue.Queue(maxsize=5)
        self.detection_queue = queue.Queue(maxsize=5)
        
        # Control
        self.running = False
        self.threads = []
        
        # Statistics
        self.stats = {
            'processed_spectrograms': 0,
            'total_detections': 0,
            'wlan_detections': 0,
            'bluetooth_detections': 0,
            'collision_detections': 0
        }
    
    def start(self):
        """Start the pipeline."""
        if not os.path.exists(self.model_path):
            print(f"Error: Model file not found: {self.model_path}")
            return False
        
        if not os.path.exists(self.packet_source):
            print(f"Error: Packet source directory not found: {self.packet_source}")
            return False
        
        os.makedirs(self.output_dir, exist_ok=True)
        
        print("Starting Real-time RF Detection Pipeline")
        print(f"Model: {self.model_path}")
        print(f"Packet Source: {self.packet_source}")
        print(f"Output Directory: {self.output_dir}")
        print(f"Processing Interval: {self.processing_interval}s")
        print("-" * 50)
        
        self.running = True
        
        # Start processing thread
        self.processing_thread = threading.Thread(target=self._processing_loop, daemon=True)
        self.processing_thread.start()
        self.threads.append(self.processing_thread)
        
        # Start statistics thread
        self.stats_thread = threading.Thread(target=self._stats_loop, daemon=True)
        self.stats_thread.start()
        self.threads.append(self.stats_thread)
        
        return True
    
    def stop(self):
        """Stop the pipeline."""
        print("Stopping pipeline...")
        self.running = False
        
        # Wait for threads to finish
        for thread in self.threads:
            if thread.is_alive():
                thread.join(timeout=2.0)
        
        print("Pipeline stopped")
        self._print_final_stats()
    
    def _processing_loop(self):
        """Main processing loop."""
        while self.running:
            try:
                # Get packet files
                packet_files = self._get_packet_files()
                
                if not packet_files:
                    print("No packet files found, waiting...")
                    time.sleep(5.0)
                    continue
                
                # Select random subset
                n_packets = min(5, len(packet_files))
                selected_files = __import__('numpy').random.choice(packet_files, size=n_packets, replace=False)
                
                # Generate spectrogram
                image, spectrogram_path = self.generator.generate_from_packets(
                    selected_files, self.output_dir
                )
                
                if image is not None:
                    # Run detection
                    result_image, detections, output_path = self.detector.process_spectrogram(
                        image, save_result=True, output_dir=self.output_dir
                    )
                    
                    # Update statistics
                    self.stats['processed_spectrograms'] += 1
                    self.stats['total_detections'] += len(detections)
                    
                    for detection in detections:
                        if detection.class_name == 'WLAN':
                            self.stats['wlan_detections'] += 1
                        elif detection.class_name == 'bluetooth':
                            self.stats['bluetooth_detections'] += 1
                        elif detection.class_name == 'collision':
                            self.stats['collision_detections'] += 1
                    
                    # Print results
                    if detections:
                        print(f"Detected {len(detections)} RF frames:")
                        for det in detections:
                            print(f"  - {det.class_name}: {det.confidence:.3f}")
                    else:
                        print("No RF frames detected")
                    
                    # Add to detection queue for GUI
                    if not self.detection_queue.full():
                        self.detection_queue.put((result_image, detections, output_path))
                
                time.sleep(self.processing_interval)
                
            except KeyboardInterrupt:
                print("Interrupted by user")
                self.running = False
            except Exception as e:
                print(f"Error in processing loop: {e}")
                time.sleep(1.0)
    
    def _stats_loop(self):
        """Statistics reporting loop."""
        while self.running:
            time.sleep(10.0)  # Report every 10 seconds
            if self.running:
                self._print_stats()
    
    def _get_packet_files(self):
        """Get list of available packet files."""
        packet_files = []
        for root, dirs, files in os.walk(self.packet_source):
            for file in files:
                if file.endswith('.packet'):
                    packet_files.append(os.path.join(root, file))
        return packet_files
    
    def _print_stats(self):
        """Print current statistics."""
        print(f"\n--- Statistics ---")
        print(f"Processed Spectrograms: {self.stats['processed_spectrograms']}")
        print(f"Total Detections: {self.stats['total_detections']}")
        print(f"WLAN: {self.stats['wlan_detections']} | Bluetooth: {self.stats['bluetooth_detections']} | Collision: {self.stats['collision_detections']}")
        print("-" * 30)
    
    def _print_final_stats(self):
        """Print final statistics."""
        print("\n=== Final Statistics ===")
        self._print_stats()
        print("Pipeline completed")

def main():
    """Main function."""
    parser = argparse.ArgumentParser(description="Real-time RF Detection Pipeline")
    parser.add_argument("--gui", action="store_true", help="Launch GUI interface")
    parser.add_argument("--model", type=str, help="Path to YOLO model file")
    parser.add_argument("--packets", type=str, help="Path to packet source directory")
    parser.add_argument("--output", type=str, default="realtime_output", help="Output directory")
    parser.add_argument("--interval", type=float, default=2.0, help="Processing interval in seconds")
    
    args = parser.parse_args()
    
    if args.gui:
        # Launch GUI
        print("Launching GUI interface...")
        app = RealtimeRFDetectionGUI()
        app.run()
    else:
        # Command line mode
        if not args.model:
            # Try to find a model automatically
            model_paths = []
            for root, dirs, files in os.walk("."):
                for file in files:
                    if file.endswith('.pt') and 'best' in file:
                        model_paths.append(os.path.join(root, file))
            
            if model_paths:
                args.model = model_paths[0]
                print(f"Auto-selected model: {args.model}")
            else:
                print("Error: No model specified and no models found")
                print("Use --model PATH or --gui to select a model")
                return
        
        if not args.packets:
            args.packets = "../spectrogram_training_data_20220711/single_packet_samples/"
            print(f"Using default packet source: {args.packets}")
        
        # Create and run pipeline
        pipeline = RealtimeRFPipeline(
            model_path=args.model,
            packet_source=args.packets,
            output_dir=args.output,
            processing_interval=args.interval
        )
        
        if pipeline.start():
            try:
                # Keep running until interrupted
                while pipeline.running:
                    time.sleep(1.0)
            except KeyboardInterrupt:
                print("\nInterrupted by user")
            finally:
                pipeline.stop()

if __name__ == "__main__":
    main()
