#!/usr/bin/env python3
"""
Test script for generating spectrograms from merged packets.

This script demonstrates how to use the generator_from_merged_packets.py
to create spectrograms with labels from the mixed signals.
"""

import subprocess
import sys
from pathlib import Path

def test_merged_packets_generator():
    """Test the merged packets spectrogram generation"""
    
    # Path to the merged_packets directory
    merged_packets_dir = Path("/home/gabriel/Desktop/HackathonNATO_Drones_2025/GitHubs/public-mdpi-dataset-helper-scripts-dataset_20220711/spectrogram_training_data_20220711/merged_packets")
    
    # Check if directory exists
    if not merged_packets_dir.exists():
        print(f"Merged packets directory not found at {merged_packets_dir}")
        print("Please update the merged_packets_dir variable in this script")
        return False
    
    print("Testing merged packets spectrogram generation...")
    print(f"Merged packets directory: {merged_packets_dir}")
    
    # Test with a small number of samples first
    cmd = [
        sys.executable, 
        "spectrogram_images/generator_from_merged_packets.py",
        "-p", str(merged_packets_dir),
        "-o", "test_merged_output",
        "-r", "1024", "192",
        "--max-samples", "5"
    ]
    
    print(f"Running command: {' '.join(cmd)}")
    
    try:
        result = subprocess.run(cmd, capture_output=True, text=True, cwd=Path(__file__).parent)
        
        if result.returncode == 0:
            print("✅ Merged packets generation test completed successfully!")
            print("Check the 'test_merged_output' directory for generated spectrograms")
            print("\nGenerated files should show:")
            print("- Spectrograms with colored bounding boxes")
            print("- Class labels on each bounding box")
            print("- Different colors for different classes:")
            print("  - Red: WLAN")
            print("  - Yellow: collision") 
            print("  - Blue: BT_classic")
            print("  - Green: BLE_1MHz")
            print("  - Purple: BLE_2MHz")
        else:
            print("❌ Merged packets generation test failed!")
            print(f"Error: {result.stderr}")
            return False
            
    except Exception as e:
        print(f"❌ Error running test: {e}")
        return False
    
    return True

def show_usage_examples():
    """Show usage examples for the merged packets generator"""
    
    print("\n" + "="*60)
    print("USAGE EXAMPLES")
    print("="*60)
    
    print("\n1. Generate spectrograms from merged packets:")
    print("   python3 spectrogram_images/generator_from_merged_packets.py \\")
    print("       -p /path/to/merged_packets \\")
    print("       -o output_directory \\")
    print("       -r 1024 192")
    
    print("\n2. Generate with custom settings:")
    print("   python3 spectrogram_images/generator_from_merged_packets.py \\")
    print("       -p /path/to/merged_packets \\")
    print("       -o custom_output \\")
    print("       -c viridis \\")
    print("       -r 1024 192 \\")
    print("       --max-samples 100")
    
    print("\n3. Test with small sample:")
    print("   python3 spectrogram_images/generator_from_merged_packets.py \\")
    print("       -p /path/to/merged_packets \\")
    print("       -o test_output \\")
    print("       --max-samples 5")
    
    print("\n" + "="*60)
    print("FEATURES")
    print("="*60)
    print("✅ Reads mixed signals from merged_packets")
    print("✅ Converts CSV labels to YOLO format")
    print("✅ Generates spectrograms with bounding boxes")
    print("✅ Color-coded bounding boxes by class")
    print("✅ Class labels on bounding boxes")
    print("✅ Multiple bandwidth support (25MHz, 45MHz, 60MHz, 125MHz)")
    print("✅ Parallel processing for speed")

if __name__ == "__main__":
    success = test_merged_packets_generator()
    
    if success:
        print("\n🎉 Test completed successfully!")
        show_usage_examples()
    else:
        print("\n❌ Test failed. Please check the error messages above.")
        sys.exit(1)
