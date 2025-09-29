#!/usr/bin/env python3
"""
Test script for generating spectrograms with frequency labels and YOLO bounding boxes.

This script demonstrates how to use the enhanced generator for creating spectrograms with:
1. Frequency axis labels (MHz)
2. YOLO bounding box labels overlaid
3. Proper coordinate system handling
"""

import subprocess
import sys
from pathlib import Path

def test_frequency_spectrogram_generation():
    """Test the enhanced spectrogram generation with frequency labels"""
    
    # Path to the dataset
    dataset_path = Path("/home/gabriel/Desktop/HackathonNATO_Drones_2025/GitHubs/public-mdpi-dataset-helper-scripts-dataset_20220711/spectrogram_training_data_20220711")
    
    # Check if dataset exists
    if not dataset_path.exists():
        print(f"Dataset not found at {dataset_path}")
        print("Please update the dataset_path variable in this script")
        return False
    
    print("Testing enhanced spectrogram generation with frequency labels and YOLO bounding boxes...")
    print(f"Dataset path: {dataset_path}")
    
    # Test 1: Generate spectrograms with frequency axis labels
    print("\n=== Test 1: Spectrograms with frequency axis labels ===")
    cmd1 = [
        sys.executable, 
        "spectrogram_images/main_with_frequency_labels.py",
        "-p", str(dataset_path),
        "-r", "1024", "192",
        "-o", "test_frequency_output",
        "--max-samples", "3",
        "--center-freq", "2.412e9"
    ]
    
    print(f"Running command: {' '.join(cmd1)}")
    
    try:
        result1 = subprocess.run(cmd1, capture_output=True, text=True, cwd=Path(__file__).parent)
        
        if result1.returncode == 0:
            print("✅ Frequency spectrogram generation test completed successfully!")
            print("Check the 'test_frequency_output' directory for generated spectrograms")
        else:
            print("❌ Frequency spectrogram generation test failed!")
            print(f"Error: {result1.stderr}")
            return False
            
    except Exception as e:
        print(f"❌ Error running frequency test: {e}")
        return False
    
    # Test 2: Generate spectrograms without frequency axis (for YOLO training)
    print("\n=== Test 2: Spectrograms without frequency axis (YOLO training format) ===")
    cmd2 = [
        sys.executable, 
        "spectrogram_images/main_with_frequency_labels.py",
        "-p", str(dataset_path),
        "-r", "1024", "192",
        "-o", "test_yolo_output",
        "--max-samples", "3",
        "--no-frequency-axis",
        "--center-freq", "2.412e9"
    ]
    
    print(f"Running command: {' '.join(cmd2)}")
    
    try:
        result2 = subprocess.run(cmd2, capture_output=True, text=True, cwd=Path(__file__).parent)
        
        if result2.returncode == 0:
            print("✅ YOLO format spectrogram generation test completed successfully!")
            print("Check the 'test_yolo_output' directory for generated spectrograms")
        else:
            print("❌ YOLO format spectrogram generation test failed!")
            print(f"Error: {result2.stderr}")
            return False
            
    except Exception as e:
        print(f"❌ Error running YOLO test: {e}")
        return False
    
    return True

def show_usage_examples():
    """Show usage examples for the enhanced spectrogram generator"""
    
    print("\n" + "="*60)
    print("USAGE EXAMPLES")
    print("="*60)
    
    print("\n1. Generate spectrograms with frequency axis labels:")
    print("   python3 spectrogram_images/main_with_frequency_labels.py \\")
    print("       -p /path/to/dataset \\")
    print("       -r 1024 192 \\")
    print("       -o frequency_spectrograms \\")
    print("       --center-freq 2.412e9")
    
    print("\n2. Generate spectrograms for YOLO training (no frequency axis):")
    print("   python3 spectrogram_images/main_with_frequency_labels.py \\")
    print("       -p /path/to/dataset \\")
    print("       -r 1024 192 \\")
    print("       -o yolo_training_data \\")
    print("       --no-frequency-axis")
    
    print("\n3. Generate spectrograms with custom settings:")
    print("   python3 spectrogram_images/main_with_frequency_labels.py \\")
    print("       -p /path/to/dataset \\")
    print("       -c viridis \\")
    print("       -r 1024 192 \\")
    print("       -o custom_output \\")
    print("       --center-freq 2.472e9 \\")
    print("       --max-samples 100")
    
    print("\n" + "="*60)
    print("FEATURES")
    print("="*60)
    print("✅ Frequency axis labels (MHz)")
    print("✅ YOLO bounding box labels")
    print("✅ Class labels on bounding boxes")
    print("✅ Color-coded bounding boxes")
    print("✅ Proper coordinate system handling")
    print("✅ Multiple sample rate support")
    print("✅ Configurable center frequency")
    print("✅ YOLO training format option")

if __name__ == "__main__":
    success = test_frequency_spectrogram_generation()
    
    if success:
        print("\n🎉 All tests completed successfully!")
        show_usage_examples()
    else:
        print("\n❌ Tests failed. Please check the error messages above.")
        sys.exit(1)
