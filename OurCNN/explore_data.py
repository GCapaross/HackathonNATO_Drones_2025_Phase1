#!/usr/bin/env python3
"""
Smart data exploration script for NATO hackathon
Analyzes the spectrogram dataset without loading everything into memory
"""

import os
import glob
import numpy as np
from PIL import Image
import matplotlib.pyplot as plt
from collections import Counter
import json

def analyze_dataset_structure(data_path):
    """Analyze the dataset structure and get basic statistics"""
    print("Analyzing dataset structure...")
    
    # Get all files
    all_files = glob.glob(os.path.join(data_path, "*"))
    
    # Categorize files
    png_files = [f for f in all_files if f.endswith('.png')]
    txt_files = [f for f in all_files if f.endswith('.txt')]
    marked_files = [f for f in all_files if 'marked' in f]
    regular_files = [f for f in all_files if f.endswith('.png') and 'marked' not in f]
    
    print(f" Dataset Statistics:")
    print(f"   Total files: {len(all_files)}")
    print(f"   PNG files: {len(png_files)}")
    print(f"   TXT files: {len(txt_files)}")
    print(f"   Marked files: {len(marked_files)}")
    print(f"   Regular spectrograms: {len(regular_files)}")
    
    # Analyze bandwidths
    bandwidths = []
    for file in png_files[:100]:  # Sample first 100 files
        if 'bw_' in file:
            bw_part = file.split('bw_')[1].split('_')[0]
            bandwidths.append(bw_part)
    
    bw_counts = Counter(bandwidths)
    print(f"\n Bandwidth distribution (sample):")
    for bw, count in bw_counts.items():
        print(f"   {bw}: {count} files")
    
    return {
        'total_files': len(all_files),
        'png_files': len(png_files),
        'txt_files': len(txt_files),
        'marked_files': len(marked_files),
        'regular_files': len(regular_files),
        'bandwidths': dict(bw_counts)
    }

def sample_images(data_path, num_samples=5):
    """Sample a few images to understand the data format"""
    print(f"\n Sampling {num_samples} images...")
    
    # Get sample files
    png_files = glob.glob(os.path.join(data_path, "*.png"))
    regular_files = [f for f in png_files if 'marked' not in f]
    
    samples = regular_files[:num_samples]
    
    for i, img_path in enumerate(samples):
        try:
            # Load image
            img = Image.open(img_path)
            print(f"   Sample {i+1}: {os.path.basename(img_path)}")
            print(f"      Size: {img.size}")
            print(f"      Mode: {img.mode}")
            
            # Check if there's a corresponding txt file
            txt_path = img_path.replace('.png', '.txt')
            if os.path.exists(txt_path):
                with open(txt_path, 'r') as f:
                    content = f.read().strip()
                    print(f"      Label: {content[:50]}...")
            
        except Exception as e:
            print(f"   Error with {img_path}: {e}")
    
    return samples

def create_data_summary(data_path):
    """Create a summary of the dataset for our CNN approach"""
    print("\n Creating dataset summary...")
    
    # Analyze structure
    stats = analyze_dataset_structure(data_path)
    
    # Sample images
    samples = sample_images(data_path, 3)
    
    # Create summary
    summary = {
        'dataset_path': data_path,
        'statistics': stats,
        'sample_files': samples,
        'recommendations': {
            'use_regular_png': True,
            'check_txt_labels': True,
            'preprocessing_needed': True,
            'suggested_approach': 'CNN for spectrogram classification'
        }
    }
    
    # Save summary
    with open('dataset_summary.json', 'w') as f:
        json.dump(summary, f, indent=2)
    
    print("Dataset summary saved to 'dataset_summary.json'")
    return summary

if __name__ == "__main__":
    # Set data path
    data_path = "/home/gabriel/Desktop/HackathonNATO_Drones_2025/spectrogram_training_data_20220711/results"
    
    print("Starting smart data exploration...")
    print(f"Data path: {data_path}")
    
    # Create summary
    summary = create_data_summary(data_path)
    
    print("\n Next steps:")
    print("1. Review dataset_summary.json")
    print("2. Set up preprocessing pipeline")
    print("3. Create CNN model")
    print("4. Implement YOLO for detection")
