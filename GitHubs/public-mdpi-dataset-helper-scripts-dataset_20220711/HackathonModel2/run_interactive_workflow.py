#!/usr/bin/env python3
"""
Interactive Workflow Runner
=========================
Generate spectrograms first, then open interactive pipeline for evaluation.

Usage:
    python3 run_interactive_workflow.py --model MODEL_PATH --num_images 20
"""

import os
import sys
import subprocess
import argparse
from pathlib import Path

def generate_spectrograms(num_images=20, packets_per_image=5, output_dir="interactive_output"):
    """Generate spectrograms for interactive evaluation."""
    print(f"Generating {num_images} spectrograms...")
    
    cmd = [
        "python3", "realtime_spectrogram_generator.py",
        "--num_images", str(num_images),
        "--packets_per_image", str(packets_per_image),
        "--output_dir", output_dir,
        "--packet_type", "single"
    ]
    
    try:
        result = subprocess.run(cmd, check=True, capture_output=True, text=True)
        print("Spectrograms generated successfully!")
        print(f"Output directory: {output_dir}")
        return True
    except subprocess.CalledProcessError as e:
        print(f"Error generating spectrograms: {e}")
        print(f"Error output: {e.stderr}")
        return False

def open_interactive_pipeline(model_path, output_dir="interactive_output"):
    """Open the interactive pipeline."""
    print(f"Opening interactive pipeline with model: {model_path}")
    
    cmd = [
        "python3", "interactive_pipeline.py",
        "--model", model_path,
        "--output", output_dir
    ]
    
    try:
        subprocess.run(cmd, check=True)
    except subprocess.CalledProcessError as e:
        print(f"Error opening interactive pipeline: {e}")

def main():
    """Main workflow function."""
    parser = argparse.ArgumentParser(description="Interactive RF Detection Workflow")
    parser.add_argument("--model", type=str, required=True, help="Path to YOLO model")
    parser.add_argument("--num_images", type=int, default=20, help="Number of spectrograms to generate")
    parser.add_argument("--packets_per_image", type=int, default=5, help="Packets per spectrogram")
    parser.add_argument("--output_dir", type=str, default="interactive_output", help="Output directory")
    parser.add_argument("--skip_generation", action="store_true", help="Skip spectrogram generation")
    
    args = parser.parse_args()
    
    # Check if model exists
    if not os.path.exists(args.model):
        print(f"Error: Model file not found: {args.model}")
        return
    
    # Generate spectrograms (unless skipped)
    if not args.skip_generation:
        if not generate_spectrograms(args.num_images, args.packets_per_image, args.output_dir):
            print("Failed to generate spectrograms. Exiting.")
            return
    else:
        print("Skipping spectrogram generation...")
    
    # Check if output directory has images
    output_path = Path(args.output_dir)
    png_files = list(output_path.glob("*.png"))
    if not png_files:
        print(f"No PNG files found in {args.output_dir}")
        print("Generate spectrograms first or check the output directory.")
        return
    
    print(f"Found {len(png_files)} spectrograms in {args.output_dir}")
    
    # Open interactive pipeline
    open_interactive_pipeline(args.model, args.output_dir)

if __name__ == "__main__":
    main()
