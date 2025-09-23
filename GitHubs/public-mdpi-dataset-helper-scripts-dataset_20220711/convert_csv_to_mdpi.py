#!/usr/bin/env python3
"""
Convert DroneRF CSV files to mdpi dataset format.
Creates complex64 binary files and label CSVs for spectrogram generation.
"""

import numpy as np
import pathlib
import csv

def convert_csv_to_mdpi():
    # Paths
    csv_dir = pathlib.Path("G:/Programing/HackathonNATO_Drones_2025/csv_files_dataset")
    out_root = pathlib.Path("G:/Programing/HackathonNATO_Drones_2025/GitHubs/public-mdpi-dataset-helper-scripts-dataset_20220711")
    results = out_root / "results"
    labels_dir = out_root / "merged_packets" / "bw_40e6"
    BW_HZ = 40000000  # 40 MHz bandwidth
    
    # Create directories
    results.mkdir(parents=True, exist_ok=True)
    labels_dir.mkdir(parents=True, exist_ok=True)
    
    print("Converting CSVs to mdpi format...")
    print(f"Input: {csv_dir}")
    print(f"Output: {out_root}")
    
    count = 0
    csv_files = list(csv_dir.glob("*.csv"))
    print(f"Found {len(csv_files)} CSV files")
    
    for csv_path in sorted(csv_files):
        stem = csv_path.stem
        print(f"Processing: {csv_path.name}")
        
        # Load CSV data
        data = np.loadtxt(csv_path, delimiter=",", dtype=np.float32, ndmin=1)
        print(f"  Shape: {data.shape}, Range: {data.min():.1f} to {data.max():.1f}")
        
        # Convert to complex64 (imaginary part = 0)
        complex_samples = data.astype(np.complex64)
        
        # Save as binary file
        bin_name = f"result_frame_{stem}_bw_{BW_HZ}.bin"
        complex_samples.tofile(results / bin_name)
        
        # Create label CSV (mdpi expects labels_ prefix, and will add .csv)
        label_name = f"labels_{stem}_bw_{BW_HZ}.bin.csv"
        with open(labels_dir / label_name, "w", newline="") as f:
            w = csv.writer(f)
            w.writerow(["noise_lvl"])
            w.writerow(["normal"])  # Not "usrp_txrx_loop" to avoid filtering
        
        count += 1
        if count % 5 == 0:
            print(f"  Converted {count} files...")
    
    print(f"\nDone! Converted {count} CSV files to mdpi format")
    print(f"Binary files: {results}")
    print(f"Label files: {labels_dir}")

if __name__ == "__main__":
    convert_csv_to_mdpi()
