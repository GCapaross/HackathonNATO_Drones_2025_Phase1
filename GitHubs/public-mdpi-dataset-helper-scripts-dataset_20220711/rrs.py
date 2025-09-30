"""
RF Frame Aggregation and Spectrogram Generation
-----------------------------------------------
Reads raw .packet IQ samples, aggregates multiple frames into
a composite signal, applies noise and frequency offset, and
generates a spectrogram image.

Author: ChatGPT (based on Wicht et al. 2022 dataset description)
"""

import numpy as np
import matplotlib.pyplot as plt
from scipy.signal import spectrogram
import os
import datetime
# -------------------------------
# 1. Helper: read .packet file
# -------------------------------
def read_packet_file(filename: str) -> np.ndarray:
    """
    Read a .packet file and return complex IQ samples.
    Each file = interleaved float32 pairs [I0, Q0, I1, Q1, ...].
    """
    raw = np.fromfile(filename, dtype=np.float32)
    iq_samples = raw[0::2] + 1j * raw[1::2]
    return iq_samples

# -------------------------------
# 2. Aggregate frames
# -------------------------------
def aggregate_frames(file_list, section_length=4.5e-3, fs=125e6, noise_sigma=0.01):
    """
    Create a composite buffer of several frames.
    - section_length: length of the output buffer in seconds
    - fs: sample rate (Hz)
    - noise_sigma: std of added Gaussian noise
    """
    n_samples = int(section_length * fs) # 4.5 ms at 125 MS/s
    buffer = np.zeros(n_samples, dtype=np.complex64)

    for f in file_list:
        iq = read_packet_file(f)

        # Random start position (leave space so frame fits)
        start = np.random.randint(0, max(1, n_samples - len(iq)))
        buffer[start:start+len(iq)] += iq

    # Add AWGN
    noise = (np.random.normal(0, noise_sigma, n_samples) +
             1j*np.random.normal(0, noise_sigma, n_samples))
    buffer += noise

    return buffer

# -------------------------------
# 3. Generate spectrogram
# -------------------------------
def make_spectrogram(iq_samples, fs, out_file="spectrogram.png",
                     nperseg=512, noverlap=256):
    """
    Generate and save a spectrogram from IQ samples.
    """
    f, t, Sxx = spectrogram(iq_samples,
                            fs=fs,
                            window='hann',
                            nperseg=nperseg,
                            noverlap=noverlap,
                            return_onesided=False)  # <-- keep both halves

    # Convert to dB
    Sxx_dB = 10 * np.log10(np.abs(Sxx) + 1e-12)

    plt.figure(figsize=(10, 4))
    plt.pcolormesh(t*1e6, f/1e6, Sxx_dB, shading='gouraud', cmap='viridis')
    plt.ylabel("Frequency [MHz]")
    plt.xlabel("Time [µs]")
    plt.colorbar(label="Power [dB]")
    plt.title("Generated Spectrogram")
    plt.tight_layout()
    plt.savefig(out_file, dpi=150)
    plt.close()
    print(f"Spectrogram saved as {out_file}")

# -------------------------------
# 4. Example usage
# -------------------------------
if __name__ == "__main__":
    # Path to folder with .packet files (change to your dataset path!)
    data_dir = "../single_packet_samples/"
    data_frames = 40
    files = []
    for d in os.listdir(data_dir):
        if not os.path.isdir(os.path.join(data_dir, d)): 
            continue
        files +=[os.path.join(data_dir, os.path.join(d,f)) for f in os.listdir(os.path.join(data_dir,d)) if f.endswith(".packet")]

    # Pick a few random frames
    selected_files = np.random.choice(files, size=10, replace=False)

    print(f"Selected files:\n ")
    for f in selected_files:
        print(f" - {f}")
    # Aggregate them
    # AWGN
    composite_signal = aggregate_frames(selected_files,
                                        section_length=4.5e-3,
                                        fs=125e6,
                                        noise_sigma=0.005)

    # Generate spectrogram
    make_spectrogram(composite_signal, fs=125e6, out_file=f"example_spectrogram_{datetime.datetime.now().strftime('%Y-%m-%d_%H-%M-%S')}.png")
