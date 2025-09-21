"""Spectrogram image generator

This script allows the user to create spectrograms with own parameters like
resolution and color from signal-sample files of the data set published on MPDI.
Spectrogram images that has been generated using an USRP hardware are not
reproducible by that script.
"""

import multiprocessing
import warnings
from dataclasses import dataclass
from pathlib import Path

import matplotlib
import matplotlib.pyplot as plt
import numpy as np
from joblib import Parallel, delayed

from spectrogram_images.sample_files_list import SampleFileList

# Switch to non-interactive backend for faster saving the spectrogram image to file
matplotlib.use("Agg")


@dataclass
class Resolution:
    """Container for image resolution - number of pixels in x and y."""

    x: int
    y: int

    def __post_init__(self):
        if (self.x % 32 != 0) or (self.y % 32 != 0):
            warnings.warn(
                "Resolution must be a multiple of 32 when used for darknet yolo"
                " algorithm!"
            )


def make_spectrograms(
    sample_files: SampleFileList,
    png_resolution: Resolution,
    colormap: matplotlib.colors.Colormap = "viridis",
    auto_normalization=False,
):
    """
    It takes a list of sample files, creates a spectrogram for each file, and saves the spectrogram as a
    PNG file

    :param sample_files: SampleFileList
    :type sample_files: SampleFileList
    :param png_resolution: The resolution of the output PNG file
    :type png_resolution: Resolution
    :param colormap: The color scheme to use for the spectrogram, defaults to viridis
    :type colormap: matplotlib.colors.Colormap (optional)
    :param auto_normalization: If True, the spectrogram will be normalized to the min and max values of
    the entire dataset. If False, the spectrogram will be normalized to -150dB and -50dB, defaults to
    False (optional)
    """
    print("creating spectrograms...")
    if auto_normalization:
        norm_vmin = None
        norm_vmax = None
    else:
        norm_vmin = -150
        norm_vmax = -50

    def __spectrogram(
        file_path: Path,
        png_resolution: Resolution,
        fft_size,
        norm_vmin=None,
        norm_vmax=None,
    ):
        # Read samples from file
        result_samples = np.fromfile(file_path, dtype=np.complex64)
        # generate spectrogram
        fig, ax = plt.subplots(figsize=(png_resolution.x / 100, png_resolution.y / 100))
        ax.specgram(
            result_samples,
            NFFT=fft_size,
            Fs=42e6,  # Dummy sample rate
            noverlap=0,  # int(FFT_size / 4),
            mode="default",
            sides="default",
            vmin=norm_vmin,
            vmax=norm_vmax,
            window=np.hanning(fft_size),
            cmap=colormap,
        )
        # turn axis off and save spectrogram as png
        fig.subplots_adjust(left=0, right=1, bottom=0, top=1)
        ax.axis("tight")
        ax.axis("off")
        plt.savefig(file_path.with_suffix(".png"))  # , dpi=80)
        plt.close("all")

    # Perform multithreaded spectrogram generation
    num_cores = multiprocessing.cpu_count()
    Parallel(n_jobs=num_cores)(
        delayed(__spectrogram)(
            file_path=sample_file.path,
            png_resolution=png_resolution,
            fft_size=256 if sample_file.sample_rate >= 40 else 128,
            norm_vmin=norm_vmin,
            norm_vmax=norm_vmax,
        )
        for sample_file in sample_files
    )
