#!/usr/bin/env python

from setuptools import setup, find_packages

setup(
    name="sunrise_mdpi_helper_scripts",
    version="1.0",
    description="helper scripts for published Sunrise mdpi dataset",
    author="Jakob Wicht",
    author_email="jakob.wicht@eas.iis.fraunhofer.de",
    packages=find_packages(
        include=[
            "spectrogram_images.generator",
            "spectrogram_images.directory_manager",
            "spectrogram_images.sample_files_list",
            "label_converter.yolo_to_coco.dtype_labels",
            "label_converter.yolo_to_coco.converter",
        ]
    ),
    install_requires=["matplotlib", "numpy", "pandas", "joblib"],
)