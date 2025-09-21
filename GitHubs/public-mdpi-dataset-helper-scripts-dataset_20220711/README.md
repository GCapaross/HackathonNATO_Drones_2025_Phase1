# public-mdpi-dataset-helper-scripts

This is a public repository providing scripts for the sunrise data set publication within the mdpi data journal. The corresponding data set is available at [https://fordatis.fraunhofer.de/handle/fordatis/287](https://fordatis.fraunhofer.de/handle/fordatis/287).

## Install

```console
cd /path/to/this/repo
pip install -e .
```

## How to use

### Adjust colormap or resolution of spectrograms

```console
cd /path/to/this/repo
python3 spectrogram_images/main.py --path /root/path/to/dataset --colormap viridis --resolution 1024 192
```

### Store yolo format labels additionally in the COCO json format

```console
cd /path/to/this/repo
python3 label_converter/yolo_to_coco/main.py --path /root/path/to/dataset
```

## License

Creative Commons Attribution 4.0 (cc-by-4.0)
