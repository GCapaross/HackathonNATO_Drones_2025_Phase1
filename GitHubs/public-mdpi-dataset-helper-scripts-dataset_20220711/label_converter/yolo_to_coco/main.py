import argparse
from pathlib import Path

from PIL import Image
from spectrogram_images.directory_manager import DirectoryManager

from label_converter.yolo_to_coco.dtype_labels import read_yolo_labels
from label_converter.yolo_to_coco.converter import (
    CocoConverter,
    CocoLabels,
    ImageDescriptor,
)

parser = argparse.ArgumentParser(
    description="Generate spectrogram images from the time signal sample files. ",
    formatter_class=argparse.ArgumentDefaultsHelpFormatter,
)
parser.add_argument(
    "-p",
    "--path",
    type=Path,
    required=True,
    help="root path of the spectrogram data set",
)

if __name__ == "__main__":
    args = parser.parse_args()

    # container of the datasets folder structure
    dir_manager = DirectoryManager(args.path)

    # iterate through images in the results directory
    im_path_list = dir_manager.path_results.rglob("*6.png")

    # initialize COCO container
    covo_converter = CocoConverter(categories=CocoLabels.categories)

    # convert labels for all images found in the results folder
    img_index = 0
    for im_path in im_path_list:
        im = Image.open(im_path)

        # read corresponding yolo labels for image
        yolo_labels = read_yolo_labels(im_path.with_suffix(".txt"))

        # create descriptor for image file
        img_descr = ImageDescriptor(
            id=img_index,
            license=1,
            file_name=im_path.name,
            width_px=im.size[0],  # (width,height) tuple
            height_px=im.size[1],  # (width,height) tuple
        )
        # convert yolo to coco label
        covo_converter.append_annotations(CocoLabels(img_descr, yolo_labels), img_descr)
        img_index += 1

    # store converted labels to json file
    covo_converter.export(dir_manager.path_results / "labels_coco_format.json")
