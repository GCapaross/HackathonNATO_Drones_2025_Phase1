import datetime
import json
from dataclasses import dataclass
from pathlib import Path
from typing import List

from label_converter.yolo_to_coco.dtype_labels import YoloLabel


@dataclass
class BoundingBox:
    """
    x and y describe the top left corner of the Bounding Box in absolute pixels in the image.
    Image coordinate systems origin (0,0) is also defined at top left corner."""

    top_left_x: float
    top_left_y: float
    width: float
    height: float
    rf_std: str


@dataclass
class ImageDescriptor:
    """Image properties required by the COCO format"""

    id: int
    license: int
    file_name: str
    license: int
    height_px: int
    width_px: int
    flickr_url: str = ""
    coco_url: str = ""
    date_captured = None


class CocoLabels:
    """COCO Label object that contains the bounding boxes for an image in the correct format.
    Expects an ImageDescriptor object and a List[YoloLabel] as input to infer the correct bounding boxes.
    """

    categories = ["WLAN", "collision", "bluetooth"]

    def __init__(
        self, img_descr: ImageDescriptor, yolo_labels: List[YoloLabel]
    ) -> None:
        self.bboxes = []
        for yolo_label in yolo_labels:
            _top_left_y = (yolo_label.y - yolo_label.height / 2) * img_descr.height_px
            _top_left_x = (yolo_label.x - yolo_label.width / 2) * img_descr.width_px
            _width = yolo_label.width * img_descr.width_px
            _height = yolo_label.height * img_descr.height_px
            self.bboxes.append(
                BoundingBox(
                    top_left_x=_top_left_x,
                    top_left_y=_top_left_y,
                    width=_width,
                    height=_height,
                    rf_std=self.categories[yolo_label.class_id],
                )
            )


class CocoConverter:
    """COCO container. Initializes COCO header"""

    def __init__(self, categories: List[str], info=None, licenses=None):
        self.info = info or {
            "description": "Spectrogram Data Set for Deep Learning Based RF-Frame Detection",
            "url": "https://fordatis.fraunhofer.de/handle/fordatis/287",
            "version": "1.0.0",
            "year": 2022,
            "contributor": "Jakob Wicht",
            "date_created": datetime.datetime.utcnow().isoformat(" "),
        }
        self.licenses = licenses or [
            {
                "id": int(1),
                "name": "Attribution 4.0 International License",
                "url": "https://creativecommons.org/licenses/by/4.0/",
            }
        ]
        self.categories = []
        for index, cat in enumerate(categories):
            self.categories.append(
                {
                    "id": int(index),
                    "name": cat,
                    "supercategory": "RF Frame",
                }
            )
        self.images = []
        self.annotations = []
        self.annotation_index = 0

    def append_annotations(self, coco_labels: CocoLabels, im_descr: ImageDescriptor):
        """Add an image and corresponding bounding boxes to COCO handle.

        :param labels_list: _description_
        :type labels_list: LabelsList
        """

        # Append image
        self.images.append(
            {
                "id": im_descr.id,
                "license": im_descr.license,
                "file_name": im_descr.file_name,
                "height": im_descr.height_px,  # 480,
                "width": im_descr.width_px,  # 640,
                "date_captured": im_descr.date_captured,
                "flickr_url": im_descr.flickr_url,
                "coco_url": im_descr.coco_url,
            }
        )

        # Append annotation
        for bbox in coco_labels.bboxes:
            # bbox = labels_list.get_bbox(label, im_descr)
            # width = bbox.bot_right_x - bbox.top_left_x
            # height = bbox.bot_right_y - bbox.top_left_y
            self.annotations.append(
                {
                    "id": self.annotation_index,
                    "image_id": im_descr.id,
                    "category_id": bbox.rf_std,
                    # bbox = [x,y,width,height]; x, y:
                    # the upper-left coordinates of the bounding box
                    "bbox": [
                        float(bbox.top_left_x),
                        float(bbox.top_left_y),
                        float(bbox.width),
                        float(bbox.height),
                        # int(bbox.top_left_x),
                        # int(bbox.top_left_y),
                        # int(width),
                        # int(height),
                    ],
                    "segmentation": [],
                    "area": int(bbox.width * bbox.height),
                    "iscrowd": 0,
                }
            )
            self.annotation_index += 1

    def export(self, file_out: Path):
        """Store COCO handle to actual json file in the datasets result folder

        :param file_out: output file path
        :type file_out: Path
        """

        coco_dict = {
            "info": self.info,
            "licenses": self.licenses,
            "categories": self.categories,
            "images": self.images,
            "annotations": self.annotations,
        }

        with open(file_out, "w", encoding="utf-8") as f:
            json.dump(coco_dict, f, ensure_ascii=False, indent=4)
