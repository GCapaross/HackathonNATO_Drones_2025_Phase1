from dataclasses import dataclass
from pathlib import Path
from typing import List


@dataclass
class YoloLabel:
    """Container for label in yolo format"""

    x: float
    y: float
    width: float
    height: float
    class_id: int


def read_yolo_labels(labels_file: Path) -> List[YoloLabel]:
    """Read yolo label from .txt file.

    :param labels_file: path to label file in yolo format
    :type labels_file: Path
    :return: All labels from file
    :rtype: List[YoloLabel]
    """
    labels = []
    with open(labels_file) as file:
        for line in file:
            [_class_id, _x, _y, _width, _height] = line.split()

            labels.append(
                YoloLabel(
                    x=float(_x),
                    y=float(_y),
                    width=float(_width),
                    height=float(_height),
                    class_id=int(_class_id),
                )
            )

    return labels
