from typing import Tuple, Optional
from copy import deepcopy
import csv
import os


"""
FOR ANNOTATIONS
"""


def read_annotations(csv_file):
    annotations = {}
    with open(csv_file, "r") as f:
        reader = csv.DictReader(f)
        for row in reader:
            filename = row["filename"]
            if filename not in annotations:
                annotations[filename] = []
            annotations[filename].append(
                {
                    "x1": int(row["x1"]),
                    "x2": int(row["x2"]),
                    "y1": int(row["y1"]),
                    "y2": int(row["y2"]),
                    "max_x": int(row["max_x"]),
                    "max_y": int(row["max_y"]),
                }
            )
    return annotations


def parse_bad_annotations(bad_bbox_dir_path):
    bad_rois = os.listdir(bad_bbox_dir_path)
    names_xy = dict()
    additional_names = []

    for i in range(len(bad_rois)):
        name_xy = bad_rois[i].split(".")[0].split("_")
        if len(name_xy) == 6:
            name_xy[0:4]
            name = "_".join(name_xy[:2]) + ".tif"
            xy = list(map(int, name_xy[2:]))
            names_xy[name] = xy

        else:
            additional_names.append(name_xy[0] + ".tif")

    return names_xy, additional_names


def clean_annotations(annotations, bad_annotations):
    names_xy, additional_names = bad_annotations
    cleaned_annotations = deepcopy(annotations)

    for bad_name, (x1, y1, x2, y2) in names_xy.items():
        if bad_name in annotations:
            cleaned_annotations[bad_name] = [
                ann for ann in annotations[bad_name] if not (x1 == ann["x1"] and x2 == ann["x2"] and y1 == ann["y1"] and y2 == ann["y2"])
            ]
            if not cleaned_annotations[bad_name]:
                del cleaned_annotations[bad_name]

    for name in additional_names:
        if name in cleaned_annotations:
            del cleaned_annotations[name]

    return cleaned_annotations


def transform_boxes_format(boxes):
    mapped_boxes = []

    for box in boxes:
        x1, y1 = box["x1"], box["y1"]
        width = box["x2"] - box["x1"]
        height = box["y2"] - box["y1"]

        mapped_boxes.append((int(x1), int(y1), int(width), int(height)))

    return mapped_boxes


def get_cleaned_annotations_xywh(csv_file_path, bad_bbox_dir_path):
    annotations = read_annotations(csv_file_path)
    bad_annotations = parse_bad_annotations(bad_bbox_dir_path)
    cleaned_annotations = clean_annotations(annotations, bad_annotations)

    for wsi_name in cleaned_annotations:
        cleaned_annotations[wsi_name] = transform_boxes_format(cleaned_annotations[wsi_name])

    return cleaned_annotations


def is_box_included(inner_box: Tuple[int, int, int, int], outer_box: Tuple[int, int, int, int]) -> bool:
    x1, y1, w1, h1 = inner_box
    x2, y2, w2, h2 = outer_box
    return x1 >= x2 and y1 >= y2 and x1 + w1 <= x2 + w2 and y1 + h1 <= y2 + h2


def find_intersection(box1: Tuple[int, int, int, int], box2: Tuple[int, int, int, int]) -> Optional[Tuple[int, int, int, int]]:
    x1, y1, w1, h1 = box1
    x2, y2, w2, h2 = box2
    x_intersection = max(x1, x2)
    y_intersection = max(y1, y2)
    x_end_intersection = min(x1 + w1, x2 + w2)
    y_end_intersection = min(y1 + h1, y2 + h2)

    if x_intersection < x_end_intersection and y_intersection < y_end_intersection:
        return (x_intersection, y_intersection, x_end_intersection - x_intersection, y_end_intersection - y_intersection)
    return None


def adjust_tissue_box(tissue_box: Tuple[int, int, int, int], annotation_box: Tuple[int, int, int, int]) -> Tuple[int, int, int, int]:
    x_tissue, y_tissue, w_tissue, h_tissue = tissue_box
    x_annotation, y_annotation, w_annotation, h_annotation = annotation_box

    # Calculate the new coordinates of the tissue box if needed
    if x_annotation < x_tissue:
        x_tissue = x_annotation
    if y_annotation < y_tissue:
        y_tissue = y_annotation
    if x_annotation + w_annotation > x_tissue + w_tissue:
        w_tissue = x_annotation + w_annotation - x_tissue
    if y_annotation + h_annotation > y_tissue + h_tissue:
        h_tissue = y_annotation + h_annotation - y_tissue

    # Return the adjusted tissue box coordinates
    return (x_tissue, y_tissue, w_tissue, h_tissue)


def convert_to_yolo_format(
    box: Tuple[int, int, int, int], image_width: int, image_height: int, class_id=0
) -> Tuple[int, float, float, float, float]:
    x, y, w, h = box

    if image_width == 0 or image_height == 0:
        raise ValueError("Image width and height must be greater than zero.")

    x_center = (x + w / 2) / image_width
    y_center = (y + h / 2) / image_height
    width = w / image_width
    height = h / image_height

    assert all(0 <= val <= 1 for val in (x_center, y_center, width, height)), "YOLO coordinates must be in [0, 1]"

    return (class_id, x_center, y_center, width, height)
