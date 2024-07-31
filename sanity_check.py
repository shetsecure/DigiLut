import json
import os
import cv2
from tqdm import tqdm


def load_annotations(coco_ann_file):
    with open(coco_ann_file, "r") as f:
        return json.load(f)


def check_annotations(annotations_file, images_dir):
    data = load_annotations(annotations_file)
    images_info = {img["id"]: img for img in data["images"]}
    categories = {cat["id"]: cat["name"] for cat in data["categories"]}

    for ann in tqdm(data["annotations"], total=len(data["annotations"])):
        img_info = images_info[ann["image_id"]]
        img_path = os.path.join(images_dir, img_info["file_name"])
        img = cv2.imread(img_path)

        if img is None:
            print(f"Image {img_info['file_name']} not found.")
            continue

        height, width = img.shape[:2]
        bbox = ann["bbox"]
        x, y, w, h = bbox

        if x < 0 or y < 0 or (x + w) > width or (y + h) > height:
            print(f"Invalid bounding box {bbox} in image {img_info['file_name']}")

        # Optionally visualize the bounding boxes
        # category_name = categories[ann['category_id']]
        # cv2.rectangle(img, (int(x), int(y)), (int(x+w), int(y+h)), (255, 0, 0), 2)
        # cv2.putText(img, category_name, (int(x), int(y-10)), cv2.FONT_HERSHEY_SIMPLEX, 0.9, (255, 0, 0), 2)
        # cv2.imshow("Image", img)
        # cv2.waitKey(0)

    print("Annotation check completed.")


import os
import json
import random
from pathlib import Path
from tqdm import tqdm
import albumentations as A
import cv2
import numpy as np
from pycocotools.coco import COCO


def load_image(image_path):
    return cv2.imread(str(image_path))


def load_coco_annotations(annotation_file):
    coco = COCO(annotation_file)
    return coco


def coco_to_yolo(bbox, image_width, image_height):
    x, y, w, h = bbox
    return [(x + w / 2) / image_width, (y + h / 2) / image_height, w / image_width, h / image_height]


def yolo_to_coco(bbox, image_width, image_height):
    cx, cy, w, h = bbox
    return [(cx - w / 2) * image_width, (cy - h / 2) * image_height, w * image_width, h * image_height]


def save_yolo_annotation(file_path, bboxes, labels, image_width, image_height):
    with open(file_path, "w") as f:
        for bbox, label in zip(bboxes, labels):
            yolo_bbox = coco_to_yolo(bbox, image_width, image_height)
            f.write(f"{label} {' '.join(map(str, yolo_bbox))}\n")


def save_coco_annotation(file_path, bboxes, labels, image_id, annotation_id):
    annotations = []
    for bbox, label in zip(bboxes, labels):
        annotation = {
            "id": annotation_id,
            "image_id": image_id,
            "category_id": label,
            "bbox": bbox,
            "area": bbox[2] * bbox[3],
            "iscrowd": 0,
        }
        annotations.append(annotation)
        annotation_id += 1

    with open(file_path, "w") as f:
        json.dump(annotations, f)

    return annotation_id


def is_valid_bbox(bbox, image_width, image_height):
    x, y, w, h = bbox
    return 0 <= x < image_width and 0 <= y < image_height and w > 0 and h > 0 and x + w <= image_width and y + h <= image_height


def sanity_check(dataset_dir, annotation_format):
    image_files = list(Path(dataset_dir).rglob("*.jpg"))

    for image_file in tqdm(image_files, desc="Performing sanity check"):
        annotation_file = image_file.with_suffix(".txt" if annotation_format == "yolo" else ".json")

        if not annotation_file.exists():
            print(f"Warning: Missing annotation file for {image_file}")
            continue

        image = cv2.imread(str(image_file))
        height, width = image.shape[:2]

        if annotation_format == "yolo":
            with open(annotation_file, "r") as f:
                lines = f.readlines()
            for line in lines:
                parts = line.strip().split()
                if len(parts) != 5:
                    print(f"Warning: Invalid YOLO format in {annotation_file}")
                    continue
                cx, cy, w, h = map(float, parts[1:])
                if not (0 <= cx <= 1 and 0 <= cy <= 1 and 0 <= w <= 1 and 0 <= h <= 1):
                    print(f"Warning: Invalid bounding box in {annotation_file}")
                # Convert YOLO to COCO format for additional checks
                x, y, w, h = yolo_to_coco([cx, cy, w, h], width, height)
                if not is_valid_bbox([x, y, w, h], width, height):
                    print(f"Warning: Bounding box out of image bounds in {annotation_file}")
        else:  # COCO format
            with open(annotation_file, "r") as f:
                annotations = json.load(f)
            for ann in annotations:
                bbox = ann["bbox"]
                if len(bbox) != 4:
                    print(f"Warning: Invalid COCO format in {annotation_file}")
                    continue
                if not is_valid_bbox(bbox, width, height):
                    print(f"Warning: Invalid bounding box in {annotation_file}")


sanity_check("/media/lipade/Crucial X6/Trusti/training_set_slides/cores_with_lesions_0_jpg", "yolo")
