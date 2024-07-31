import os

from pathlib import Path
from tqdm import tqdm
from wsi import WholeSlideImage
from ultralytics import YOLO
from typing import NamedTuple, List, Dict

import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import matplotlib.patches as patches

yolo_weight_path = "/home/lipade/Challenge/code/weights/yolov10s_b64_i1024_e600_aug/train/weights/best.pt"
print(f"Loading YOLO model from {yolo_weight_path}")
model = YOLO(yolo_weight_path, task="detect")
print(model.info())
# model.eval()
# model.val()

"""
Results:
1) 0.21747516383401097 by: train18 
batch: 64, imgsz: 1024

Ultralytics YOLOv8.2.48 🚀 Python-3.9.16 torch-2.3.1+cu118 CUDA:0 (NVIDIA A100-SXM4-80GB, 81038MiB)
YOLOv10s summary (fused): 293 layers, 8035734 parameters, 0 gradients, 24.4 GFLOPs
                Class     Images  Instances      Box(P          R      mAP50  mAP50-95)
                   all        282        175      0.649      0.383      0.463      0.159

2) 0.21291676027319084 by: yolov10s_b64_i1024_e600_aug/train/weights/best.pt
batch: 64, imgsz: 1024

Validating runs/yolov10s_b64_i1024_e600_aug/train/weights/best.pt...
WARNING ⚠️ End2End model does not support 'augment=True' prediction. Reverting to single-scale prediction.

Ultralytics YOLOv8.2.48 🚀 Python-3.9.16 torch-2.3.1+cu118 CUDA:0 (NVIDIA A100-SXM4-80GB, 81038MiB)
YOLOv10s summary (fused): 293 layers, 8035734 parameters, 0 gradients, 24.4 GFLOPs
                Class     Images  Instances      Box(P          R      mAP50  mAP50-95)
                   all        282        175       0.49      0.474      0.441      0.175

3) 0.19624658939144907 by: yolov10s_b64_i1024_e600/train/weights/best.pt
batch: 64, imgsz: 1024

Validating runs/yolov10s_b64_i1024_e600/train/weights/best.pt...
Ultralytics YOLOv8.2.48 🚀 Python-3.9.16 torch-2.3.1+cu118 CUDA:0 (NVIDIA A100-SXM4-80GB, 81038MiB)
YOLOv10s summary (fused): 293 layers, 8035734 parameters, 0 gradients, 24.4 GFLOPs
                Class     Images  Instances      Box(P          R      mAP50  mAP50-95)
                   all        282        175       0.58      0.434      0.445      0.163


4) 0.18638224630985167 by: train20

batch: 384, imgsz: 1024
Ultralytics YOLOv8.2.48 🚀 Python-3.9.16 torch-2.3.1+cu118 CUDA:0 (NVIDIA A100-SXM4-80GB, 81038MiB)
                                                           CUDA:1 (NVIDIA A100-SXM4-80GB, 81038MiB)
                                                           CUDA:2 (NVIDIA A100-SXM4-80GB, 81038MiB)
                                                           CUDA:3 (NVIDIA A100-SXM4-80GB, 81038MiB)
YOLOv10s summary (fused): 293 layers, 8035734 parameters, 0 gradients, 24.4 GFLOPs
                Class     Images  Instances      Box(P          R      mAP50  mAP50-95)
                   all        282        175      0.686      0.474      0.496      0.174

5) 0.17254910486183486 by: train19

batch: 96, imgsz: 1024
Ultralytics YOLOv8.2.48 🚀 Python-3.9.16 torch-2.3.1+cu118 CUDA:0 (NVIDIA A100-SXM4-80GB, 81038MiB)
YOLOv10s summary (fused): 293 layers, 8035734 parameters, 0 gradients, 24.4 GFLOPs
                Class     Images  Instances      Box(P          R      mAP50  mAP50-95)
                   all        282        175      0.625      0.451       0.46      0.151
Speed: 0.2ms preprocess, 3.1ms inference, 0.0ms loss, 0.1ms postprocess per image
Results saved to runs/detect/train19


6) 0.17239092376091603 by: train26

batch: 100, imgsz: 2048
Ultralytics YOLOv8.2.48 🚀 Python-3.9.16 torch-2.3.1+cu118 CUDA:0 (NVIDIA A100-SXM4-80GB, 81038MiB)
                                                           CUDA:1 (NVIDIA A100-SXM4-80GB, 81038MiB)
                                                           CUDA:2 (NVIDIA A100-SXM4-80GB, 81038MiB)
                                                           CUDA:3 (NVIDIA A100-SXM4-80GB, 81038MiB)
                                                           CUDA:4 (NVIDIA A100-SXM4-80GB, 81038MiB)
YOLOv10s summary (fused): 293 layers, 8035734 parameters, 0 gradients, 24.4 GFLOPs
                Class     Images  Instances      Box(P          R      mAP50  mAP50-95)
                   all        282        175      0.512       0.32      0.291        0.1

7) 0.15734953840707855 by train28

batch: 16, imgsz: 1024

Ultralytics YOLOv8.2.48 🚀 Python-3.9.16 torch-2.3.1+cu118 CUDA:0 (NVIDIA A100-SXM4-80GB, 81038MiB)
YOLOv10s summary (fused): 293 layers, 8035734 parameters, 0 gradients, 24.4 GFLOPs
                Class     Images  Instances      Box(P          R      mAP50  mAP50-95)
                   all        282        175      0.484      0.349      0.338      0.124

8) 0.11064068004844395 by: train27

batch: 32, imgsz: 1024

Ultralytics YOLOv8.2.48 🚀 Python-3.9.16 torch-2.3.1+cu118 CUDA:0 (NVIDIA A100-SXM4-80GB, 81038MiB)
YOLOv10s summary (fused): 293 layers, 8035734 parameters, 0 gradients, 24.4 GFLOPs
                Class     Images  Instances      Box(P          R      mAP50  mAP50-95)
                   all        282        175      0.529      0.411      0.383      0.147


"""


class Core(NamedTuple):
    """
    Core: x, y are always in Level 0, level is used only for width and height.
    """

    x: int
    y: int
    level: int
    width: int
    height: int


def parse_core_from_path(core_name, wsi_name):
    path = str(core_name)
    path = path.replace(".png", "")
    path = path.replace(wsi_name + "_", "")
    path = path.replace("L", "")
    x, y, l, w, h = map(int, path.split("_"))

    return Core(x, y, l, w, h)


def adjust_box_coords_from_core_to_level(core: Core, box, slide, thumbnail_level):
    """
    Adjust the bounding box coordinates from the core level to the thumbnail level.
    Core: (x_core, y_core, extract_level, w_c, h_c)
    Box: (x, y, width, height) in the core coordinates
    """

    x_core, y_core, extract_level, w_c, h_c = core

    # Scaling factors
    scale_to_thumbnail = 1 / slide.level_downsamples[thumbnail_level]
    scale_to_extract_level = slide.level_downsamples[extract_level]

    # Adjust box coordinates
    x, y, xx, yy = box
    width = xx - x
    height = yy - y

    # Map box coordinates from core level to Level 0
    x = x_core + x * scale_to_extract_level
    y = y_core + y * scale_to_extract_level
    width = width * scale_to_extract_level
    height = height * scale_to_extract_level

    # Map from Level 0 to thumbnail level
    x = int(x * scale_to_thumbnail)
    y = int(y * scale_to_thumbnail)
    width = int(width * scale_to_thumbnail)
    height = int(height * scale_to_thumbnail)

    return x, y, width, height


def visualize_extracted_bboxes_with_annotations(slide, thumbnail_level, cores_with_lesions: Dict[Core, np.ndarray], save_to):
    thumbnail_scale = 1 / slide.level_downsamples[thumbnail_level]

    thumbnail = slide.read_region((0, 0), thumbnail_level, slide.level_dimensions[thumbnail_level]).convert("RGB")
    thumbnail_np = np.asarray(thumbnail)

    fig, ax = plt.subplots(1, figsize=(12, 12))
    ax.imshow(thumbnail_np)

    def adjust_core_coords_to_level(core: Core):
        x_core, y_core, extract_level, w_c, h_c = list(core)
        x_core, y_core = int(x_core * thumbnail_scale), int(y_core * thumbnail_scale)

        # map from extraction level to thumbnail level
        # map to 0
        w_c *= slide.level_downsamples[extract_level]
        h_c *= slide.level_downsamples[extract_level]

        # map back to thumbnail level
        w_c = int(w_c * thumbnail_scale)
        h_c = int(h_c * thumbnail_scale)

        return Core(x_core, y_core, extract_level, w_c, h_c)

    for idx, core in enumerate(cores_with_lesions):
        relative_core_to_thumbnail = adjust_core_coords_to_level(core)
        x_core, y_core, extract_level, w_c, h_c = list(relative_core_to_thumbnail)

        rect = patches.Rectangle((x_core, y_core), w_c, h_c, linewidth=0.5, edgecolor="black", facecolor="none")
        ax.add_patch(rect)

        center_x = x_core + w_c / 2
        center_y = y_core + h_c / 2
        ax.text(center_x, center_y, str(idx), color="black", fontsize=6, ha="center", va="center")

        boxes = cores_with_lesions[core]

        for box in boxes:
            x, y, width, height = adjust_box_coords_from_core_to_level(core, box[:4], slide, thumbnail_level)

            rect = patches.Rectangle((x, y), width, height, linewidth=0.5, edgecolor="red", facecolor="none")
            ax.add_patch(rect)

    plt.axis("off")
    plt.savefig(save_to, dpi=300)
    # plt.show()


def box_to_0(wsi: WholeSlideImage, core: Core, box):

    x_core, y_core, extract_level, w_c, h_c = core
    extract_level_to_0 = wsi.wsi.level_downsamples[extract_level]

    # Adjust box coordinates
    x, y, xx, yy = box
    width = xx - x
    height = yy - y

    # Map box coordinates from core level to Level 0
    x = x_core + x * extract_level_to_0
    y = y_core + y * extract_level_to_0
    width = width * extract_level_to_0
    height = height * extract_level_to_0

    return int(x), int(y), round(width), round(height)


def extract_and_save_lesions(wsi: WholeSlideImage, cores_with_lesions: Dict[Core, np.ndarray], save_to):

    save_to = Path(save_to)
    save_to.mkdir(parents=True, exist_ok=True)
    all_lesions_coords_xyxyc = []  # xyxyc

    for core, boxes in cores_with_lesions.items():
        for i, box in enumerate(tqdm(boxes, desc="Saving the lesions from level 0")):
            x, y, w, h = box_to_0(wsi, core, box[:4])
            all_lesions_coords_xyxyc.append((x, y, x + w, y + h, box[-1]))
            box_region = wsi.wsi.read_region((x, y), 0, (w, h)).convert("RGB")
            box_name = f"{wsi.name}_{x}_{y}_L{extraction_level}_{w}_{h}_box{i}.png"
            box_region.save(save_to / box_name)

    sorted_boxes = sorted(all_lesions_coords_xyxyc, key=lambda x: x[-1], reverse=True)
    sorted_boxes = [sublist[:-1] for sublist in sorted_boxes]

    d = dict()
    d[wsi.name + ".tif"] = sorted_boxes


def get_sorted_boxes_from_slide(wsi: WholeSlideImage, cores_with_lesions: Dict[Core, np.ndarray]):
    all_lesions_coords_xyxyc = []  # xyxyc

    for core, boxes in cores_with_lesions.items():
        for box in boxes:
            x, y, w, h = box_to_0(wsi, core, box[:4])
            all_lesions_coords_xyxyc.append((x, y, x + w, y + h, box[-1]))

    sorted_boxes = sorted(all_lesions_coords_xyxyc, key=lambda x: x[-1], reverse=True)
    sorted_boxes = [sublist[:-1] for sublist in sorted_boxes]

    return sorted_boxes


def detect_wsi(
    yolo_model: YOLO, wsi_path, extraction_level, patch_size, batch_size, save_cores_dir=None, thumbnail_level=6, visualize=False, tta=False
):

    wsi = WholeSlideImage(wsi_path)

    if not save_cores_dir:
        save_cores_dir = f"inference_extracted_cores/{wsi.name}_extracted_cores"
    out_dir = Path(save_cores_dir)
    out_dir.mkdir(parents=True, exist_ok=True)

    n_files = sum(os.path.isfile(os.path.join(out_dir, f)) for f in os.listdir(out_dir))

    yolo_save_dir = out_dir / "yolo_preds"
    yolo_save_dir.mkdir(parents=True, exist_ok=True)

    print(f"Segmenting the cores in the WSI {wsi.name}")
    filters = {"a_t": 2, "a_h": 10, "max_n_holes": 9}
    wsi.segmentTissue(5, filter_params=filters)
    wsi.filterThinContours(min_aspect_ratio=0.4, min_area=5e6)
    cores_bboxes = wsi.extract_tissue_tiles_coords(extraction_level)

    if n_files >= len(cores_bboxes):
        print(
            f"Skipping saving cores to disk for {wsi.name}. There is {n_files} files in directory and {len(cores_bboxes)} extracted boxes"
        )
    else:
        for core in tqdm(cores_bboxes, desc="Extracting the WSI cores"):
            x, y, w, h = core
            core_region = wsi.wsi.read_region((x, y), extraction_level, (w, h)).convert("RGB")
            core_region.resize(patch_size)
            core_name = f"{wsi.name}_{x}_{y}_L{extraction_level}_{w}_{h}.png"
            core_region.save(out_dir / core_name)

    results = yolo_model.predict(out_dir, batch=batch_size, verbose=False, augment=tta, conf=0.05, device="cuda")
    count = 0

    cores_with_lesions = dict()

    for result in results:
        boxes = result.boxes
        if len(boxes) > 0:
            core_path = Path(result.path)
            save_to = yolo_save_dir / core_path.name

            # print(f"Predicted {len(boxes)} boxes for {core_path.name}")
            core = parse_core_from_path(core_path.name, wsi.name)
            # BIG FUCKING BUG
            # cores_with_lesions[core] = boxes.xywh.cpu().numpy()

            # Adding the confs cuz we need it
            corresponding_boxes = boxes.xyxy.cpu().numpy()
            confs_for_each_box = boxes.conf.cpu().numpy()
            boxes_with_confs = np.concatenate((corresponding_boxes, confs_for_each_box.reshape(len(confs_for_each_box), 1)), axis=1)
            cores_with_lesions[core] = boxes_with_confs

            count += len(boxes)

            result.save(filename=save_to)

    print(f"Predicted {count} lesions for this slide {wsi.name}")

    # Grab everything and plot it to thumbnail

    if visualize:
        visualize_extracted_bboxes_with_annotations(wsi.wsi, thumbnail_level, cores_with_lesions, yolo_save_dir / "thumbnail_preds.png")
    # extract_and_save_lesions(wsi, cores_with_lesions, yolo_save_dir)

    wsi.wsi.close()

    return cores_with_lesions


def prepare_predictions(wsi: WholeSlideImage, cores_with_lesions: Dict[Core, np.ndarray], number_of_required_boxes):
    boxes = get_sorted_boxes_from_slide(wsi, cores_with_lesions)

    if len(boxes) > number_of_required_boxes:
        return boxes[:number_of_required_boxes]
    else:
        # for now we just gonna duplicate it
        if len(boxes) > 0:
            box = boxes[0]
        else:
            box = [0, 0, 0, 0]

        for _ in range(number_of_required_boxes - len(boxes)):
            boxes.append(box)

        return boxes


def predict_sample_submission(slides_path, csv_path, model, extraction_level, patch_size, batch_size, tta=False):
    df = pd.read_csv(csv_path, sep=",")

    required_boxes_per_slide = df["filename"].value_counts().to_dict()
    predicted_boxes_per_slide = dict()

    for slide_name, n_required_boxes in tqdm(required_boxes_per_slide.items(), desc=f"Predicting slides"):
        slide_path = Path(slides_path) / slide_name
        wsi = WholeSlideImage(slide_path)

        cores_with_lesions = detect_wsi(model, slide_path, extraction_level, patch_size, batch_size, tta=tta)
        total_boxes_predicted = sum(len(l) for l in cores_with_lesions.values())

        predicted_boxes = prepare_predictions(wsi, cores_with_lesions, n_required_boxes)

        print(f"Original number of predicted boxes is {total_boxes_predicted}")
        print(f"Made them {len(predicted_boxes)} boxes to match the submission file")

        predicted_boxes_per_slide[wsi.name + ".tif"] = predicted_boxes

    for i in range(len(df)):
        slide_name = df.iloc[i]["filename"]
        x, y, xx, yy = predicted_boxes_per_slide[slide_name].pop(0)
        df.at[i, "x1"] = x
        df.at[i, "y1"] = y
        df.at[i, "x2"] = xx
        df.at[i, "y2"] = yy

    df.to_csv("yolov10s_b64_i1024_e600_aug.csv", sep=",", index=None)


wsi_path = "/home/lipade/Challenge/zefEncNeVL_b.tif"
# wsi_path = "/home/lipade/Challenge/xIcIp7GzFh_a.tif"

extraction_level = 2
patch_size = (1024, 1024)

# results = detect_wsi(model, wsi_path, extraction_level, patch_size, 4)

predict_sample_submission(
    "/home/lipade/Challenge/val",
    csv_path="submission_sample.csv",
    model=model,
    extraction_level=extraction_level,
    patch_size=patch_size,
    batch_size=16,
    tta=False,
)
