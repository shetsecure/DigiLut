"""
import os
from PIL import Image
import numpy as np
from glob import glob
from tqdm import tqdm


def process_image(image_path):
    image = Image.open(image_path)
    image = np.array(image)
    mean = image.mean(axis=(0, 1))
    std = image.std(axis=(0, 1))
    return mean, std


def compute_mean_std(image_paths):
    means = []
    stds = []

    for image_path in tqdm(image_paths, desc="Processing images"):
        mean, std = process_image(image_path)
        means.append(mean)
        stds.append(std)

    mean = np.mean(means, axis=0)
    std = np.mean(stds, axis=0)
    return mean, std

# print(f"Computing the mean & std")
# image_paths = glob(f"{os.path.join(data_root, train_data_prefix)}/*.png")
# mean, std = compute_mean_std(image_paths)
# print(f"Mean: {mean}, Std: {std}")
# Mean: [234.7728757  210.40890574 223.23247727], Std: [24.12386117 43.14978828 30.36893309]
"""

"""
08/07 19:21:44 - mmengine - INFO - load backbone. in model from: https://download.openmmlab.com/mmdetection/v3.0/rtmdet/cspnext_rsb_pretrain/cspnext-s_imagenet_600e.pth
Loads checkpoint by http backend from path: https://download.openmmlab.com/mmdetection/v3.0/rtmdet/cspnext_rsb_pretrain/cspnext-s_imagenet_600e.pth
Downloading: "https://download.openmmlab.com/mmdetection/v3.0/rtmdet/cspnext_rsb_pretrain/cspnext-s_imagenet_600e.pth" to /home/lipade/.cache/torch/hub/checkpoints/cspnext-s_imagenet_600e.pth
100%|█████████████████████████████████████████████████████████████████████████████████████████████████████████████████████████████████████████████████████████████████████████| 56.2M/56.2M [00:04<00:00, 14.2MB/s]
Loads checkpoint by local backend from path: /home/lipade/mmdetection/rtmdet_tiny_8xb32-300e_coco_20220902_112414-78e30dcc.pth
The model and loaded state dict do not match exactly

"""

_base_ = ["/home/lipade/mmdetection/configs/rtmdet/rtmdet_s_8xb32-300e_coco.py"]

dataset_type = "CocoDataset"
backend_args = None
data_root = r"/media/lipade/Crucial X6/Trusti/good_dataset_with_soft_labels/"

train_annotations = r"annotations/train.json"
val_annotations = r"annotations/val.json"
train_data_prefix = "train/"
val_data_prefix = "val/"


experiment_id = "rtmdet_s_8xb32-300e"

classes = ("lesion",)
num_classes = len(classes)
metainfo = {"classes": classes, "palette": [(255, 0, 0)]}

model = dict(
    data_preprocessor=dict(
        type="DetDataPreprocessor",
        mean=[234.7728757, 210.40890574, 223.23247727],
        std=[24.12386117, 43.14978828, 30.36893309],
        bgr_to_rgb=False,
        batch_augments=None,
        pad_size_divisor=32,
    ),
    bbox_head=dict(num_classes=num_classes),
)

image_scale = (640, 640)
batch_size = 4
n_workers = 2
n_gpu = 1
base_lr = 0.004
eta = base_lr * (batch_size * n_gpu / 16) ** 0.5
n_epochs = 300

# pipelines setting
albu_train_transforms = [
    dict(type="ShiftScaleRotate", shift_limit=(-0.0625, 0.0625), scale_limit=(0.0, 0.0), rotate_limit=(-3.0, 3.0), interpolation=1, p=0.5),
    dict(type="OneOf", transforms=[dict(type="Blur", blur_limit=(3, 5), p=1.0), dict(type="MedianBlur", blur_limit=(3, 5), p=1.0)], p=0.5),
    dict(type="RandomBrightnessContrast", brightness_limit=[0.0, 0.05], contrast_limit=[0.0, 0.05], p=0.5),
    dict(
        type="OneOf",
        transforms=[
            dict(type="HueSaturationValue", hue_shift_limit=2, sat_shift_limit=2, val_shift_limit=2, p=1.0),
            dict(type="RGBShift", r_shift_limit=(-2, 2), g_shift_limit=(-2, 2), b_shift_limit=(-2, 2), p=1.0),
        ],
        p=0.5,
    ),
    # dict(type="JpegCompression", quality_lower=98, quality_upper=100, p=0.5),
]
packed_inputs_items = ("img_id", "img_path", "img_shape", "scale_factor", "ori_shape")
train_pipeline = [
    dict(type="LoadImageFromFile"),
    dict(type="LoadAnnotations", with_bbox=True),
    dict(type="Resize", scale=image_scale, keep_ratio=False),
    # dict(
    #     type="Albu",
    #     transforms=albu_train_transforms,
    #     bbox_params=dict(
    #         type="BboxParams",
    #         format="coco",
    #         label_fields=["gt_bboxes_labels", "gt_ignore_flags"],
    #         min_visibility=0.0,
    #         filter_lost_elements=True,
    #     ),
    # ),
    dict(type="Pad", size=image_scale, pad_val=dict(img=(114, 114, 114))),
    dict(type="PackDetInputs", meta_keys=packed_inputs_items),
]

val_pipeline = [
    dict(type="LoadImageFromFile"),
    dict(type="Resize", scale=image_scale, keep_ratio=False),
    dict(type="Pad", size=image_scale, pad_val=dict(img=(114, 114, 114))),
    dict(type="PackDetInputs", meta_keys=packed_inputs_items),
]


# workers_configs = dict(num_workers=n_workers, persistent_workers=True, drop_last=False, pin_memory=True)
workers_configs = dict(num_workers=n_workers, persistent_workers=True, pin_memory=True)

train_dataset_config = {
    "type": dataset_type,
    "metainfo": metainfo,
    "data_root": data_root,
    "data_prefix": dict(img=train_data_prefix),
    "backend_args": backend_args,
}

val_dataset_config = {
    "type": dataset_type,
    "metainfo": metainfo,
    "data_root": data_root,
    "data_prefix": dict(img=val_data_prefix),
    "backend_args": backend_args,
}

# dataloader settings
train_dataloader = dict(
    batch_size=batch_size,
    sampler=dict(type="DefaultSampler", shuffle=True),
    **workers_configs,
    dataset=dict(
        ann_file=train_annotations, filter_cfg=dict(filter_empty_gt=True, min_size=32), pipeline=train_pipeline, **train_dataset_config
    ),
)
val_dataloader = dict(
    batch_size=1,
    drop_last=False,
    sampler=dict(type="DefaultSampler", shuffle=False),
    **workers_configs,
    dataset=dict(ann_file=val_annotations, pipeline=val_pipeline, test_mode=True, **val_dataset_config),
)

# evaluator settings
val_evaluator = dict(type="CocoMetric", ann_file=data_root + val_annotations, metric="bbox", format_only=False, backend_args=backend_args)

# optimizer settings
optimizer = dict(type="AdamW", lr=eta, weight_decay=0.0001)
optim_wrapper = dict(
    type="OptimWrapper",
    optimizer=optimizer,
)

# workflow and workdir settings
train_cfg = dict(type="EpochBasedTrainLoop", max_epochs=n_epochs, val_interval=1)
val_cfg = dict(type="ValLoop")

# checkpoint: save the best model
default_hooks = dict(checkpoint=dict(type="CheckpointHook", interval=-1, save_best="auto"), logger=dict(type="LoggerHook", interval=50))

workflow = [("train", 1), ("val", 1)]

work_dir = f"rtmdet_s_8xb32_logs/{experiment_id}"

print("work dir ", work_dir)

# logger settings
vis_backends = [
    dict(type="LocalVisBackend"),
    dict(type="TensorboardVisBackend"),
]
visualizer = dict(type="DetLocalVisualizer", vis_backends=vis_backends, name="visualizer")

load_from = r"/home/lipade/mmdetection/rtmdet_s_8xb32-300e_coco_20220905_161602-387a891e.pth"
