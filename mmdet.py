_base_ = [
    "/home/lipade/mmdetection/configs/rtmdet/rtmdet_s_8xb32-300e_coco.py",
]

experiment_id = "rtmdet_s_8xb32-300e"


classes = ("A", "B")
num_classes = len(classes)
metainfo = {"classes": classes, "palette": [(190, 77, 37), (37, 150, 190)]}

model = dict(
    data_preprocessor=dict(
        type="DetDataPreprocessor",
        mean=[213.308, 102.466, 48.287],
        std=[31.949, 31.084, 20.839],
        bgr_to_rgb=False,
        batch_augments=None,
        pad_size_divisor=32,
    ),
    bbox_head=dict(num_classes=num_classes),
)

image_scale = (512, 672)
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
    dict(type="JpegCompression", quality_lower=98, quality_upper=100, p=0.5),
]
packed_inputs_items = ("img_id", "img_path", "img_shape", "scale_factor", "ori_shape")
train_pipeline = [
    dict(type="LoadImageFromFile"),
    dict(type="LoadAnnotations", with_bbox=True),
    dict(type="Resize", scale=image_scale, keep_ratio=True),
    dict(
        type="Albu",
        transforms=albu_train_transforms,
        bbox_params=dict(
            type="BboxParams",
            format="pascal_voc",
            label_fields=["gt_bboxes_labels", "gt_ignore_flags"],
            min_visibility=0.0,
            filter_lost_elements=True,
        ),
        keymap={"img": "image", "gt_bboxes": "bboxes"},
        skip_img_without_anno=True,
    ),
    dict(type="RandomFlip", prob=0.5, direction="horizontal"),
    dict(type="PackDetInputs", meta_keys=packed_inputs_items),
]
val_pipeline = [
    dict(type="LoadImageFromFile"),
    dict(type="LoadAnnotations", with_bbox=True),
    dict(type="Resize", scale=image_scale, keep_ratio=True),
    dict(type="PackDetInputs", meta_keys=packed_inputs_items),
]
test_pipeline = val_pipeline

# dataset settings
dataset_type = "CocoDataset"
data_root = "/data/xxx"
train_annotations = "annotations/xxx.json"
val_annotations = "annotations/xxx.json"
test_annotations = "annotations/xxx.json"
image_data_prefix = data_root + "images/"
backend_args = None

workers_configs = {
    "num_workers": 2,
    "persistent_workers": True,
}
dataset_configs = {
    "type": dataset_type,
    "metainfo": metainfo,
    "data_root": data_root,
    "data_prefix": dict(img=image_data_prefix),
    "backend_args": backend_args,
}

# dataloader settings
train_dataloader = dict(
    batch_size=batch_size,
    sampler=dict(type="DefaultSampler", shuffle=True),
    **workers_configs,
    dataset=dict(
        ann_file=train_annotations, filter_cfg=dict(filter_empty_gt=True, min_size=32), pipeline=train_pipeline, **dataset_configs
    ),
)
val_dataloader = dict(
    batch_size=1,
    drop_last=False,
    sampler=dict(type="DefaultSampler", shuffle=False),
    **workers_configs,
    dataset=dict(ann_file=val_annotations, pipeline=val_pipeline, test_mode=True, **dataset_configs),
)
test_dataloader = dict(
    batch_size=1,
    drop_last=False,
    sampler=dict(type="DefaultSampler", shuffle=False),
    **workers_configs,
    dataset=dict(ann_file=test_annotations, pipeline=test_pipeline, test_mode=True, **dataset_configs),
)

# evaluator settings
val_evaluator = dict(type="CocoMetric", ann_file=data_root + val_annotations, metric="bbox", format_only=False, backend_args=backend_args)
test_evaluator = dict(type="CocoMetric", ann_file=data_root + test_annotations, metric="bbox", format_only=False, backend_args=backend_args)

# optimizer settings
optimizer = dict(type="AdamW", lr=eta, weight_decay=0.0001)
optim_wrapper = dict(
    type="OptimWrapper",
    optimizer=optimizer,
)

# workflow and workdir settings
train_cfg = dict(type="EpochBasedTrainLoop", max_epochs=n_epochs, val_interval=1)
val_cfg = dict(type="ValLoop")
test_cfg = dict(type="TestLoop")

# checkpoint: save the best model
default_hooks = dict(checkpoint=dict(type="CheckpointHook", interval=-1, save_best="auto"), logger=dict(type="LoggerHook", interval=50))

workflow = [("train", 1), ("val", 1)]

work_dir = f"logs/{experiment_id}"

# logger settings
vis_backends = [
    dict(type="LocalVisBackend"),
    dict(type="TensorboardVisBackend"),
]
visualizer = dict(type="DetLocalVisualizer", vis_backends=vis_backends, name="visualizer")
