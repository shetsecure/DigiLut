import os
from abc import ABC, abstractmethod
from typing import List, Any, Union, Dict
from pathlib import Path

import torch
import numpy as np

from config import Segmentationconfig, Coreconfig
from tissue_segmenter import TissueSegmenter
from wsi import WholeSlideImage
from core_extractor import CoreExtractor, Core
from core_manager import CoreAnnotationManager

from tqdm import tqdm

from ultralytics import YOLO


def extract_wsi_cores(seg_config: Segmentationconfig, wsi: WholeSlideImage, extraction_level: int):
    # Segment tissue in WSI
    segmenter = TissueSegmenter(seg_config)
    tissue_segments = segmenter.segment(wsi)

    # Extract cores
    core_extractor = CoreExtractor(wsi)
    cores = core_extractor.extract_cores(tissue_segments, extraction_level=extraction_level)

    return cores


class InferenceInterface(ABC):
    """
    Any model must implement this interfence to infer a slide.

    To save time, the predict method must take a directory path, in which we can find the extracted cores of a given WSI.

    post_process is to post_process a given prediction.

    At the end we're going to call predict_slide_dir to get the all the predictions of the slides
    """

    @abstractmethod
    def load_model(self, model_path: Path):
        pass

    @abstractmethod
    def preprocess_core(self, image: np.ndarray) -> Any:
        pass

    @abstractmethod
    def postprocess_core_prediction(self, prediction: Any) -> Any:
        pass

    @abstractmethod
    def predict_slide_dir(self, slide_dir_path: Union[str, Path]) -> List:
        pass


class YOLOInference(InferenceInterface):
    def __init__(self, model_path, model_args) -> None:
        self.all_predictions = []
        self.model: YOLO = self.load_model(model_path)

        assert "batch" in model_args
        assert "augment" in model_args
        assert "conf" in model_args

        self.model_args = model_args
        # yolo_model.predict(out_dir, batch=batch_size, verbose=False, augment=tta, conf=0.05, device="cuda")
        if not ("device" in self.model_args):
            self.model_args["device"] = "cuda"

    def postprocess_core_prediction(self, prediction: Any) -> Any:
        pass

    def preprocess_core(self, image: np.ndarray) -> Any:
        pass

    def load_model(self, model_path: Path):
        print(f"Loading YOLO model from {model_path}")
        model = YOLO(model_path, task="detect")
        print(model.info())

        return model

    def predict_slide_dir(
        self,
        wsi: WholeSlideImage,
        seg_config: Segmentationconfig,
        core_config: Coreconfig,
        extraction_level: int,
        slide_extracted_cores_path=None,
    ) -> Dict[Core, List]:

        # Extracting the cores and saving them to disk if they have never been extracted
        if not slide_extracted_cores_path:
            slide_extracted_cores_path = f"inference_extracted_cores/{wsi.name}_extracted_cores"

        slide_extracted_cores_path = Path(slide_extracted_cores_path)
        slide_extracted_cores_path = slide_extracted_cores_path / f"{wsi.name}_extracted_cores"

        n_files = sum(os.path.isfile(os.path.join(slide_extracted_cores_path, f)) for f in os.listdir(slide_extracted_cores_path))

        yolo_save_dir = slide_extracted_cores_path / "yolo_preds"
        yolo_save_dir.mkdir(parents=True, exist_ok=True)

        cores = extract_wsi_cores(wsi=wsi, seg_config=seg_config, extraction_level=extraction_level)
        cores_manager = CoreAnnotationManager(output_dir=slide_extracted_cores_path, core_config=core_config)

        if n_files == len(cores):
            print(f"Skipping saving cores to disk for {wsi.name}. There is {n_files} files in directory and {len(cores)} extracted boxes")
        else:
            cores_manager.save_cores(wsi, cores, with_annotations=False)

        # Now running inference with YOLO on the directory that contains the extracted cores

        results = self.model.predict(slide_extracted_cores_path, verbose=False, **self.model_args)

        count = 0

        cores_with_lesions = dict()

        for result in results:
            boxes = result.boxes
            if len(boxes) > 0:
                core_path = Path(result.path)
                save_to = yolo_save_dir / core_path.name

                # print(f"Predicted {len(boxes)} boxes for {core_path.name}")
                core = cores_manager.parse_core_from_path(core_path=core_path)

                # Adding the confs cuz we need it
                corresponding_boxes = boxes.xyxy.cpu().numpy()
                confs_for_each_box = boxes.conf.cpu().numpy()
                boxes_with_confs = np.concatenate((corresponding_boxes, confs_for_each_box.reshape(len(confs_for_each_box), 1)), axis=1)
                cores_with_lesions[core] = boxes_with_confs

                count += len(boxes)

                result.save(filename=save_to)

        print(f"Predicted {count} lesions for this slide {wsi.name}")

        return cores_with_lesions


import sys
import cv2
import glob
import importlib
from PIL import Image
from damo.base_models.core.ops import RepConv
from damo.detectors.detector import build_local_model
from damo.utils import vis
from damo.utils.demo_utils import transform_img
from damo.structures.image_list import ImageList, to_image_list


def get_config_by_file(config_file):
    try:
        sys.path.append(os.path.dirname(config_file))
        current_config = importlib.import_module(os.path.basename(config_file).split(".")[0])
        exp = current_config.Config()
    except Exception:
        raise ImportError("{} doesn't contains class named 'Config'".format(config_file))
    return exp


def parse_config(config_file):
    """
    get config object by file.
    Args:
        config_file (str): file path of config.
    """
    assert config_file is not None, "plz provide config file"
    if config_file is not None:
        return get_config_by_file(config_file)


class DamoYOLOInference(InferenceInterface):

    def __init__(
        self,
        config_file_path,
        model_args,
        save_result=False,
        infer_size=[640, 640],
        output_dir="./",
        ckpt=None,
    ):
        assert "batch_size" in model_args
        assert "conf" in model_args

        self.model_args = model_args
        self.model_args["save_result"] = save_result
        if not ("device" in self.model_args):
            self.model_args["device"] = "cuda"

        self.device = self.model_args["device"]

        config = get_config_by_file(config_file_path)

        self.ckpt_path = ckpt
        self.output_dir = output_dir

        if "class_names" in config.dataset:
            self.class_names = config.dataset.class_names
        else:
            self.class_names = []
            for i in range(config.model.head.num_classes):
                self.class_names.append(str(i))
            self.class_names = tuple(self.class_names)

        self.infer_size = infer_size
        config.dataset.size_divisibility = 0
        self.config = config
        self.model = self._build_engine(self.config, "torch")

    def _pad_image(self, imgs, target_size):
        if isinstance(target_size, (list, tuple)) and len(target_size) == 2:
            max_size = (3, target_size[0], target_size[1])  # Add channel dimension
        else:
            raise ValueError("target_size should be a list or tuple of length 2")

        return to_image_list(imgs, size_divisible=32, max_size=max_size)

    def _build_engine(self, config, engine_type):

        print(f"Inference with {engine_type} engine!")
        if engine_type == "torch":
            model = build_local_model(config, self.device)
            ckpt = torch.load(self.ckpt_path, map_location=self.device)
            model.load_state_dict(ckpt["model"], strict=True)
            for layer in model.modules():
                if isinstance(layer, RepConv):
                    layer.switch_to_deploy()
            model.eval()
        else:
            NotImplementedError(f"{engine_type} is not supported yet! Please use one of [onnx, torch, tensorRT]")

        return model

    def preprocess(self, origin_imgs):

        processed_imgs = []
        original_sizes = []

        for origin_img in origin_imgs:
            origin_img = np.asarray(origin_img)

            img = transform_img(origin_img, 0, **self.config.test.augment.transform, infer_size=self.infer_size)
            oh, ow, _ = origin_img.shape
            processed_imgs.append(img.tensors.squeeze(0))
            original_sizes.append((ow, oh))

        # Pad images to the same size
        imgs = self._pad_image(processed_imgs, self.infer_size)
        imgs = imgs.to(self.device)

        return imgs, original_sizes

    def postprocess(self, preds, image, origin_shapes=None):
        batch_bboxes, batch_scores, batch_cls_inds = [], [], []

        for output, origin_shape in zip(preds, origin_shapes):
            output = output.resize(origin_shape)
            bboxes = output.bbox
            scores = output.get_field("scores")
            cls_inds = output.get_field("labels")

            batch_bboxes.append(bboxes)
            batch_scores.append(scores)
            batch_cls_inds.append(cls_inds)

        return batch_bboxes, batch_scores, batch_cls_inds

    def forward(self, origin_images):
        images, origin_shapes = self.preprocess(origin_images)

        outputs = self.model(images.tensors)

        batch_bboxes, batch_scores, batch_cls_inds = self.postprocess(outputs, images, origin_shapes)

        batch_results = list(zip(batch_bboxes, batch_scores, batch_cls_inds))

        return batch_results

    def load_model(self, model_path: Path):
        pass

    def preprocess_core(self, image: np.ndarray) -> Any:
        pass

    def postprocess_core_prediction(self, prediction: Any) -> Any:
        pass

    def visualize(self, image, bboxes, scores, cls_inds, conf, save_name="vis.jpg", save_result=True):
        vis_img = vis(image, bboxes, scores, cls_inds, conf, self.class_names)
        if save_result:
            save_path = os.path.join(self.output_dir, save_name)
            cv2.imwrite(save_path, vis_img[:, :, ::-1])
        return vis_img

    def predict_slide_dir(
        self,
        wsi: WholeSlideImage,
        seg_config: Segmentationconfig,
        core_config: Coreconfig,
        extraction_level: int,
        slide_extracted_cores_path=None,
    ) -> List:

        # Extracting the cores and saving them to disk if they have never been extracted
        if not slide_extracted_cores_path:
            slide_extracted_cores_path = f"inference_extracted_cores/{wsi.name}_extracted_cores"
        else:
            slide_extracted_cores_path = Path(slide_extracted_cores_path)
            slide_extracted_cores_path = slide_extracted_cores_path / f"{wsi.name}_extracted_cores"

        print(slide_extracted_cores_path)

        yolo_save_dir = slide_extracted_cores_path / "damo_yolo_preds"
        yolo_save_dir.mkdir(parents=True, exist_ok=True)
        self.output_dir = yolo_save_dir

        n_files = sum(os.path.isfile(os.path.join(slide_extracted_cores_path, f)) for f in os.listdir(slide_extracted_cores_path))

        cores = extract_wsi_cores(wsi=wsi, seg_config=seg_config, extraction_level=extraction_level)
        cores_manager = CoreAnnotationManager(output_dir=slide_extracted_cores_path, core_config=core_config)

        if n_files == len(cores):
            print(f"Skipping saving cores to disk for {wsi.name}. There is {n_files} files in directory and {len(cores)} extracted boxes")
        else:
            cores_manager.save_cores(wsi, cores, with_annotations=False)

        # Now running inference with YOLO on the directory that contains the extracted cores

        cores_path = glob.glob(f"{slide_extracted_cores_path}/*.{cores_manager.core_format}")
        cores_with_lesions = dict()
        count = 0

        batch_size = self.model_args["batch_size"]

        for i in tqdm(range(0, len(cores_path), batch_size), desc=f"Predicting cores {wsi.name}"):
            batch_paths = cores_path[i : i + batch_size]
            batch_images = [Image.open(path).convert("RGB") for path in batch_paths]

            with torch.no_grad():
                batch_results = self.forward(batch_images)

            for core_path, result in zip(batch_paths, batch_results):
                bboxes, scores, cls_inds = result
                bboxes, scores = bboxes.cpu().numpy(), scores.cpu().numpy()
                indices = np.where(scores > self.model_args["conf"])
                bboxes, scores = bboxes[indices], scores[indices]

                if len(bboxes) > 0:
                    core = cores_manager.parse_core_from_path(core_path=core_path)
                    boxes_with_confs = np.concatenate((bboxes, scores.reshape(len(scores), 1)), axis=1)
                    cores_with_lesions[core] = boxes_with_confs
                    count += len(bboxes)

                    self.visualize(
                        np.asarray(Image.open(core_path).convert("RGB")),
                        bboxes,
                        scores,
                        cls_inds,
                        conf=self.model_args["conf"],
                        save_name=Path(core_path).name,
                        save_result=self.model_args["save_result"],
                    )

        print(f"Predicted {count} lesions for this slide {wsi.name}")
        return cores_with_lesions
