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


import cv2
import glob
import argparse
from PIL import Image
from damo.base_models.core.ops import RepConv
from damo.config.base import parse_config
from damo.detectors.detector import build_local_model
from damo.utils import get_model_info, vis, postprocess
from damo.utils.demo_utils import transform_img
from damo.structures.image_list import ImageList
from damo.structures.bounding_box import BoxList


class DamoYOLOInference(InferenceInterface):

    @staticmethod
    def make_parser():
        parser = argparse.ArgumentParser("DAMO-YOLO Demo")

        # parser.add_argument("input_type", default="image", help="input type, support [image, video, camera]")
        parser.add_argument(
            "-f",
            "--config_file",
            default=None,
            type=str,
            help="pls input your config file",
        )
        parser.add_argument("-p", "--path", default="./assets/dog.jpg", type=str, help="path to image or video")
        parser.add_argument("--camid", type=int, default=0, help="camera id, necessary when input_type is camera")
        parser.add_argument("--engine", default=None, type=str, help="engine for inference")
        parser.add_argument("--device", default="cuda", type=str, help="device used to inference")
        parser.add_argument("--output_dir", default="./demo", type=str, help="where to save inference results")
        parser.add_argument("--conf", default=0.3, type=float, help="conf of visualization")
        parser.add_argument("--infer_size", nargs="+", type=int, help="test img size")
        parser.add_argument("--end2end", action="store_true", help="trt engine with nms")
        parser.add_argument("--save_result", default=True, type=bool, help="whether save visualization results")

        return parser

    def __init__(self, config_file_path, infer_size=[640, 640], device="cuda", output_dir="./", ckpt=None, end2end=False):
        args = self.make_parser().parse_args()
        self.args = args
        # self.args.input_type = "image"

        config = parse_config(config_file_path)

        self.ckpt_path = ckpt
        suffix = ckpt.split(".")[-1]
        if suffix == "onnx":
            self.engine_type = "onnx"
        elif suffix == "trt":
            self.engine_type = "tensorRT"
        elif suffix in ["pt", "pth"]:
            self.engine_type = "torch"
        self.end2end = end2end  # only work with tensorRT engine
        self.output_dir = output_dir

        if torch.cuda.is_available() and device == "cuda":
            self.device = "cuda"
        else:
            self.device = "cpu"

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
        self.model = self._build_engine(self.config, self.engine_type)

    def _pad_image(self, img, target_size):
        n, c, h, w = img.shape
        assert n == 1
        assert h <= target_size[0] and w <= target_size[1]
        target_size = [n, c, target_size[0], target_size[1]]
        pad_imgs = torch.zeros(*target_size)
        pad_imgs[:, :c, :h, :w].copy_(img)

        img_sizes = [img.shape[-2:]]
        pad_sizes = [pad_imgs.shape[-2:]]

        return ImageList(pad_imgs, img_sizes, pad_sizes)

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

    def preprocess(self, origin_img):

        img = transform_img(origin_img, 0, **self.config.test.augment.transform, infer_size=self.infer_size)
        # img is a image_list
        oh, ow, _ = origin_img.shape
        img = self._pad_image(img.tensors, self.infer_size)

        img = img.to(self.device)
        return img, (ow, oh)

    def postprocess(self, preds, image, origin_shape=None):

        if self.engine_type == "torch":
            output = preds

        output = output[0].resize(origin_shape)
        bboxes = output.bbox
        scores = output.get_field("scores")
        cls_inds = output.get_field("labels")

        return bboxes, scores, cls_inds

    def forward(self, origin_image):

        image, origin_shape = self.preprocess(origin_image)

        if self.engine_type == "torch":
            output = self.model(image)

        bboxes, scores, cls_inds = self.postprocess(output, image, origin_shape=origin_shape)

        return bboxes, scores, cls_inds

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

        # exit(1)

        # Now running inference with YOLO on the directory that contains the extracted cores

        cores_path = glob.glob(f"{slide_extracted_cores_path}/*.{cores_manager.core_format}")
        cores_with_lesions = dict()
        count = 0

        for core_path in tqdm(cores_path, desc=f"Predicting cores {wsi.name}"):
            origin_img = np.asarray(Image.open(core_path).convert("RGB"))
            with torch.no_grad():
                bboxes, scores, cls_inds = self.forward(origin_img)
                bboxes, scores = bboxes.cpu().numpy(), scores.cpu().numpy()

                indices = np.where(scores > self.args.conf)

                bboxes, scores = bboxes[indices], scores[indices]

            # print(scores)

            if len(bboxes) > 0:

                core = cores_manager.parse_core_from_path(core_path=core_path)
                boxes_with_confs = np.concatenate((bboxes, scores.reshape(len(scores), 1)), axis=1)

                cores_with_lesions[core] = boxes_with_confs
                count += len(bboxes)

                self.visualize(
                    origin_img,
                    bboxes,
                    scores,
                    cls_inds,
                    conf=self.args.conf,
                    save_name=Path(core_path).name,
                    save_result=self.args.save_result,
                )

        print(f"Predicted {count} lesions for this slide {wsi.name}")

        return cores_with_lesions
