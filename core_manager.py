import cv2
import numpy as np

from tqdm import tqdm
from pathlib import Path

from typing import List, Tuple, Optional, Union
from dataclasses import asdict
from enum import Enum, auto

import utils
from core_extractor import Core, BoundingBox
from wsi import WholeSlideImage


class AnnotationFormat(Enum):
    XYWH = auto()
    PASCAL_VOC = auto()
    COCO = auto()
    YOLO = auto()


class CoreAnnotationManager:
    def __init__(
        self, output_dir: Union[Path, str], annotation_format: AnnotationFormat = AnnotationFormat.YOLO, cores_format: str = "png"
    ):
        self.output_dir = Path(output_dir)
        self.annotation_format = annotation_format
        self.output_dir.mkdir(parents=True, exist_ok=True)
        self.cores_format = cores_format

    def save_cores(self, wsi_image: WholeSlideImage, cores: List[Core], resize: Optional[Union[int, Tuple[int, int]]] = None):
        for idx, core in enumerate(tqdm(cores)):
            core_image = self._extract_core_image(wsi_image, core)
            core_name = f"{wsi_image.name}_Core_{idx}_L{core.level}_{core.bbox.w}_{core.bbox.h}"

            if resize:
                resize = (resize, resize) if isinstance(resize, int) else resize
                core_image = core_image.resize(resize)
                core_name += f"_resized_{resize[0]}_{resize[1]}"

            core_image.save(self.output_dir / f"{core_name}.{self.cores_format}")

            if core.annotations:
                if resize:
                    self.resize_annotations(core=core, new_size=resize)

            # Save anyway and let the others like YOLO format handle it
            self._save_annotations(core, idx, core_name)

    @staticmethod
    def _extract_core_image(wsi_image: WholeSlideImage, core: Core):
        return wsi_image.wsi.read_region((core.bbox.x, core.bbox.y), core.level, (core.bbox.w, core.bbox.h)).convert("RGB")

    def _save_annotations(self, core: Core, core_idx: int, core_name: str):
        if self.annotation_format == AnnotationFormat.YOLO:
            self._save_yolo_annotations(core, core_name)
        else:
            raise ValueError(f"Unsupported annotation format: {self.annotation_format}")

    def _save_yolo_annotations(self, core: Core, core_name: str):
        yolo_annotations = [
            utils.convert_to_yolo_format((ann.x, ann.y, ann.w, ann.h), core.bbox.w, core.bbox.h) for ann in core.annotations
        ]

        with open(self.output_dir / f"{core_name}.txt", "w") as f:
            for annotation in yolo_annotations:
                f.write(" ".join(map(str, annotation)) + "\n")

    def resize_annotations(self, core: Core, new_size: Tuple[int, int]) -> Core:
        scale_x = new_size[0] / core.bbox.w
        scale_y = new_size[1] / core.bbox.h

        new_annotations = []
        for ann in core.annotations:
            new_ann = BoundingBox(int(ann.x * scale_x), int(ann.y * scale_y), int(ann.w * scale_x), int(ann.h * scale_y))
            new_annotations.append(new_ann)

        return Core(bbox=BoundingBox(core.bbox.x, core.bbox.y, new_size[0], new_size[1]), annotations=new_annotations, level=core.level)

    def visualize_core_with_annotations(self, core: Core, wsi_image) -> np.ndarray:
        core_image = self._extract_core_image(wsi_image, core)
        for ann in core.annotations:
            cv2.rectangle(core_image, (ann.x, ann.y), (ann.x + ann.w, ann.y + ann.h), (0, 255, 0), 2)
        return core_image
