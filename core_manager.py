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
from config import Coreconfig


class AnnotationFormat(Enum):
    XYWH = auto()
    PASCAL_VOC = auto()
    COCO = auto()
    YOLO = auto()


class CoreAnnotationManager:
    def __init__(
        self,
        output_dir: Union[Path, str],
        annotation_format: AnnotationFormat = AnnotationFormat.YOLO,
        core_config: Coreconfig = Coreconfig(),
    ):
        self.output_dir = Path(output_dir)
        self.annotation_format = annotation_format
        self.output_dir.mkdir(parents=True, exist_ok=True)
        self.core_config = core_config

        self.core_format = self.core_config.core_format.value
        self.resize = self.core_config.resize
        self.resize = (self.resize, self.resize) if isinstance(self.resize, int) else self.resize

    def get_core_naming_convention(self, wsi_name, core: Core) -> str:
        # How are we naming our cores so they can be saved as independant images
        # Only the name, the suffix is determined by self.core_format

        core_name = f"{wsi_name}_{core.bbox.x}_{core.bbox.y}_L{core.level}_{core.bbox.w}_{core.bbox.h}"

        if self.resize:
            core_name += f"_resized_{self.resize[0]}_{self.resize[1]}"

        return core_name

    @staticmethod
    def parse_core_from_path(core_path: Path) -> Core:
        core_path = Path(core_path)

        core_name = core_path.stem
        x, y, l, w, h = core_name.split("_")[-5:]
        l = l[1]  # Removing L
        x, y, l, w, h = map(int, [x, y, l, w, h])

        return Core(BoundingBox(x, y, w, h), annotations=[], level=l)

    def save_cores(self, wsi_image: WholeSlideImage, cores: List[Core], with_annotations=False):
        for core in tqdm(cores, desc=f"Extracting {wsi_image.name} cores and saving to disk"):
            core_image = self._extract_core_image(wsi_image, core)
            core_name = self.get_core_naming_convention(wsi_name=wsi_image.name, core=core)

            if self.resize:
                core_image = core_image.resize(self.resize)

            core_image.save(self.output_dir / f"{core_name}.{self.core_format}")

            if core.annotations:
                if self.resize:
                    self.resize_annotations(core=core, new_size=self.resize)

            if with_annotations:
                self._save_annotations(core, core_name=core_name)

    @staticmethod
    def _extract_core_image(wsi_image: WholeSlideImage, core: Core):
        return wsi_image.wsi.read_region((core.bbox.x, core.bbox.y), core.level, (core.bbox.w, core.bbox.h)).convert("RGB")

    def _save_annotations(self, core: Core, core_name: str):
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
