import cv2
import numpy as np
from pathlib import Path
import matplotlib.pyplot as plt
import matplotlib.patches as patches

from dataclasses import dataclass
from typing import List, NamedTuple, Optional, Tuple
from types import FunctionType
from copy import deepcopy


import utils

# import mapping
from tissue_segmenter import TissueSegment
from config import Config


class BoundingBox(NamedTuple):
    x: int
    y: int
    w: int
    h: int


@dataclass
class Core:
    bbox: BoundingBox
    annotations: List[BoundingBox]
    level: int


class CoreExtractor:
    def __init__(self, wsi, config: Config):
        self.wsi = wsi
        self.config = config

    def extract_cores(self, tissue_segments: List[TissueSegment], extraction_level: int) -> List[Core]:
        """
        Extracts cores from given tissue segments at a specified extraction level.

        Parameters:
        - tissue_segments (List[TissueSegment]): List of TissueSegment objects representing tissue segments.
        - extraction_level (int): The level at which to extract the cores.

        Returns:
        - List[Core]: List of Core objects extracted from the tissue segments.
        """
        cores = []
        scale = 1 / np.asarray(self.wsi.level_downsamples[extraction_level])

        for segment in tissue_segments:
            x, y, w, h = cv2.boundingRect(segment.contour)
            w = round(w * scale[0])
            h = round(h * scale[1])

            cores.append(Core(BoundingBox(x, y, w, h), annotations=[], level=extraction_level))

        return cores

    def extract_cores_with_annotations(
        self,
        tissue_segments: List[TissueSegment],
        extraction_level,
        wsi_annotation_bboxes,
        only_cores_with_ann=False,
    ) -> List[Core]:
        """
        Extract core images and their corresponding annotations, relative to their cores, from a whole slide image.

        :param extraction_level: The level at which to extract the cores
        :param wsi_annotation_bboxes: List of annotation bounding boxes (assumed to be at level 0)
        :param output_dir: Directory to save extracted cores and annotations
        :param format: Format of the annotations (currently only supports "yolo")
        """
        # Here, the cores will be at level 0.
        cores = self.adjust_tissue_segments_with_annotations(tissue_segments, wsi_annotation_bboxes)

        # Scale them to the extraction level
        scaled_cores = self._scale_cores(cores, extraction_level, round)

        if only_cores_with_ann:
            scaled_cores = list(filter(lambda core: len(core.annotations) > 0, scaled_cores))

        return scaled_cores

    def adjust_tissue_segments_with_annotations(self, tissue_segments: List[TissueSegment], annotations: List[BoundingBox]) -> List[Core]:
        """
        Adjust and associate each annotation box with its corresponding core.
        This has to be done at level 0 MANDATORY, because we only scale w,h without xy.

        Assuming the annotations boxes have the following format: x, y, w, h

        Adjusts the tissue segments with corresponding annotations.

        :param tissue_segments: List of TissueSegment objects representing tissue segments.
        :param annotations: List of BoundingBox objects representing annotations.
        :return: List of adjusted Core objects to encapsulate their corresponding annotations.

        """
        cores = self.extract_cores(tissue_segments=tissue_segments, extraction_level=0)

        for core in cores:
            for ann in annotations:
                is_included = utils.is_box_included(ann, core.bbox)
                intersection = utils.find_intersection(ann, core.bbox)

                if not is_included and intersection:
                    core.bbox = BoundingBox(*utils.adjust_tissue_box(core.bbox, ann))
                    ann = BoundingBox(*ann)

                if intersection or is_included:
                    # Link the corresponding annotations and make it relative
                    ann = BoundingBox(*ann)
                    ann = BoundingBox(ann.x - core.bbox.x, ann.y - core.bbox.y, ann.w, ann.h)
                    core.annotations.append(ann)

        return cores

    # def save_cores(self, cores: List[Core], output_dir: Path, extraction_level: int):
    #     output_dir.mkdir(parents=True, exist_ok=True)

    #     for i, core in enumerate(cores):
    #         core_region = self.wsi.wsi.read_region((core.bbox.x, core.bbox.y), extraction_level, (core.bbox.w, core.bbox.h)).convert("RGB")
    #         core_name = f"{self.wsi.name}_Core_{i}_L{extraction_level}_{core.bbox.w}_{core.bbox.h}"
    #         core_region.save(output_dir / f"{core_name}.png")

    #         yolo_annotations = [
    #             utils.convert_to_yolo_format((ann.x, ann.y, ann.w, ann.h), core.bbox.w, core.bbox.h) for ann in core.annotations
    #         ]

    #         with open(output_dir / f"{core_name}.txt", "w") as f:
    #             for annotation in yolo_annotations:
    #                 f.write(" ".join(map(str, annotation)) + "\n")

    def visualize_cores_with_annotations(self, cores: List[Core], extraction_level: int, cast_fn=round):
        """
        Visualize cores with their annotations at a specified extraction level.

        Parameters:
        - cores (List[Core]): List of Core objects to visualize.
        - extraction_level (int): The level at which to extract the cores for visualization.
        - cast_fn (FunctionType): Function to cast float to int, round for cores, int for bounding boxes.
        """
        scaled_cores = self._scale_cores(cores, to_level=extraction_level, cast_fn=cast_fn)

        for i, core in enumerate(scaled_cores):
            if core.annotations:
                core_region = self.wsi.wsi.read_region((core.bbox.x, core.bbox.y), extraction_level, (core.bbox.w, core.bbox.h)).convert(
                    "RGB"
                )

                _, ax = plt.subplots(1, figsize=(10, 10))
                ax.imshow(np.array(core_region))

                for annotation in core.annotations:
                    rect = patches.Rectangle(
                        (annotation.x, annotation.y),
                        annotation.w,
                        annotation.h,
                        linewidth=1,
                        edgecolor="r",
                        facecolor="none",
                    )
                    ax.add_patch(rect)

                ax.set_title(f"Core {i} with {len(core.annotations)} Annotations")
                plt.axis("off")

                plt.show()

    def _scale_cores(self, cores: List[Core], to_level: int, cast_fn) -> List[Core]:
        """
        Scale the bounding boxes.

        :param cast_fn: Function to cast float to int, round for cores, int for bounding boxes.
        """

        for i in range(len(cores)):
            cores[i] = self.map_core_to_level(core=cores[i], to_level=to_level, cast_fn=cast_fn)

        return cores

    def core_to_0(self, core: Core, cast_fn: FunctionType = round) -> Core:
        """
        Adjusts the given core to level 0 by scaling its bounding box dimensions and annotations accordingly.

        Parameters:
        - wsi (WholeSlideImage): The WholeSlideImage object containing the core.
        - core (Core): The core to be adjusted to level 0.
        - cast_fn (FunctionType): The function used for casting the dimensions to int ( round or int ).

        Returns:
        - Core: The core adjusted to level 0.
        """
        assert core.level in range(self.wsi.wsi.level_count), f"Core level {core.level} isn't valid for this WSI. probably invalid Core"

        extract_level_to_0_factor = self.wsi.wsi.level_downsamples[core.level]

        if core.level == 0:
            # print(f"core already at level 0, didn't change anything")
            return core

        elif core.level > 0:
            # Adjust the core, only w and h cuz x,y for openslide are always in level 0

            new_w = cast_fn(core.bbox.w * extract_level_to_0_factor)
            new_h = cast_fn(core.bbox.h * extract_level_to_0_factor)
            new_core = Core(BoundingBox(core.bbox.x, core.bbox.y, new_w, new_h), deepcopy(core.annotations), level=0)

            # adjusting the annotations
            for i in range(len(new_core.annotations)):
                adjusted_annotation_bbox = list(map(lambda x: cast_fn(x * extract_level_to_0_factor), new_core.annotations[i]))
                new_core.annotations[i] = BoundingBox(*adjusted_annotation_bbox)

        else:
            print(f"level of core {core} is negative ! ")

        return new_core

    def map_core_to_level(self, core: Core, to_level: int, cast_fn: FunctionType = round) -> Core:
        """
        Adjusts the given core to the specified level by scaling its bounding box dimensions and annotations accordingly.

        Parameters:
        - wsi (WholeSlideImage): The WholeSlideImage object containing the core.
        - core (Core): The core to be adjusted to the specified level.
        - to_level (int): The level to which the core will be adjusted.
        - cast_fn (FunctionType): The function used for casting the dimensions to the appropriate type.

        Returns:
        - Core: The core adjusted to the specified level.
        """
        assert to_level in range(self.wsi.wsi.level_count), f"to_level {to_level} isn't valid for this WSI"

        core_0 = self.core_to_0(core, cast_fn)

        scale_factor = 1 / np.asarray(self.wsi.level_downsamples[to_level])

        x, y, w, h = core_0.bbox
        core_to_level = Core(BoundingBox(x, y, cast_fn(w * scale_factor[0]), cast_fn(h * scale_factor[1])), [], to_level)

        for j in range(len(core_0.annotations)):
            x, y, w, h = core_0.annotations[j]
            core_to_level.annotations.append(
                BoundingBox(
                    cast_fn(x * scale_factor[0]), cast_fn(y * scale_factor[1]), cast_fn(w * scale_factor[0]), cast_fn(h * scale_factor[1])
                )
            )

        return core_to_level
