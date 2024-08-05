import os

from config import Segmentationconfig, Coreconfig
from wsi import WholeSlideImage
from core_extractor import Core, CoreExtractor
from core_manager import CoreAnnotationManager
from infer_interface import InferenceInterface
from tissue_segmenter import TissueSegmenter

from pathlib import Path
from tqdm import tqdm

import random

random.seed = 43

SEG_CONFIG = Segmentationconfig()
CORE_CONFIG = Coreconfig()

SLIDES_PATH = r"/home/lipade/Challenge/no_lesions_slides"
EXTRACTION_LEVEL = 2
OUTPUT_DIR = r"/media/lipade/Crucial X6/Trusti/training_set_slides/cores_with_no_lesions_2"
N_CORES = 700


def extract_wsi_cores(seg_config: Segmentationconfig, wsi: WholeSlideImage, extraction_level: int):
    # Segment tissue in WSI
    segmenter = TissueSegmenter(seg_config)
    tissue_segments = segmenter.segment(wsi)

    # Extract cores
    core_extractor = CoreExtractor(wsi)
    cores = core_extractor.extract_cores(tissue_segments, extraction_level=extraction_level)

    return cores


# /home/lipade/Challenge/no_lesions_slides
def extract_cores_from_slides(slides_path: str, extraction_level: int, output_dir: str, n_cores: int = None):

    wsis_names = [f for f in list(os.listdir(slides_path)) if f.endswith(".tif")]

    if n_cores:
        print(f"Requested extracting {n_cores} cores from {len(wsis_names)} given slides")
        take_n_cores_per_slide = n_cores // len(wsis_names)

        print(f"Will take {take_n_cores_per_slide} cores from each slide")

        for wsi_name in tqdm(wsis_names, desc=f"Extracting wsi cores"):
            wsi_path = Path(os.path.join(slides_path, wsi_name))
            wsi = WholeSlideImage(wsi_path)

            cores = extract_wsi_cores(seg_config=SEG_CONFIG, wsi=wsi, extraction_level=extraction_level)

            random.shuffle(cores)

            cores_to_save = cores[:take_n_cores_per_slide]

            core_manager = CoreAnnotationManager(output_dir=output_dir, core_config=CORE_CONFIG)

            core_manager.save_cores(wsi, cores_to_save, with_annotations=True)


if __name__ == "__main__":

    extract_cores_from_slides(slides_path=SLIDES_PATH, extraction_level=EXTRACTION_LEVEL, output_dir=OUTPUT_DIR, n_cores=N_CORES)
