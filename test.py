from pathlib import Path

from config import Config
from wsi import WholeSlideImage
from tissue_segmenter import TissueSegmenter
from core_extractor import CoreExtractor
from core_manager import CoreAnnotationManager, AnnotationFormat

import utils

config = Config()


def extract_and_save_cores(
    wsi_path, annotations, extraction_level, save_dir, annotation_format=AnnotationFormat.YOLO, cores_format="png", resize=None
):
    wsi = WholeSlideImage(wsi_path)

    # Segment tissue in WSI
    segmenter = TissueSegmenter(config)
    tissue_segments = segmenter.segment(wsi)

    # Extract cores
    core_extractor = CoreExtractor(wsi, config)

    if len(annotations) > 0:
        cores = core_extractor.extract_cores_with_annotations(tissue_segments, extraction_level, annotations, True)
    else:
        cores = core_extractor.extract_cores(tissue_segments, extraction_level=extraction_level)

    # Save them
    cores_manager = CoreAnnotationManager(output_dir=save_dir, annotation_format=annotation_format, cores_format=cores_format)
    cores_manager.save_cores(wsi, cores, resize=resize)


if __name__ == "__main__":
    wsi_path = Path("/media/lipade/Crucial X6/Trusti/training_set_slides/data/images/bGaslniO4a_a.tif")
    extraction_level = 2
    save_dir = r"/media/lipade/Crucial X6/Trusti/training_set_slides/test/test_new"

    all_annotations = utils.get_cleaned_annotations_xywh("challenge_annotations/train.csv", "challenge_annotations/Boundig_Box_IDs/")
    annotations = all_annotations["bGaslniO4a_a.tif"]

    extract_and_save_cores(wsi_path, annotations, extraction_level=0, save_dir=save_dir, resize=1024)
    extract_and_save_cores(wsi_path, annotations, extraction_level=extraction_level, save_dir=save_dir, resize=None)
    extract_and_save_cores(wsi_path, annotations, extraction_level=extraction_level, save_dir=save_dir, resize=1024)
