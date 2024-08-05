from typing import List
from pathlib import Path

import pandas as pd

import utils
from config import Segmentationconfig, Coreconfig
from wsi import WholeSlideImage
from core_extractor import Core
from infer_interface import InferenceInterface

from tqdm import tqdm


class ChallengeInferenceManager:
    def __init__(
        self,
        inference_interface: InferenceInterface,
        seg_config: Segmentationconfig,
        core_config: Coreconfig,
        extraction_level: int,
        slides_path: str,
        challenge_csv_path: str,
        save_to: str,
    ):
        self.inference_interface = inference_interface

        self.seg_config = seg_config
        self.core_config = core_config
        self.extraction_level = extraction_level

        self.slides_path = slides_path
        self.challenge_csv_path = challenge_csv_path
        self.save_to = save_to

        self.df = pd.read_csv(challenge_csv_path, sep=",")

    def run_slide_inference(
        self,
        wsi: WholeSlideImage,
        seg_config: Segmentationconfig,
        core_config: Coreconfig,
        extraction_level: int,
        number_of_required_boxes: int,
        slide_extracted_cores_path=None,
    ) -> List:

        cores_with_lesions = self.inference_interface.predict_slide_dir(
            wsi=wsi,
            seg_config=seg_config,
            core_config=core_config,
            extraction_level=extraction_level,
            slide_extracted_cores_path=slide_extracted_cores_path,
        )

        predicted_boxes_0 = utils.xyxys_to_0(wsi, cores_with_lesions)
        total_boxes_predicted = len(predicted_boxes_0)
        print(f"Original number of predicted boxes is {total_boxes_predicted}")

        final_pred_boxes = utils.post_process_predicted_boxes_0(
            predicted_boxes_0=predicted_boxes_0, number_of_required_boxes=number_of_required_boxes
        )

        print(f"Made them {len(final_pred_boxes)} boxes to match the submission file")

        wsi.wsi.close()

        return final_pred_boxes

    def run_validation_set_inference(self, slide_extracted_cores_path=None):
        required_boxes_per_slide = self.df["filename"].value_counts().to_dict()
        predicted_boxes_per_slide = dict()

        df = self.df

        for slide_name, n_required_boxes in tqdm(required_boxes_per_slide.items(), desc=f"Predicting slides"):
            slide_path = Path(self.slides_path) / slide_name
            wsi = WholeSlideImage(slide_path)

            predicted_boxes = self.run_slide_inference(
                wsi=wsi,
                seg_config=self.seg_config,
                core_config=self.core_config,
                extraction_level=self.extraction_level,
                number_of_required_boxes=n_required_boxes,
                slide_extracted_cores_path=slide_extracted_cores_path,
            )

            predicted_boxes_per_slide[wsi.name + ".tif"] = predicted_boxes

        for i in range(len(df)):
            slide_name = df.iloc[i]["filename"]
            x, y, xx, yy = predicted_boxes_per_slide[slide_name].pop(0)
            df.at[i, "x1"] = x
            df.at[i, "y1"] = y
            df.at[i, "x2"] = xx
            df.at[i, "y2"] = yy

        df.to_csv(Path(self.save_to).stem + ".csv", sep=",", index=None)
