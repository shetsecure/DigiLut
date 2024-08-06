from config import Segmentationconfig, Coreconfig
from infer_interface import YOLOInference, DamoYOLOInference
from inference_manager import ChallengeInferenceManager

from wsi import WholeSlideImage


def main(
    batch_size=16,
    tta=False,
    conf=0.25,
    extraction_level=2,
    slides_path=r"/home/lipade/Challenge/val",
    challenge_csv_path="/home/lipade/Challenge/code/submission_sample.csv",
    save_to="test_submission.csv",
    slide_extracted_cores_path=r"/home/lipade/Challenge/code/inference_extracted_cores",
):
    seg_config = Segmentationconfig()
    # seg_config.min_aspect_ratio = 0.3
    # seg_config.min_area = 2e5

    min_aspect_ratio: float = 0.4
    min_area: int = 5e6

    # slide_extracted_cores_path += f"_min_area{seg_config.min_area}_aspect_{seg_config.min_aspect_ratio}"

    core_config = Coreconfig()

    damo_yolo_path = r"/home/lipade/damo-yolo/latest_ckpt.pth"
    config_file = r"damo/damoyolo_tinynasL20_T.py"
    damo_args = {"batch_size": batch_size, "conf": conf}
    damo_yolo_interface = DamoYOLOInference(config_file_path=config_file, model_args=damo_args, ckpt=damo_yolo_path)

    # wsi = WholeSlideImage("/home/lipade/Challenge/val/0Rv3MjnLWH_a.tif")
    # damo_yolo_interface.predict_slide_dir(
    #     wsi,
    #     seg_config=seg_config,
    #     core_config=core_config,
    #     extraction_level=extraction_level,
    #     slide_extracted_cores_path=r"/home/lipade/Challenge/code/inference_extracted_cores/0Rv3MjnLWH_a_extracted_cores",
    # )

    inference_manager = ChallengeInferenceManager(
        inference_interface=damo_yolo_interface,
        seg_config=seg_config,
        core_config=core_config,
        extraction_level=extraction_level,
        slides_path=slides_path,
        challenge_csv_path=challenge_csv_path,
        save_to=save_to,
    )

    inference_manager.run_validation_set_inference(slide_extracted_cores_path=slide_extracted_cores_path)

    print(f"Submission file ready at {save_to}")


if __name__ == "__main__":
    main(conf=0.425, save_to="submissions/damo_yolo_old_seg_conf_425_batched_128.csv", batch_size=128)
