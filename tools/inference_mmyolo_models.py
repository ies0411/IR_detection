# Copyright (c) OpenMMLab. All rights reserved.
import os
from argparse import ArgumentParser
from pathlib import Path

import mmcv
from mmdet.apis import inference_detector, init_detector
from mmengine.config import Config, ConfigDict
from mmengine.logging import print_log
from mmengine.utils import ProgressBar, path

from mmyolo.registry import VISUALIZERS
from mmyolo.utils import switch_to_deploy
from mmyolo.utils.labelme_utils import LabelmeFormat
from mmyolo.utils.misc import get_file_list, show_data_classes

from PIL import Image
from natsort import natsorted
import json


import argparse


def parse_config():
    parser = argparse.ArgumentParser(description="arg parser")
    parser.add_argument(
        "--config_path",
        type=str,
        required=True,
    )
    parser.add_argument(
        "--checkpoint",
        type=str,
        required=True,
    )
    parser.add_argument(
        "--output",
        type=str,
        required=True,
    )

    parser.add_argument("--tta_flag", action="store_true", default=False)

    args = parser.parse_args()
    return args


def main():

    device = "cuda:0"
    norm_scale = 640
    path.mkdir_or_exist("../output/")
    test_dir_prefix = "../datasets/test_open/"

    args = parse_config()
    config = Config.fromfile(args.config_path)

    if "init_cfg" in config.model.backbone:
        config.model.backbone.init_cfg = None

    if args.tta_flag:
        assert "tta_model" in config, (
            "Cannot find ``tta_model`` in config." " Can't use tta !"
        )
        assert "tta_pipeline" in config, (
            "Cannot find ``tta_pipeline`` " "in config. Can't use tta !"
        )
        config.model = ConfigDict(**config.tta_model, module=config.model)
        test_data_cfg = config.test_dataloader.dataset
        while "dataset" in test_data_cfg:
            test_data_cfg = test_data_cfg["dataset"]

        # batch_shapes_cfg will force control the size of the output image,
        # it is not compatible with tta.
        if "batch_shapes_cfg" in test_data_cfg:
            test_data_cfg.batch_shapes_cfg = None
        test_data_cfg.pipeline = config.tta_pipeline

    model = init_detector(config, args.checkpoint, device=device, cfg_options={})
    switch_to_deploy(model)

    images = os.listdir(test_dir_prefix)
    sorted_images = natsorted(images)

    sub_results_for_fusion = []
    total_num = 0
    for image_dir in sorted_images:
        result = inference_detector(model, os.path.join(test_dir_prefix, image_dir))

        bboxes = result.get("pred_instances").get("bboxes").tolist()
        scores = result.get("pred_instances").get("scores").tolist()
        labels = result.get("pred_instances").get("labels").tolist()
        obj_num = len(bboxes)
        for idx in range(obj_num):

            min_x, min_y, max_x, max_y = bboxes[idx]

            normalized_min_x = min_x / norm_scale
            normalized_min_y = min_y / norm_scale
            normalized_max_x = max_x / norm_scale
            normalized_max_y = max_y / norm_scale

            ret_fusion = {
                "image_id": str(image_dir.split(".")[0]),
                "category_id": labels[idx],
                "bbox": [
                    normalized_min_x,
                    normalized_min_y,
                    normalized_max_x,
                    normalized_max_y,
                ],
                "score": scores[idx],
            }

            # sub_results.append(ret)
            sub_results_for_fusion.append(ret_fusion)
        #     break
        total_num += 1
        print(f"image : {total_num}")

    with open(args.output, "w") as file:
        json.dump(sub_results_for_fusion, file, indent=4)


if __name__ == "__main__":
    main()
