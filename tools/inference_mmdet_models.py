import os
from natsort import natsorted
from mmdet.apis import DetInferencer
from PIL import Image
from mmengine.utils import ProgressBar, path

import json
from mmengine import ConfigDict
from mmengine.config import Config
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

    parser.add_argument("--tta", action="store_true", default=False)
    parser.add_argument("--scale", type=int, nargs=2, default=(1024, 1024))

    args = parser.parse_args()
    return args


def main():
    args = parse_config()
    test_dir_prefix = "../datasets/test_open/"
    norm_scale = 640
    print(f"scale : {args.scale}")
    device = "cuda:0"
    path.mkdir_or_exist("../output/")

    cfg = Config.fromfile(args.config_path)
    if args.tta:
        cfg.model = ConfigDict(**cfg.tta_model, module=cfg.model)
        cfg.test_dataloader.dataset.pipeline = cfg.tta_pipeline
    else:
        cfg.test_pipeline[1].scale = tuple(args.scale)
    inferencer = DetInferencer(cfg, args.checkpoint, device)

    # img = 'data/all_dataset/test/test_open_85.png'
    # result = inferencer(img, out_dir='./output')
    # Image.open('./output/vis/'+img.split("/")[-1])

    images = os.listdir(test_dir_prefix)
    sorted_images = natsorted(images)

    sub_results = []
    sub_results_for_fusion = []

    total_num = 0
    for image_dir in sorted_images:
        result = inferencer(os.path.join(test_dir_prefix, image_dir))
        obj_num = len(result["predictions"][0]["labels"])
        for idx in range(obj_num):
            min_x, min_y, max_x, max_y = result["predictions"][0]["bboxes"][idx]

            normalized_min_x = min_x / norm_scale
            normalized_min_y = min_y / norm_scale
            normalized_max_x = max_x / norm_scale
            normalized_max_y = max_y / norm_scale
            ret_fusion = {
                "image_id": str(image_dir.split(".")[0]),
                "category_id": result["predictions"][0]["labels"][idx],
                "bbox": [
                    normalized_min_x,
                    normalized_min_y,
                    normalized_max_x,
                    normalized_max_y,
                ],
                "score": result["predictions"][0]["scores"][idx],
            }
            sub_results_for_fusion.append(ret_fusion)

        print(f"====image : {total_num}=====")
        total_num += 1

    with open(
        args.output,
        "w",
    ) as file:
        json.dump(sub_results_for_fusion, file, indent=4)


if __name__ == "__main__":
    main()
