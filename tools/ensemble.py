import os
import json
from natsort import natsorted

from ensemble_boxes import *


def load_json(file_path):
    with open(file_path, "r") as f:
        data = json.load(f)
    return data


def load_txt_files_natsorted(folder_path):
    txt_files_content = []

    # Get a list of all .txt files in the folder and sort them naturally
    txt_files = natsorted([f for f in os.listdir(folder_path) if f.endswith(".txt")])

    # Loop through the naturally sorted list of files
    for filename in txt_files:
        file_path = os.path.join(folder_path, filename)
        txt_files_content.append(load_json(file_path))

    return txt_files_content


def denormalize_bbox(
    normalized_min_x,
    normalized_min_y,
    normalized_max_x,
    normalized_max_y,
):
    norm_scale = 640
    min_x = normalized_min_x * norm_scale
    min_y = normalized_min_y * norm_scale
    max_x = normalized_max_x * norm_scale
    max_y = normalized_max_y * norm_scale

    return min_x, min_y, max_x, max_y


def converting_data(datas):
    bboxes_list1 = []
    scores_list1 = []
    labels_list1 = []
    images_list1 = []

    idx = 0
    boxes = []
    scores = []
    labels = []
    for data in datas:
        if int(data["image_id"].split("_")[-1]) == idx:
            boxes.append(data["bbox"])
            labels.append(data["category_id"])
            scores.append(data["score"])
            image_name = data["image_id"]
        else:
            bboxes_list1.append(boxes)
            scores_list1.append(scores)
            labels_list1.append(labels)
            images_list1.append(image_name)
            boxes = []
            scores = []
            labels = []
            boxes.append(data["bbox"])
            scores.append(data["score"])
            labels.append(data["category_id"])
            idx += 1
    bboxes_list1.append(boxes)
    scores_list1.append(scores)
    labels_list1.append(labels)
    images_list1.append(image_name)
    return bboxes_list1, scores_list1, labels_list1, images_list1


def main():

    data_list = load_txt_files_natsorted("../output")
    bboxes_list = []
    scores_list = []
    labels_list = []
    images_list = []

    for data in data_list:
        bboxes, scores, labels, images = converting_data(data)
        bboxes_list.append(bboxes)
        scores_list.append(scores)
        labels_list.append(labels)
        images_list.append(images)

    iou_thr = 0.85
    skip_box_thr = 0.0001
    # sigma = 0.05
    sub_results = []

    # images_list1
    for idx, bboxes in enumerate(bboxes_list[0]):
        c_boxes_list = [bboxes_list[i][idx] for i in range(14)]
        c_scores_list = [scores_list[i][idx] for i in range(14)]
        c_labels_list = [labels_list[i][idx] for i in range(14)]

        image_name = images_list[0][idx]
        weights = [
            1,
            1,
            1,
            1,
            1.5,
            1.5,
            1.5,
            1.5,
            2.5,
            2.5,
            2.5,
            2.5,
            2.5,
            2.5,
            3.0,
            3.0,
        ]

        boxes, scores, labels = weighted_boxes_fusion(
            c_boxes_list,
            c_scores_list,
            c_labels_list,
            weights=weights,
            iou_thr=iou_thr,
            skip_box_thr=skip_box_thr,
        )

        for bbox_idx, box in enumerate(boxes):

            min_x, min_y, max_x, max_y = denormalize_bbox(
                box[0], box[1], box[2], box[3]
            )

            ret = {
                "image_id": image_name,
                "category_id": int(labels[bbox_idx]),
                "bbox": [min_x, min_y, max_x - min_x, max_y - min_y],
                "score": scores[bbox_idx],
            }
            sub_results.append(ret)

    with open("../submit_output/final_results.txt", "w") as file:
        json.dump(sub_results, file, indent=4)


if __name__ == "__main__":
    main()
