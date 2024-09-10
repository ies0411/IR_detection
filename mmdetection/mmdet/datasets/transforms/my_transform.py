import random
from mmcv.transforms import BaseTransform
from mmdet.registry import TRANSFORMS

import cv2
from matplotlib import pyplot as plt

import albumentations as A


@TRANSFORMS.register_module()
class MyTransform(BaseTransform):
    """Add your transform

    Args:
        p (float): Probability of shifts. Default 0.5.
    """

    def __init__(self, prob=0.5):
        self.prob = prob

    def transform(self, results):
        print(f"results : {results}")
        # results["img"]
        # print(f'results : {results["img"].shape}')
        # print(print(results["img"][:, :, 0]))
        # print("!!!!!!!!!!!!!!!!!!!!!!!")
        # print(print(results["img"][:, :, 1]))
        # print("!!!!!!!!!!!!!!!!!!!!!!!")
        # print(results["img"][:, :, 2])
        # print("--------")

        if random.random() > self.prob:
            results["dummy"] = True
        return results
