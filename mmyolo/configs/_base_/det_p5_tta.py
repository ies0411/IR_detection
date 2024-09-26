# TODO: Need to solve the problem of multiple backend_args parameters
# _backend_args = dict(
#     backend='petrel',
#     path_mapping=dict({
#         './data/': 's3://openmmlab/datasets/detection/',
#         'data/': 's3://openmmlab/datasets/detection/'
#     }))

_backend_args = None

tta_model = dict(
    type="mmdet.DetTTAModel",
    tta_cfg=dict(nms=dict(type="nms", iou_threshold=0.75), max_per_img=200),  # 300
    # tta_cfg=dict(nms=dict(type="nms", iou_threshold=0.5), max_per_img=100),
)

# img_scales = [(640, 640), (320, 320), (960, 960)]
img_scales = [(640, 640), (320, 320), (240, 240), (160, 160), (450, 450), (800, 800)]
# (320, 800)
# img_scales = [(640, 640), (320, 320), (960, 960), (1280, 1280)]

#                                LoadImageFromFile
#                     /                 |                     \
# (RatioResize,LetterResize) (RatioResize,LetterResize) (RatioResize,LetterResize) # noqa
#        /      \                    /      \                    /        \
#  RandomFlip RandomFlip      RandomFlip RandomFlip        RandomFlip RandomFlip # noqa
#      |          |                |         |                  |         |
#  LoadAnn    LoadAnn           LoadAnn    LoadAnn           LoadAnn    LoadAnn
#      |          |                |         |                  |         |
#  PackDetIn  PackDetIn         PackDetIn  PackDetIn        PackDetIn  PackDetIn # noqa

_multiscale_resize_transforms = [
    dict(
        type="Compose",
        transforms=[
            dict(type="YOLOv5KeepRatioResize", scale=s),
            dict(
                type="LetterResize",
                scale=s,
                allow_scale_up=False,
                pad_val=dict(img=114),
            ),
        ],
    )
    for s in img_scales
]

tta_pipeline = [
    dict(type="LoadImageFromFile", backend_args=_backend_args),
    dict(
        type="TestTimeAug",
        transforms=[
            _multiscale_resize_transforms,
            [
                dict(type="mmdet.RandomFlip", direction="horizontal", prob=1.0),
                dict(type="mmdet.RandomFlip", direction="horizontal", prob=0.0),
            ],
            [dict(type="mmdet.LoadAnnotations", with_bbox=True)],
            [
                dict(
                    type="mmdet.PackDetInputs",
                    meta_keys=(
                        "img_id",
                        "img_path",
                        "ori_shape",
                        "img_shape",
                        "scale_factor",
                        "pad_param",
                        "flip",
                        "flip_direction",
                    ),
                )
            ],
        ],
    ),
]
