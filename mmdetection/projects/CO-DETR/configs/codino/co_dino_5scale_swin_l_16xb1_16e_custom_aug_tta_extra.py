_base_ = ["co_dino_5scale_r50_8xb2_1x_coco.py"]

pretrained = "https://github.com/SwinTransformer/storage/releases/download/v1.0.0/swin_large_patch4_window12_384_22k.pth"  # noqa
# load_from = "https://download.openmmlab.com/mmdetection/v3.0/codetr/co_dino_5scale_swin_large_16e_o365tococo-614254c9.pth"  # noqa
# load_from = "checkpoints/co_dino_5scale_swin_large_16e_o365tococo-614254c9.pth"
# load_from = "work_dirs/co_dino_5scale_swin_l_16xb1_16e_custom/epoch_4.pth"
# load_from = "work_dirs/ver4/epoch_1.pth"
# load_from = "work_dirs/ver6/epoch_4.pth"
load_from = "work_dirs/extra/epoch_1.pth"
resume = True
# dataset_type = "CocoDataset"
data_root = "data/datasets"
# image_size = (1024, 1024)

# train_data_root = "data/datasets/train"  # dataset root
# val_data_root = "data/datasets/val"
classes = ("person", "car", "truck", "bus", "bicycle", "bike", "extra_vehicle", "dog")
num_classes = 8
# base_batch_size = 1
image_size = (1024, 1024)

mean = [139.4229080324816, 139.4229080324816, 139.4229080324816]
std = [56.34895042201581, 56.34895042201581, 56.34895042201581]
# model settings
batch_augments = [dict(type="BatchFixedSizePad", size=image_size, pad_mask=True)]
model = dict(
    data_preprocessor=dict(
        type="DetDataPreprocessor",
        mean=mean,
        std=std,
        bgr_to_rgb=True,
        pad_mask=False,
        batch_augments=None,
    ),
    backbone=dict(
        _delete_=True,
        type="SwinTransformer",
        pretrain_img_size=384,
        embed_dims=192,
        depths=[2, 2, 18, 2],
        num_heads=[6, 12, 24, 48],
        window_size=12,
        mlp_ratio=4,
        qkv_bias=True,
        qk_scale=None,
        drop_rate=0.0,
        attn_drop_rate=0.0,
        drop_path_rate=0.3,
        patch_norm=True,
        out_indices=(0, 1, 2, 3),
        # Please only add indices that would be used
        # in FPN, otherwise some parameter will not be used
        with_cp=True,
        convert_weights=True,
        init_cfg=dict(type="Pretrained", checkpoint=pretrained),
    ),
    neck=dict(in_channels=[192, 384, 768, 1536]),
    query_head=dict(
        num_classes=num_classes,
        dn_cfg=dict(box_noise_scale=0.4, group_cfg=dict(num_dn_queries=500)),
        transformer=dict(encoder=dict(with_cp=6)),
    ),
    # tta_cfg=dict(nms=dict(type="nms", iou_threshold=0.6), max_per_img=100),
)
tta_model = dict(
    type="DetTTAModel",
    tta_cfg=dict(nms=dict(type="nms", iou_threshold=0.5), max_per_img=100),
)

img_norm_cfg = dict(
    mean=[139.4229080324816, 139.4229080324816, 139.4229080324816],
    std=[56.34895042201581, 56.34895042201581, 56.34895042201581],
    # to_rgb=True,
)
# 640 × 480
albu_train_transforms = [
    dict(
        type="ShiftScaleRotate",
        shift_limit=0.0625,
        scale_limit=0.1,
        rotate_limit=5,
        interpolation=1,
        p=0.1,
    ),
    # dict(type="Cutout", num_holes=1, max_h_size=20, max_w_size=20, p=0.1),
    dict(
        type="OneOf",
        transforms=[
            dict(type="Blur", blur_limit=1, p=0.2),
            dict(type="GaussNoise", var_limit=(10.0, 20.0), p=0.2),
            # dict(
            #     type="RandomBrightnessContrast",
            #     brightness_limit=[0.1, 0.3],
            #     contrast_limit=[0.1, 0.3],
            #     p=0.2,
            # ),
        ],
        p=0.5,
    ),
]

# img_scale = (640, 480)
train_pipeline = [
    dict(type="LoadImageFromFile"),
    dict(type="LoadAnnotations", with_bbox=True),
    # dict(type="MyTransform", prob=0.2),
    dict(
        type="Albu",
        transforms=albu_train_transforms,
        bbox_params=dict(
            type="BboxParams",
            format="pascal_voc",  # Ensure this matches your bbox format, could be 'coco' or 'pascal_voc'
            label_fields=["gt_bboxes_labels", "gt_ignore_flags"],
            min_visibility=0.0,
            filter_lost_elements=True,
        ),
        keymap={"img": "image", "gt_bboxes": "bboxes"},
        skip_img_without_anno=True,
    ),
    dict(type="RandomFlip", prob=0.3, direction="horizontal"),
    dict(
        type="RandomChoice",
        transforms=[
            [
                dict(
                    type="RandomChoiceResize",
                    scales=[
                        (480, 2048),
                        (512, 2048),
                        (544, 2048),
                        (576, 2048),
                        (608, 2048),
                        (640, 2048),
                        (672, 2048),
                        (704, 2048),
                        (736, 2048),
                        (768, 2048),
                        (800, 2048),
                        (832, 2048),
                        (864, 2048),
                        (896, 2048),
                        (928, 2048),
                        (960, 2048),
                        (992, 2048),
                        (1024, 2048),
                        (1056, 2048),
                        (1088, 2048),
                        (1120, 2048),
                        (1152, 2048),
                        (1184, 2048),
                        (1216, 2048),
                        (1248, 2048),
                        (1280, 2048),
                        (1312, 2048),
                        (1344, 2048),
                        (1376, 2048),
                        (1408, 2048),
                        (1440, 2048),
                        (1472, 2048),
                        (1504, 2048),
                        (1536, 2048),
                    ],
                    keep_ratio=True,
                )
            ],
            [
                dict(
                    type="RandomChoiceResize",
                    # The radio of all image in train dataset < 7
                    # follow the original implement
                    scales=[(400, 4200), (500, 4200), (600, 4200)],
                    keep_ratio=True,
                ),
                dict(
                    type="RandomCrop",
                    crop_type="absolute_range",
                    crop_size=(384, 600),
                    allow_negative_crop=True,
                ),
                dict(
                    type="RandomChoiceResize",
                    scales=[
                        (480, 2048),
                        (512, 2048),
                        (544, 2048),
                        (576, 2048),
                        (608, 2048),
                        (640, 2048),
                        (672, 2048),
                        (704, 2048),
                        (736, 2048),
                        (768, 2048),
                        (800, 2048),
                        (832, 2048),
                        (864, 2048),
                        (896, 2048),
                        (928, 2048),
                        (960, 2048),
                        (992, 2048),
                        (1024, 2048),
                        (1056, 2048),
                        (1088, 2048),
                        (1120, 2048),
                        (1152, 2048),
                        (1184, 2048),
                        (1216, 2048),
                        (1248, 2048),
                        (1280, 2048),
                        (1312, 2048),
                        (1344, 2048),
                        (1376, 2048),
                        (1408, 2048),
                        (1440, 2048),
                        (1472, 2048),
                        (1504, 2048),
                        (1536, 2048),
                    ],
                    keep_ratio=True,
                ),
            ],
        ],
    ),
    # dict(type="Normalize", **img_norm_cfg),
    dict(type="PackDetInputs"),
]

train_dataloader = dict(
    batch_size=1,
    num_workers=1,
    sampler=dict(type="DefaultSampler", shuffle=True),
    batch_sampler=dict(type="AspectRatioBatchSampler"),
    dataset=dict(
        metainfo=dict(classes=classes),
        data_root=data_root,
        data_prefix=dict(img="total_extra/"),
        ann_file="anno/total_extra.json",
        pipeline=train_pipeline,
    ),
)

# img_scale = (640, 480)
# image_size = (1024, 1024)

scale = (2048, 1280)
# scale = (1024, 1024)
# scale = (640, 480)
test_pipeline = [
    dict(type="LoadImageFromFile"),
    dict(type="Resize", scale=scale, keep_ratio=True),
    dict(type="LoadAnnotations", with_bbox=True),
    # dict(type="Normalize", **img_norm_cfg),
    dict(
        type="PackDetInputs",
        meta_keys=("img_id", "img_path", "ori_shape", "img_shape", "scale_factor"),
    ),
]

# tta_pipeline = [
#     dict(type="LoadImageFromFile"),
#     dict(
#         type="TestTimeAug",
#         transforms=[
#             [dict(type="Resize", scale=(1333, 800), keep_ratio=True)],
#             [  # It uses 2 flipping transformations (flipping and not flipping).
#                 dict(type="RandomFlip", prob=1.0),
#                 dict(type="RandomFlip", prob=0.0),
#             ],
#             [
#                 dict(
#                     type="PackDetInputs",
#                     meta_keys=(
#                         "img_id",
#                         "img_path",
#                         "ori_shape",
#                         "img_shape",
#                         "scale_factor",
#                         "flip",
#                         "flip_direction",
#                     ),
#                 )
#             ],
#         ],
#     ),
# ]

img_scales = [(2048, 1280), (1333, 800), (666, 400), (2000, 1200), (640, 480)]
# img_scale = (640, 480)
tta_pipeline = [
    dict(type="LoadImageFromFile"),
    dict(
        type="TestTimeAug",
        transforms=[
            [dict(type="Resize", scale=s, keep_ratio=True) for s in img_scales],
            [
                dict(type="RandomFlip", direction="horizontal", prob=1.0),
                dict(type="RandomFlip", prob=0.0),
            ],
            [
                dict(
                    type="PackDetInputs",
                    meta_keys=(
                        "img_id",
                        "img_path",
                        "ori_shape",
                        "img_shape",
                        "scale_factor",
                        "flip",
                        "flip_direction",
                    ),
                )
            ],
        ],
    ),
]


val_dataloader = dict(
    dataset=dict(
        pipeline=test_pipeline,
        test_mode=True,
        metainfo=dict(classes=classes),
        data_prefix=dict(img="val/"),
        data_root=data_root,
        ann_file="anno/val.json",
    )
)
test_dataloader = val_dataloader

val_evaluator = dict(ann_file=data_root + "/anno/val.json")
test_evaluator = val_evaluator


optim_wrapper = dict(optimizer=dict(lr=1e-4))
# optim_wrapper = dict(
#     optimizer=dict(
#         type="Adafactor",
#         lr=1e-5,
#         beta1=(0.9, 0.999),
#         weight_decay=1e-2,
#         scale_parameter=False,
#         relative_step=False,
#     )
# )
# optim_wrapper = dict(
#     optimizer=dict(
#         type="SophiaG", lr=1e-5, betas=(0.965, 0.99), rho=0.01, weight_decay=1e-1
#     )
# )
# optim_wrapper = dict(optimizer=dict(type="Lion", lr=1e-4, weight_decay=1e-2))


max_epochs = 5


# img_scale = (640, 480)
train_pipeline_2 = [
    dict(type="LoadImageFromFile"),
    dict(type="LoadAnnotations", with_bbox=True),
    # dict(type="Normalize", **img_norm_cfg),
    dict(type="PackDetInputs"),
]

# custom_hooks = [
#     dict(
#         type="PipelineSwitchHook",
#         switch_epoch=max_epochs - 1,
#         switch_pipeline=train_pipeline_2,
#     )
# ]

train_cfg = dict(max_epochs=max_epochs, val_interval=20)

param_scheduler = [
    dict(
        type="MultiStepLR",
        begin=0,
        end=max_epochs,
        by_epoch=True,
        milestones=[8],
        gamma=0.1,
    )
]
