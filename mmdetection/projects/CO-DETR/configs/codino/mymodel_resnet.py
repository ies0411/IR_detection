_base_ = ["co_dino_5scale_r50_8xb2_1x_coco.py"]


load_from = "../weights/codetr_resnet/pretrained_model.pth"
data_root = "../datasets/"
classes = ("person", "car", "truck", "bus", "bicycle", "bike", "extra_vehicle", "dog")
num_classes = 8
image_size = (1024, 1024)
batch_size = 1
num_workers = 4
num_gpu = 3
base_lr = 1e-4
max_epochs = 35

# model settings
mean = [139.4229080324816, 139.4229080324816, 139.4229080324816]
std = [56.34895042201581, 56.34895042201581, 56.34895042201581]

batch_augments = [dict(type="BatchFixedSizePad", size=image_size, pad_mask=True)]

model = dict(
    data_preprocessor=dict(
        type="DetDataPreprocessor",
        mean=mean,
        std=std,
        bgr_to_rgb=True,
        pad_mask=True,
        batch_augments=batch_augments,
    ),
    backbone=dict(
        depth=101,
        frozen_stages=-1,  # 1 -> -1 (epoch 15)
        norm_cfg=dict(type="BN", requires_grad=False),
        style="pytorch",
        init_cfg=dict(type="Pretrained", checkpoint="torchvision://resnet101"),
    ),
    query_head=dict(
        num_classes=num_classes,
        dn_cfg=dict(box_noise_scale=0.4, group_cfg=dict(num_dn_queries=500)),
        transformer=dict(encoder=dict(with_cp=6)),
    ),
)

tta_model = dict(
    type="DetTTAModel",
    tta_cfg=dict(nms=dict(type="nms", iou_threshold=0.75), max_per_img=200),
)

albu_train_transforms = [
    dict(
        type="ShiftScaleRotate",
        shift_limit=0.0625,
        scale_limit=0.1,
        rotate_limit=10,
        interpolation=1,
        p=0.2,
    ),
    dict(
        type="OneOf",
        transforms=[
            dict(type="Cutout", num_holes=5, max_h_size=20, max_w_size=20, p=0.2),
        ],
        p=1.0,
    ),
    dict(
        type="OneOf",
        transforms=[
            dict(type="Blur", blur_limit=3, p=0.2),
            dict(type="GaussNoise", var_limit=(10.0, 20.0), p=0.2),
            # dict(type="MedianBlur", p=0.2),
            # dict(type="CLAHE", p=0.2),
            # dict(
            #     type="RandomBrightnessContrast",
            #     brightness_limit=[0.05, 0.1],
            #     contrast_limit=[0.05, 0.1],
            #     p=0.2,
            # ),
        ],
        p=1.0,
    ),
]


train_pipeline = [
    dict(type="LoadImageFromFile"),
    dict(type="LoadAnnotations", with_bbox=True),
    dict(
        type="Albu",
        transforms=albu_train_transforms,
        bbox_params=dict(
            type="BboxParams",
            format="pascal_voc",
            label_fields=["gt_bboxes_labels", "gt_ignore_flags"],
            min_visibility=0.0,
            filter_lost_elements=True,
        ),
        keymap={"img": "image", "gt_bboxes": "bboxes"},
        skip_img_without_anno=True,
    ),
    dict(type="RandomFlip", direction="horizontal", prob=0.5),
    #
    dict(
        type="RandomChoice",
        transforms=[
            [
                dict(
                    type="RandomChoiceResize",
                    scales=[
                        (400, 2048),
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
                        (400, 2048),
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
    dict(type="Pad", size=image_size, pad_val=dict(img=(114, 114, 114))),
    dict(type="FilterAnnotations", min_gt_bbox_wh=(1e-1, 1e-1)),
    dict(type="PackDetInputs"),
]
# https://mmdetection.readthedocs.io/en/v2.11.0/tutorials/customize_dataset.html :concat dataset, balancing dataset
train_dataloader = dict(
    batch_size=batch_size,
    num_workers=num_workers,
    sampler=dict(type="DefaultSampler", shuffle=True),
    batch_sampler=dict(type="AspectRatioBatchSampler"),
    dataset=dict(
        metainfo=dict(classes=classes),
        data_root=data_root,
        data_prefix=dict(img="total/"),
        ann_file="anno/total.json",
        pipeline=train_pipeline,
    ),
)
scale = (2048, 1280)
test_pipeline = [
    dict(type="LoadImageFromFile"),
    dict(type="Resize", scale=scale, keep_ratio=True),
    dict(type="Pad", size=image_size, pad_val=dict(img=(114, 114, 114))),
    dict(type="LoadAnnotations", with_bbox=True),
    dict(
        type="PackDetInputs",
        meta_keys=("img_id", "img_path", "ori_shape", "img_shape", "scale_factor"),
    ),
]

img_scales = [(2048, 1280), (1333, 800), (1024, 1024), (666, 430)]

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
            [dict(type="Pad", size=image_size, pad_val=dict(img=(114, 114, 114)))],
            [dict(type="LoadAnnotations", with_bbox=True)],
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
    batch_size=1,
    num_workers=1,
    dataset=dict(
        pipeline=test_pipeline,
        test_mode=True,
        metainfo=dict(classes=classes),
        data_prefix=dict(img="val/"),
        data_root=data_root,
        ann_file="anno/val.json",
    ),
)
test_dataloader = val_dataloader

val_evaluator = dict(ann_file=data_root + "/anno/val.json")
test_evaluator = val_evaluator
optim_wrapper = dict(
    optimizer=dict(lr=base_lr),
    paramwise_cfg=dict(custom_keys={"backbone": dict(lr_mult=0.1)}),
)

# optim_wrapper = dict(
#     _delete_=True,
#     type="OptimWrapper",
#     optimizer=dict(type="AdamW", lr=1e-4, weight_decay=0.0001),
#     clip_grad=dict(max_norm=0.1, norm_type=2),
#     paramwise_cfg=dict(custom_keys={"backbone": dict(lr_mult=0.1)}),
# )


default_hooks = dict(
    checkpoint=dict(
        type="CheckpointHook",
        interval=2000,
        by_epoch=False,
        save_best="auto",
        max_keep_ckpts=5,
    ),
)


train_cfg = dict(max_epochs=max_epochs, val_interval=5)

param_scheduler = [
    dict(
        type="MultiStepLR",
        begin=0,
        end=max_epochs,
        by_epoch=True,
        milestones=[35],
        gamma=0.1,
    )
]
# param_scheduler = [
#     dict(
#         # use cosine lr from 5 to 285 epoch
#         type="CosineAnnealingLR",
#         eta_min=base_lr * 0.1,
#         begin=0,
#         T_max=max_epochs // 4,
#         end=max_epochs,
#         by_epoch=True,
#         convert_to_iter_based=True,
#     ),
# ]
