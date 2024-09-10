_base_ = ["co_dino_5scale_r50_8xb2_1x_coco.py"]
load_from = "checkpoints/co_dino_5scale_lsj_r50_3x_coco-fe5a6829.pth"
# load_from = "work_dirs/resnet/base/epoch_4.pth"
# load_from = "work_dirs/resnet/total_v3/iter_22000.pth"
# resume = True

data_root = "data/datasets"
classes = ("person", "car", "truck", "bus", "bicycle", "bike", "extra_vehicle", "dog")
num_classes = 8
image_size = (1024, 1024)
# image_scale = (480, 480)
# model settings
mean = [139.4229080324816, 139.4229080324816, 139.4229080324816]
std = [56.34895042201581, 56.34895042201581, 56.34895042201581]

batch_augments = [dict(type="BatchFixedSizePad", size=image_size, pad_mask=True)]

model = dict(
    # use_lsj
    data_preprocessor=dict(
        type="DetDataPreprocessor",
        mean=mean,
        std=std,
        bgr_to_rgb=True,
        pad_mask=True,
        batch_augments=batch_augments,
    ),
    backbone=dict(
        # depth=101,
        frozen_stages=1,
        norm_cfg=dict(type="BN", requires_grad=False),
        # norm_cfg=dict(type="BN", requires_grad=True),
        style="pytorch",
        init_cfg=dict(type="Pretrained", checkpoint="torchvision://resnet50"),
        # init_cfg=dict(type="Pretrained", checkpoint="torchvision://resnet101"),
    ),
    query_head=dict(
        num_classes=num_classes,
        dn_cfg=dict(box_noise_scale=0.4, group_cfg=dict(num_dn_queries=500)),
        transformer=dict(encoder=dict(with_cp=6)),
    ),
)
tta_model = dict(
    type="DetTTAModel",
    tta_cfg=dict(nms=dict(type="nms", iou_threshold=0.5), max_per_img=100),
)

albu_train_transforms = [
    dict(
        type="OneOf",
        transforms=[
            dict(type="Blur", blur_limit=2, p=0.3),
            dict(type="GaussNoise", var_limit=(10.0, 20.0), p=0.3),
            dict(type="JpegCompression", quality_lower=85, quality_upper=95, p=0.3),
        ],
        p=1.0,
    ),
]


train_pipeline = [
    # dict(
    #     type="Albu",
    #     transforms=albu_train_transforms,
    #     bbox_params=dict(
    #         type="BboxParams",
    #         format="pascal_voc",  # Ensure this matches your bbox format, could be 'coco' or 'pascal_voc'
    #         label_fields=["gt_bboxes_labels", "gt_ignore_flags"],
    #         min_visibility=0.0,
    #         filter_lost_elements=True,
    #     ),
    #     keymap={"img": "image", "gt_bboxes": "bboxes"},
    #     skip_img_without_anno=True,
    # ),
    # dict(
    #     type="Mosaic",
    #     img_scale=image_size,
    #     pad_val=(114.0, 114.0, 114.0),
    #     bbox_clip_border=False,
    # ),
    # dict(
    #     type="RandomAffine",
    #     scaling_ratio_range=(0.8, 1.2),
    #     # img_scale is (width, height)
    #     border=(-image_size[0] // 2, -image_size[1] // 2),
    # ),
    # dict(
    #     type="MixUp",
    #     img_scale=image_size,
    #     ratio_range=(0.8, 1.2),
    #     pad_val=(114.0, 114.0, 114.0),
    #     bbox_clip_border=False,
    # ),
    dict(type="RandomFlip", direction="horizontal", prob=0.5),
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
                ),
            ],
            [
                dict(
                    type="RandomChoiceResize",
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
    # dict(
    #     type="Pad",
    #     pad_to_square=True,
    #     pad_val=dict(img=(114.0, 114.0, 114.0)),
    # ),
    dict(type="Pad", size=image_size, pad_val=dict(img=(114, 114, 114))),
    dict(type="FilterAnnotations", min_gt_bbox_wh=(1e-1, 1e-1)),
    dict(type="PackDetInputs"),
]

train_dataset = dict(
    type="MultiImageMixDataset",
    dataset=dict(
        type="CocoDataset",
        data_root=data_root,
        # ann_file="anno/total.json",
        # data_prefix=dict(img="total"),
        data_prefix=dict(img="extra/"),
        ann_file="anno/total_extra_2.json",
        metainfo=dict(classes=classes),
        pipeline=[
            dict(type="LoadImageFromFile", backend_args=None),
            dict(type="LoadAnnotations", with_bbox=True),
        ],
        filter_cfg=dict(filter_empty_gt=False),
        backend_args=None,
    ),
    pipeline=train_pipeline,
)

train_dataloader = dict(
    _delete_=True,
    batch_size=1,
    num_workers=3,
    persistent_workers=True,
    sampler=dict(type="DefaultSampler", shuffle=True),
    batch_sampler=dict(type="AspectRatioBatchSampler"),
    dataset=train_dataset,
)


scale = (2048, 1280)
test_pipeline = [
    dict(type="LoadImageFromFile", backend_args=None),
    dict(type="Resize", scale=scale, keep_ratio=True),
    dict(type="Pad", size=image_size, pad_val=dict(img=(114, 114, 114))),
    dict(type="LoadAnnotations", with_bbox=True),
    dict(
        type="PackDetInputs",
        meta_keys=("img_id", "img_path", "ori_shape", "img_shape", "scale_factor"),
    ),
]
#  (666, 400)
img_scales = [(2048, 1280), (1333, 800), (2000, 1200), (640, 480), (320, 240)]
tta_pipeline = [
    dict(type="LoadImageFromFile"),
    # dict(type="Pad", size=image_size, pad_val=dict(img=(114, 114, 114))),
    dict(
        type="TestTimeAug",
        transforms=[
            [dict(type="Resize", scale=s, keep_ratio=True) for s in img_scales],
            [
                # dict(type="RandomFlip", direction="horizontal", prob=1.0),
                # dict(type="RandomFlip", prob=0.5),
                dict(type="RandomFlip", direction="horizontal", prob=1.0),
                dict(type="RandomFlip", prob=0.0),
            ],
            [
                dict(
                    type="Pad",
                    pad_to_square=True,
                    pad_val=dict(img=(114.0, 114.0, 114.0)),
                ),
            ],
            [dict(type="LoadAnnotations", with_bbox=True)],
            # dict(type="Pad", size=image_size, pad_val=dict(img=(114, 114, 114))),
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
        backend_args=None,
    )
)
test_dataloader = val_dataloader

val_evaluator = dict(ann_file=data_root + "/anno/val.json")
test_evaluator = val_evaluator

optim_wrapper = dict(
    optimizer=dict(lr=5e-5),
    paramwise_cfg=dict(custom_keys={"backbone": dict(lr_mult=0.1)}),
)

# custom_hooks = [
#     dict(
#         type="PipelineSwitchHook",
#         switch_epoch=max_epochs - 1,
#         switch_pipeline=train_pipeline_2,
#     )
# ]
# default_hooks = dict(checkpoint=dict(save_last=True, interval=1, max_keep_ckpts=3))
default_hooks = dict(checkpoint=dict(by_epoch=False, interval=2000))


max_epochs = 10
num_last_epochs = 5
# max_epochs = 1
# num_last_epochs = 0

train_cfg = dict(max_epochs=max_epochs, val_interval=20)
base_lr = 5e-5

# param_scheduler = [
#     dict(
#         type="MultiStepLR",
#         begin=0,
#         end=max_epochs,
#         by_epoch=True,
#         milestones=[8],
#         gamma=0.1,
#     )
# ]

param_scheduler = [
    dict(
        # use quadratic formula to warm up 1 epochs
        type="QuadraticWarmupLR",
        by_epoch=True,
        begin=0,
        end=1,
        convert_to_iter_based=True,
    ),
    dict(
        # use cosine lr from 1 to 70 epoch
        type="CosineAnnealingLR",
        eta_min=base_lr * 0.05,
        begin=1,
        T_max=max_epochs - num_last_epochs,
        end=max_epochs - num_last_epochs,
        by_epoch=True,
        convert_to_iter_based=True,
    ),
    dict(
        # use fixed lr during last 10 epochs
        type="ConstantLR",
        by_epoch=True,
        factor=1,
        begin=max_epochs - num_last_epochs,
        end=max_epochs,
    ),
]
