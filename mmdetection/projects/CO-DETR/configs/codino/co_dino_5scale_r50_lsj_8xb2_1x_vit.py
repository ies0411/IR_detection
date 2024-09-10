_base_ = ["co_dino_5scale_r50_8xb2_1x_coco.py"]
# 분포도 확인 -> aug줄이기
# load_from = "checkpoints/co_dino_5scale_lsj_r50_3x_coco-fe5a6829.pth"
checkpoint = "checkpoints/mae_pretrain_vit_base.pth"
load_from = "work_dirs/vit/base/iter_4000.pth"
resume = True

data_root = "data/all_dataset"
classes = ("person", "car", "truck", "bus", "bicycle", "bike", "extra_vehicle", "dog")
num_classes = 8
image_size = (1024, 1024)
batch_size = 1
num_workers = 1
max_epochs = 15
# image_size = (1536, 1536)
# image_size = (2048, 2048)
base_lr = 1e-4
lr_mult = 0.01
# model settings
mean = [139.4229080324816, 139.4229080324816, 139.4229080324816]
std = [56.34895042201581, 56.34895042201581, 56.34895042201581]

backbone_norm_cfg = dict(type="LN", requires_grad=False)
norm_cfg = dict(type="LN2d", requires_grad=True)
image_size = (1024, 1024)
batch_augments = [dict(type="BatchFixedSizePad", size=image_size, pad_mask=True)]

model = dict(
    data_preprocessor=dict(pad_size_divisor=32, batch_augments=batch_augments),
    backbone=dict(
        _delete_=True,
        type="ViT",
        img_size=1024,
        patch_size=16,
        embed_dim=768,
        depth=12,
        num_heads=12,
        drop_path_rate=0.1,
        window_size=14,
        mlp_ratio=4,
        qkv_bias=True,
        norm_cfg=backbone_norm_cfg,
        window_block_indexes=[
            0,
            1,
            3,
            4,
            6,
            7,
            9,
            10,
        ],
        use_rel_pos=True,
        init_cfg=dict(type="Pretrained", checkpoint=checkpoint),
        # init_cfg=None,
    ),
    # neck=dict(
    #     type="ChannelMapper",
    #     in_channels=[192, 384, 768, 768],
    #     # in_channels=[256, 512, 1024, 2048],
    #     kernel_size=1,
    #     out_channels=256,
    #     act_cfg=None,
    #     norm_cfg=dict(type="GN", num_groups=32),
    #     num_outs=5,
    # ),
    neck=dict(
        _delete_=True,
        type="SimpleFPN",
        backbone_channel=768,
        in_channels=[192, 384, 768, 768],
        out_channels=256,
        num_outs=5,
        norm_cfg=norm_cfg,
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
        type="ShiftScaleRotate",
        shift_limit=0.0625,
        scale_limit=0.1,
        rotate_limit=5,
        interpolation=1,
        p=0.3,
    ),
    dict(
        type="OneOf",
        transforms=[
            # dict(type="GridMask", rotate=15, num_grid=3, p=0.3),
            dict(type="Cutout", num_holes=5, max_h_size=120, max_w_size=120, p=0.3),
        ],
        p=1.0,
    ),
    dict(
        type="OneOf",
        transforms=[
            dict(type="Blur", blur_limit=1, p=0.3),
            dict(type="GaussNoise", var_limit=(10.0, 20.0), p=0.3),
            # dict(
            #     type="RandomBrightnessContrast",
            #     brightness_limit=[0.1, 0.3],
            #     contrast_limit=[0.1, 0.3],
            #     p=0.2,
            # ),
        ],
        p=1.0,
    ),
]


train_pipeline = [
    dict(type="LoadImageFromFile"),
    dict(type="LoadAnnotations", with_bbox=True),
    # dict(type="CopyPaste", max_num_pasted=100),
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
    dict(type="RandomFlip", prob=0.5),
    #  direction="horizontal"
    dict(
        type="RandomChoice",
        transforms=[
            [
                dict(
                    type="RandomChoiceResize",
                    scales=[
                        (480, 1024),
                        (544, 1024),
                        (576, 1024),
                        (640, 1024),
                        (640, 480),
                        (704, 1024),
                        (736, 1024),
                        (800, 1024),
                        (1024, 1024),
                        (1024, 480),
                        (1024, 544),
                        (1024, 576),
                        (1024, 640),
                        (1024, 640),
                        (1024, 704),
                        (1024, 736),
                        (1024, 800),
                        # (1024, 1024),
                    ],
                    keep_ratio=True,
                )
            ],
            [
                dict(
                    type="RandomChoiceResize",
                    # The radio of all image in train dataset < 7
                    # follow the original implement
                    scales=[
                        (400, 1024),
                        (500, 1024),
                        (600, 1024),
                        (1024, 400),
                        (1024, 500),
                        (1024, 600),
                    ],
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
                        (480, 1024),
                        (512, 1024),
                        (544, 1024),
                        (800, 1024),
                        (832, 1024),
                        (1024, 1024),
                        (1024, 480),
                        (1024, 512),
                        (1024, 544),
                        (1024, 800),
                        (1024, 832),
                        # (1024, 1024),
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

train_dataloader = dict(
    batch_size=batch_size,
    num_workers=num_workers,
    sampler=dict(type="DefaultSampler", shuffle=True),
    batch_sampler=dict(type="AspectRatioBatchSampler"),
    dataset=dict(
        metainfo=dict(classes=classes),
        data_root=data_root,
        # data_prefix=dict(img="total/"),
        # ann_file="anno/total.json",
        data_prefix=dict(img="train/"),
        ann_file="anno/train.json",
        pipeline=train_pipeline,
    ),
)
# image_size = (1024, 1024)

scale = (1024, 1024)
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

# img_scales = [(2048, 1280), (1333, 800), (666, 400), (2000, 1200), (640, 480)]
img_scales = [(2048, 1280), (1333, 800), (2000, 1200), (640, 480), (320, 240)]

# img_scales = [(2048, 1280), (1333, 800), (666, 400), (2000, 1200), (640, 480)]

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
                    type="Pad",
                    pad_to_square=True,
                    pad_val=dict(img=(114.0, 114.0, 114.0)),
                ),
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
# optim_wrapper = dict(optimizer=dict(lr=1e-4))
optim_wrapper = dict(
    optimizer=dict(lr=base_lr),
    paramwise_cfg=dict(custom_keys={"backbone": dict(lr_mult=lr_mult)}),
)

# optim_wrapper = dict(
#     _delete_=True,
#     type="OptimWrapper",
#     optimizer=dict(type="AdamW", lr=1e-4, weight_decay=0.0001),
#     clip_grad=dict(max_norm=0.1, norm_type=2),
#     paramwise_cfg=dict(custom_keys={"backbone": dict(lr_mult=0.1)}),
# )


# img_scale = (640, 480)
train_pipeline_2 = [
    dict(type="LoadImageFromFile"),
    dict(type="LoadAnnotations", with_bbox=True),
    dict(type="PackDetInputs"),
]

# custom_hooks = [
#     dict(
#         type="PipelineSwitchHook",
#         switch_epoch=max_epochs - 1,
#         switch_pipeline=train_pipeline_2,
#     )
# ]
# default_hooks = dict(checkpoint=dict(save_last=True, interval=1, max_keep_ckpts=3))
default_hooks = dict(checkpoint=dict(by_epoch=False, interval=2000))


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
