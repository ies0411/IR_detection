_base_ = ["co_dino_5scale_r50_8xb2_1x_coco.py"]


load_from = "../weights/codetr_swin/final_1.pth"
data_root = "../datasets/"
num_workers = 2
batch_size = 1
classes = ("person", "car", "truck", "bus", "bicycle", "bike", "extra_vehicle", "dog")
num_classes = 8
image_size = (1024, 1024)
base_lr = 5e-5


mean = [139.4229080324816, 139.4229080324816, 139.4229080324816]
std = [56.34895042201581, 56.34895042201581, 56.34895042201581]
# model settings
max_epochs = 5

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
        init_cfg=None,
        # init_cfg=dict(type="Pretrained", checkpoint=pretrained),
    ),
    neck=dict(in_channels=[192, 384, 768, 1536]),
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

img_norm_cfg = dict(
    mean=[139.4229080324816, 139.4229080324816, 139.4229080324816],
    std=[56.34895042201581, 56.34895042201581, 56.34895042201581],
)
albu_train_transforms = [
    dict(
        type="OneOf",
        transforms=[
            dict(type="Blur", blur_limit=1, p=0.2),
            dict(type="GaussNoise", var_limit=(10.0, 20.0), p=0.2),
        ],
        p=0.5,
    ),
]

train_pipeline = [
    dict(
        type="Mosaic",
        img_scale=image_size,
        pad_val=(114.0, 114.0, 114.0),
        # bbox_clip_border=False,
    ),
    dict(type="RandomFlip", prob=0.5, direction="horizontal"),
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
    dict(type="FilterAnnotations", min_gt_bbox_wh=(1e-2, 1e-2)),
    dict(type="PackDetInputs"),
]


train_pipeline_2 = [
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
    dict(type="FilterAnnotations", min_gt_bbox_wh=(1e-2, 1e-2)),
    dict(type="PackDetInputs"),
]
train_dataset = dict(
    type="MultiImageMixDataset",
    dataset=dict(
        type="CocoDataset",
        data_root=data_root,
        data_prefix=dict(img="total/"),
        ann_file="anno/total_pseudo.json",
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
    batch_size=batch_size,
    num_workers=num_workers,
    persistent_workers=True,
    sampler=dict(type="DefaultSampler", shuffle=True),
    batch_sampler=dict(type="AspectRatioBatchSampler"),
    dataset=train_dataset,
)

scale = (2048, 1280)

test_pipeline = [
    dict(type="LoadImageFromFile"),
    dict(type="Resize", scale=scale, keep_ratio=True),
    dict(type="LoadAnnotations", with_bbox=True),
    dict(type="Pad", size=image_size, pad_val=dict(img=(114, 114, 114))),
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
                dict(type="RandomFlip", direction="horizontal", prob=0.0),
            ],
            [dict(type="Pad", size=image_size, pad_val=dict(img=(114, 114, 114)))],
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
default_hooks = dict(
    checkpoint=dict(
        type="CheckpointHook",
        interval=2000,
        by_epoch=False,
        save_best="auto",
        max_keep_ckpts=5,
    ),
)
test_dataloader = val_dataloader

val_evaluator = dict(ann_file=data_root + "/anno/val.json")
test_evaluator = val_evaluator


optim_wrapper = dict(optimizer=dict(lr=base_lr))


train_cfg = dict(max_epochs=max_epochs, val_interval=1)

# param_scheduler = [
#     dict(
#         type="MultiStepLR",
#         begin=0,
#         end=max_epochs,
#         by_epoch=True,
#         milestones=[5],
#         gamma=0.1,
#     )
# ]

iters_per_epoch = 10000 // batch_size  # Calculate based on your dataset and batch size

# Set the total number of iterations for T_max
total_iters = (
    max_epochs * iters_per_epoch
)  # Total number of iterations in the entire training

param_scheduler = [
    dict(
        type="CosineAnnealingLR",  # Use CosineAnnealingLR
        T_max=total_iters,  # Number of iterations for one complete cosine cycle
        eta_min=5e-5,  # Minimum learning rate at the end of the cycle
        begin=0,  # Start at the first iteration
        end=total_iters,  # End at the last iteration of the training
        by_epoch=False,  # Scheduler updates by iteration, not by epoch
    )
]
