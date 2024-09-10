_base_ = ["../_base_/datasets/coco_detection.py", "../_base_/default_runtime.py"]
# _base_ = "mmdet::common/ssj_scp_270k_coco-instance.py"

load_from = "checkpoints/deformable-detr-refine-twostage_r50_16xb2-50e_coco_20221021_184714-acc8a5ff.pth"
# resume = True

mean = [139.4229080324816, 139.4229080324816, 139.4229080324816]
std = [56.34895042201581, 56.34895042201581, 56.34895042201581]
image_size = (1024, 1024)

batch_augments = [dict(type="BatchFixedSizePad", size=image_size, pad_mask=True)]
base_lr = 0.0002
data_root = "data/all_dataset"
num_classes = 8
classes = ("person", "car", "truck", "bus", "bicycle", "bike", "extra_vehicle", "dog")

batch_size = 2
num_workers = 2

model = dict(
    type="DeformableDETR",
    num_queries=300,
    num_feature_levels=4,
    with_box_refine=True,
    as_two_stage=True,
    data_preprocessor=dict(
        type="DetDataPreprocessor",
        mean=mean,
        std=std,
        bgr_to_rgb=True,
        pad_mask=True,
        batch_augments=batch_augments,
    ),
    # data_preprocessor=dict(
    #     type="DetDataPreprocessor",
    #     mean=mean,
    #     std=std,
    #     bgr_to_rgb=False,
    #     pad_size_divisor=1,
    # ),
    backbone=dict(
        type="ResNet",
        depth=101,
        num_stages=4,
        out_indices=(1, 2, 3),
        frozen_stages=1,
        # deformable_groups=1,
        norm_cfg=dict(type="BN", requires_grad=False),
        norm_eval=True,
        style="pytorch",
        init_cfg=dict(type="Pretrained", checkpoint="torchvision://resnet101"),
    ),
    neck=dict(
        type="ChannelMapper",
        in_channels=[512, 1024, 2048],
        kernel_size=1,
        out_channels=256,
        act_cfg=None,
        norm_cfg=dict(type="GN", num_groups=32),
        num_outs=4,
    ),
    encoder=dict(  # DeformableDetrTransformerEncoder
        num_layers=6,
        layer_cfg=dict(  # DeformableDetrTransformerEncoderLayer
            self_attn_cfg=dict(  # MultiScaleDeformableAttention
                embed_dims=256, batch_first=True
            ),
            ffn_cfg=dict(embed_dims=256, feedforward_channels=1024, ffn_drop=0.1),
        ),
    ),
    decoder=dict(  # DeformableDetrTransformerDecoder
        num_layers=6,
        return_intermediate=True,
        layer_cfg=dict(  # DeformableDetrTransformerDecoderLayer
            self_attn_cfg=dict(  # MultiheadAttention
                embed_dims=256, num_heads=8, dropout=0.1, batch_first=True
            ),
            cross_attn_cfg=dict(  # MultiScaleDeformableAttention
                embed_dims=256, batch_first=True
            ),
            ffn_cfg=dict(embed_dims=256, feedforward_channels=1024, ffn_drop=0.1),
        ),
        post_norm_cfg=None,
    ),
    positional_encoding=dict(num_feats=128, normalize=True, offset=-0.5),
    bbox_head=dict(
        type="DeformableDETRHead",
        num_classes=80,
        sync_cls_avg_factor=True,
        loss_cls=dict(
            type="FocalLoss", use_sigmoid=True, gamma=2.0, alpha=0.25, loss_weight=2.0
        ),
        loss_bbox=dict(type="L1Loss", loss_weight=5.0),
        loss_iou=dict(type="GIoULoss", loss_weight=2.0),
    ),
    # training and testing settings
    train_cfg=dict(
        assigner=dict(
            type="HungarianAssigner",
            match_costs=[
                dict(type="FocalLossCost", weight=2.0),
                dict(type="BBoxL1Cost", weight=5.0, box_format="xywh"),
                dict(type="IoUCost", iou_mode="giou", weight=2.0),
            ],
        )
    ),
    test_cfg=dict(max_per_img=100),
)

# train_pipeline, NOTE the img_scale and the Pad's size_divisor is different
# from the default setting in mmdet.

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
    dict(type="LoadImageFromFile", backend_args={{_base_.backend_args}}),
    dict(type="LoadAnnotations", with_bbox=True),
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
    dict(
        type="RandomChoice",
        transforms=[
            [
                dict(
                    type="RandomChoiceResize",
                    scales=[
                        (480, 1333),
                        (512, 1333),
                        (544, 1333),
                        (576, 1333),
                        (608, 1333),
                        (640, 1333),
                        (672, 1333),
                        (704, 1333),
                        (736, 1333),
                        (768, 1333),
                        (800, 1333),
                        (1333, 480),
                        (1333, 512),
                        (1333, 544),
                        (1333, 576),
                        (1333, 608),
                        (1333, 640),
                        (1333, 672),
                        (1333, 704),
                        (1333, 736),
                        (1333, 768),
                        (1333, 800),
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
                        (400, 4200),
                        (500, 4200),
                        (600, 4200),
                        (4200, 400),
                        (4200, 500),
                        (4200, 600),
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
                        (480, 1333),
                        (512, 1333),
                        (544, 1333),
                        (576, 1333),
                        (608, 1333),
                        (640, 1333),
                        (672, 1333),
                        (704, 1333),
                        (736, 1333),
                        (768, 1333),
                        (800, 1333),
                        (1333, 480),
                        (1333, 512),
                        (1333, 544),
                        (1333, 576),
                        (1333, 608),
                        (1333, 640),
                        (1333, 672),
                        (1333, 704),
                        (1333, 736),
                        (1333, 768),
                        (1333, 800),
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
    # _delete_=True,
    batch_size=batch_size,
    num_workers=num_workers,
    sampler=dict(type="DefaultSampler", shuffle=True),
    # batch_sampler=dict(type="AspectRatioBatchSampler"),
    dataset=dict(
        metainfo=dict(classes=classes),
        data_root=data_root,
        data_prefix=dict(img="train/"),
        ann_file="anno/train.json",
        pipeline=train_pipeline,
    ),
)


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
# optimizer
optim_wrapper = dict(
    type="OptimWrapper",
    optimizer=dict(type="AdamW", lr=base_lr, weight_decay=0.0001),
    clip_grad=dict(max_norm=0.1, norm_type=2),
    paramwise_cfg=dict(
        custom_keys={
            "backbone": dict(lr_mult=0.1),
            "sampling_offsets": dict(lr_mult=0.1),
            "reference_points": dict(lr_mult=0.1),
        }
    ),
)
val_dataloader = dict(
    # _delete_=True,
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

# learning policy
max_epochs = 50
train_cfg = dict(type="EpochBasedTrainLoop", max_epochs=max_epochs, val_interval=100)
# train_cfg = dict(max_epochs=max_epochs, val_interval=100)
# default_hooks = dict(checkpoint=dict(by_epoch=False, interval=2000))
default_hooks = dict(checkpoint=dict(by_epoch=False, interval=2000, max_keep_ckpts=3))

# train_cfg = dict(type="EpochBasedTrainLoop", max_epochs=max_epochs, val_interval=1)
val_cfg = dict(type="ValLoop")
test_cfg = dict(type="TestLoop")


val_evaluator = dict(metric="bbox", ann_file=data_root + "/anno/val.json")
test_evaluator = val_evaluator

param_scheduler = [
    dict(
        type="MultiStepLR",
        begin=0,
        end=max_epochs,
        by_epoch=True,
        milestones=[40],
        gamma=0.1,
    )
]

# NOTE: `auto_scale_lr` is for automatically scaling LR,
# USER SHOULD NOT CHANGE ITS VALUES.
# base_batch_size = (16 GPUs) x (2 samples per GPU)
auto_scale_lr = dict(base_batch_size=32)
