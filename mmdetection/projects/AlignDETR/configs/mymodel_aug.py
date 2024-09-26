_base_ = [
    "../../../configs/_base_/datasets/coco_detection.py",
    "../../../configs/_base_/default_runtime.py",
]
max_epochs = 100
load_from = "work_dirs/alignDETR/epoch_48.pth"
resume = True


data_root = "data/all_dataset"

classes = ("person", "car", "truck", "bus", "bicycle", "bike", "extra_vehicle", "dog")
num_classes = 8
batch_size = 2
num_workers = 3
num_gpu = 4
# base_lr = 0.0001
base_lr = 0.00007

# * num_gpu
image_size = (1333, 1333)
# model settings
mean = [139.4229080324816, 139.4229080324816, 139.4229080324816]
std = [56.34895042201581, 56.34895042201581, 56.34895042201581]


custom_imports = dict(
    imports=["projects.AlignDETR.align_detr"], allow_failed_imports=False
)

model = dict(
    type="DINO",
    num_queries=900,  # num_matching_queries
    with_box_refine=True,
    as_two_stage=True,
    data_preprocessor=dict(
        type="DetDataPreprocessor",
        mean=mean,
        std=std,
        bgr_to_rgb=False,
        pad_size_divisor=1,
    ),
    backbone=dict(
        type="ResNet",
        depth=50,
        num_stages=4,
        out_indices=(1, 2, 3),
        # AlignDETR: Only freeze stem.
        frozen_stages=0,
        norm_cfg=dict(type="FrozenBN", requires_grad=False),
        norm_eval=True,
        style="pytorch",
        init_cfg=dict(type="Pretrained", checkpoint="torchvision://resnet50"),
    ),
    neck=dict(
        type="ChannelMapper",
        in_channels=[512, 1024, 2048],
        kernel_size=1,
        out_channels=256,
        # AlignDETR: Add conv bias.
        bias=True,
        act_cfg=None,
        norm_cfg=dict(type="GN", num_groups=32),
        num_outs=4,
    ),
    encoder=dict(
        num_layers=6,
        layer_cfg=dict(
            self_attn_cfg=dict(
                embed_dims=256, num_levels=4, dropout=0.0
            ),  # 0.1 for DeformDETR
            ffn_cfg=dict(
                embed_dims=256,
                feedforward_channels=2048,  # 1024 for DeformDETR
                ffn_drop=0.0,
            ),
        ),
    ),  # 0.1 for DeformDETR
    decoder=dict(
        num_layers=6,
        return_intermediate=True,
        layer_cfg=dict(
            self_attn_cfg=dict(
                embed_dims=256, num_heads=8, dropout=0.0
            ),  # 0.1 for DeformDETR
            cross_attn_cfg=dict(
                embed_dims=256, num_levels=4, dropout=0.0
            ),  # 0.1 for DeformDETR
            ffn_cfg=dict(
                embed_dims=256,
                feedforward_channels=2048,  # 1024 for DeformDETR
                ffn_drop=0.0,
            ),
        ),  # 0.1 for DeformDETR
        post_norm_cfg=None,
    ),
    positional_encoding=dict(
        num_feats=128,
        normalize=True,
        # AlignDETR: Set offset and temperature the same as DeformDETR.
        offset=-0.5,  # -0.5 for DeformDETR
        temperature=10000,
    ),  # 10000 for DeformDETR
    bbox_head=dict(
        type="AlignDETRHead",
        # AlignDETR: First 6 elements of `all_layers_num_gt_repeat` are for
        #   decoder layers' outputs. The last element is for encoder layer.
        all_layers_num_gt_repeat=[2, 2, 2, 2, 2, 1, 2],
        alpha=0.25,
        gamma=2.0,
        tau=1.5,
        num_classes=8,
        sync_cls_avg_factor=True,
        loss_cls=dict(
            type="CrossEntropyLoss", use_sigmoid=True, loss_weight=1.0
        ),  # 2.0 in DeformDETR
        loss_bbox=dict(type="L1Loss", loss_weight=5.0),
        loss_iou=dict(type="GIoULoss", loss_weight=2.0),
    ),
    dn_cfg=dict(  # TODO: Move to model.train_cfg ?
        label_noise_scale=0.5,
        box_noise_scale=1.0,  # 0.4 for DN-DETR
        group_cfg=dict(dynamic=True, num_groups=None, num_dn_queries=100),
    ),  # TODO: half num_dn_queries
    # training and testing settings
    train_cfg=dict(
        assigner=dict(
            type="MixedHungarianAssigner",
            match_costs=[
                dict(type="FocalLossCost", weight=2.0),
                dict(type="BBoxL1Cost", weight=5.0, box_format="xywh"),
                dict(type="IoUCost", iou_mode="giou", weight=2.0),
            ],
        )
    ),
    test_cfg=dict(max_per_img=300),
)  # 100 for DeformDETR

# train_pipeline, NOTE the img_scale and the Pad's size_divisor is different
# from the default setting in mmdet.


albu_train_transforms = [
    # dict(
    #     type="ShiftScaleRotate",
    #     shift_limit=0.0625,
    #     scale_limit=0.1,
    #     rotate_limit=10,
    #     interpolation=1,
    #     p=0.2,
    # ),
    # dict(type="Cutout", num_holes=5, max_h_size=80, max_w_size=80, p=0.2),
    dict(
        type="OneOf",
        transforms=[
            dict(type="Blur", blur_limit=3, p=0.2),
            # dict(type="MedianBlur", p=0.2),
            # dict(type="CLAHE", p=0.2),
            dict(type="GaussNoise", var_limit=(10.0, 30.0), p=0.2),
        ],
        p=1.0,
    ),
]


train_pipeline = [
    dict(
        type="Albu",
        transforms=albu_train_transforms,
        bbox_params=dict(
            type="BboxParams",
            format="pascal_voc",  # Ensure this matches your bbox format, could be 'coco' or 'pascal_voc'
            label_fields=["gt_bboxes_labels", "gt_ignore_flags"],
            # min_visibility=0.0,
            # filter_lost_elements=True,
        ),
        keymap={"img": "image", "gt_bboxes": "bboxes"},
        # skip_img_without_anno=True,
    ),
    dict(
        type="MixUp",
        img_scale=image_size,
        ratio_range=(0.6, 1.4),
        pad_val=(114.0, 114.0, 114.0),
        # bbox_clip_border=False,
    ),
    dict(type="RandomFlip", direction="horizontal", prob=0.5),
    dict(
        type="RandomChoice",
        transforms=[
            [
                dict(
                    type="RandomChoiceResize",
                    scales=[
                        (400, 1333),
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
                        (400, 1333),
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

train_dataset = dict(
    type="MultiImageMixDataset",
    dataset=dict(
        type="CocoDataset",
        data_root=data_root,
        data_prefix=dict(img="total/"),
        ann_file="anno/total.json",
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


test_pipeline = [
    dict(type="LoadImageFromFile"),
    dict(type="Resize", scale=(1333, 800), keep_ratio=True),
    dict(type="Pad", size=image_size, pad_val=dict(img=(114, 114, 114))),
    dict(type="LoadAnnotations", with_bbox=True),
    dict(
        type="PackDetInputs",
        meta_keys=("img_id", "img_path", "ori_shape", "img_shape", "scale_factor"),
    ),
]

val_dataloader = dict(
    batch_size=2,
    num_workers=2,
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
    type="OptimWrapper",
    optimizer=dict(
        type="AdamW", lr=base_lr, weight_decay=0.0001  # 0.0002 for DeformDETR
    ),
    clip_grad=dict(max_norm=0.1, norm_type=2),
    paramwise_cfg=dict(
        custom_keys={"backbone": dict(lr_mult=0.1)},
        # AlignDETR: No norm decay.
        norm_decay_mult=0.0,
    ),
)  # custom_keys contains sampling_offsets and reference_points in DeformDETR  # noqa

# learning policy
# train_cfg = dict(type="EpochBasedTrainLoop", max_epochs=max_epochs, val_interval=1)
# default_hooks = dict(checkpoint=dict(by_epoch=False, interval=2000))


default_hooks = dict(
    # param_scheduler=dict(
    #     type="PPYOLOEParamSchedulerHook",
    #     warmup_min_iter=1000,
    #     start_factor=0.0,
    #     warmup_epochs=5,
    #     min_lr_ratio=0.0,
    #     total_epochs=int(max_epochs * 1.2),
    # ),
    checkpoint=dict(
        type="CheckpointHook",
        interval=1,
        save_best="auto",
        max_keep_ckpts=3,
    ),
)
# max_keep_ckpts=3
train_cfg = dict(type="EpochBasedTrainLoop", max_epochs=max_epochs, val_interval=5)

val_cfg = dict(type="ValLoop")
test_cfg = dict(type="TestLoop")

param_scheduler = [
    dict(type="LinearLR", start_factor=0.0001, by_epoch=False, begin=0, end=2000),
    dict(
        type="MultiStepLR",
        begin=0,
        end=max_epochs,
        by_epoch=True,
        milestones=[100],
        gamma=0.1,
    ),
]


# NOTE: `auto_scale_lr` is for automatically scaling LR,
# USER SHOULD NOT CHANGE ITS VALUES.
# base_batch_size = (8 GPUs) x (2 samples per GPU)
auto_scale_lr = dict(base_batch_size=16)
