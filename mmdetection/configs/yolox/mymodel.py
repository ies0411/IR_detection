_base_ = [
    "../_base_/schedules/schedule_1x.py",
    "../_base_/default_runtime.py",
    "./yolox_tta.py",
]

load_from = "work_dirs/yolox/total/epoch_240.pth"
# load_from = "work_dirs/yolox/extra/epoch_290.pth"
# resume = True

img_scale = (640, 640)  # width, height
save_epoch_intervals = 5
data_root = "data/all_dataset"
dataset_type = "CocoDataset"
classes = ("person", "car", "truck", "bus", "bicycle", "bike", "extra_vehicle", "dog")
num_classes = 8
base_lr = 0.01
batch_size = 4

# model settings
# mean = [139.4229080324816, 139.4229080324816, 139.4229080324816]
# std = [56.34895042201581, 56.34895042201581, 56.34895042201581]

# batch_augments = [dict(type="BatchFixedSizePad", size=image_size, pad_mask=True)]

model = dict(
    type="YOLOX",
    data_preprocessor=dict(
        type="DetDataPreprocessor",
        pad_size_divisor=32,
        batch_augments=[
            dict(
                type="BatchSyncRandomResize",
                random_size_range=(480, 800),
                size_divisor=32,
                interval=10,
            )
        ],
    ),
    backbone=dict(
        type="CSPDarknet",
        deepen_factor=1.33,
        widen_factor=1.25,
        out_indices=(2, 3, 4),
        use_depthwise=False,
        spp_kernal_sizes=(5, 9, 13),
        norm_cfg=dict(type="BN", momentum=0.03, eps=0.001),
        act_cfg=dict(type="Swish"),
    ),
    neck=dict(
        type="YOLOXPAFPN",
        in_channels=[320, 640, 1280],
        out_channels=320,
        num_csp_blocks=4,
        use_depthwise=False,
        upsample_cfg=dict(scale_factor=2, mode="nearest"),
        norm_cfg=dict(type="BN", momentum=0.03, eps=0.001),
        act_cfg=dict(type="Swish"),
    ),
    bbox_head=dict(
        type="YOLOXHead",
        num_classes=num_classes,
        in_channels=320,
        feat_channels=320,
        stacked_convs=2,
        strides=(8, 16, 32),
        use_depthwise=False,
        norm_cfg=dict(type="BN", momentum=0.03, eps=0.001),
        act_cfg=dict(type="Swish"),
        loss_cls=dict(
            type="CrossEntropyLoss", use_sigmoid=True, reduction="sum", loss_weight=1.0
        ),
        loss_bbox=dict(
            type="IoULoss", mode="square", eps=1e-16, reduction="sum", loss_weight=5.0
        ),
        loss_obj=dict(
            type="CrossEntropyLoss", use_sigmoid=True, reduction="sum", loss_weight=1.0
        ),
        loss_l1=dict(type="L1Loss", reduction="sum", loss_weight=1.0),
    ),
    train_cfg=dict(assigner=dict(type="SimOTAAssigner", center_radius=2.5)),
    # In order to align the source code, the threshold of the val phase is
    # 0.01, and the threshold of the test phase is 0.001.
    test_cfg=dict(score_thr=0.01, nms=dict(type="nms", iou_threshold=0.65)),
)


backend_args = None

# mean = [139.4229080324816, 139.4229080324816, 139.4229080324816]
# std = [56.34895042201581, 56.34895042201581, 56.34895042201581]


# img_norm_cfg = dict(mean=mean, std=std, to_rgb=False)

train_pipeline = [
    dict(type="Mosaic", img_scale=img_scale, pad_val=114.0),
    dict(
        type="RandomAffine",
        scaling_ratio_range=(0.1, 2),
        # img_scale is (width, height)
        border=(-img_scale[0] // 2, -img_scale[1] // 2),
    ),
    dict(type="MixUp", img_scale=img_scale, ratio_range=(0.8, 1.6), pad_val=114.0),
    dict(type="YOLOXHSVRandomAug"),
    dict(type="RandomFlip", direction="horizontal", prob=0.5),
    # According to the official implementation, multi-scale
    # training is not considered here but in the
    # 'mmdet/models/detectors/yolox.py'.
    # Resize and Pad are for the last 15 epochs when Mosaic,
    # RandomAffine, and MixUp are closed by YOLOXModeSwitchHook.
    dict(type="Resize", scale=img_scale, keep_ratio=True),
    dict(
        type="Pad",
        pad_to_square=True,
        # If the image is three-channel, the pad value needs
        # to be set separately for each channel.
        pad_val=dict(img=(114.0, 114.0, 114.0)),
    ),
    dict(type="FilterAnnotations", min_gt_bbox_wh=(1, 1), keep_empty=False),
    dict(type="PackDetInputs"),
]

train_dataset = dict(
    # use MultiImageMixDataset wrapper to support mosaic and mixup
    type="MultiImageMixDataset",
    dataset=dict(
        type=dataset_type,
        data_root=data_root,
        ann_file="anno/total.json",
        data_prefix=dict(img="total/"),
        metainfo=dict(classes=classes),
        pipeline=[
            dict(type="LoadImageFromFile", backend_args=backend_args),
            dict(type="LoadAnnotations", with_bbox=True),
        ],
        filter_cfg=dict(filter_empty_gt=False, min_size=32),
        backend_args=backend_args,
    ),
    pipeline=train_pipeline,
)

test_pipeline = [
    dict(type="LoadImageFromFile", backend_args=backend_args),
    dict(type="Resize", scale=img_scale, keep_ratio=True),
    dict(type="Pad", pad_to_square=True, pad_val=dict(img=(114.0, 114.0, 114.0))),
    dict(type="LoadAnnotations", with_bbox=True),
    dict(
        type="PackDetInputs",
        meta_keys=("img_id", "img_path", "ori_shape", "img_shape", "scale_factor"),
    ),
]

train_dataloader = dict(
    batch_size=batch_size,
    num_workers=4,
    persistent_workers=True,
    sampler=dict(type="DefaultSampler", shuffle=True),
    batch_sampler=dict(type="AspectRatioBatchSampler"),
    dataset=train_dataset,
)
val_dataloader = dict(
    batch_size=batch_size,
    num_workers=4,
    persistent_workers=True,
    drop_last=False,
    sampler=dict(type="DefaultSampler", shuffle=False),
    dataset=dict(
        type=dataset_type,
        data_root=data_root,
        metainfo=dict(classes=classes),
        ann_file="anno/val.json",
        data_prefix=dict(img="val/"),
        test_mode=True,
        pipeline=test_pipeline,
        backend_args=backend_args,
    ),
)
test_dataloader = val_dataloader

val_evaluator = dict(
    type="CocoMetric",
    ann_file=data_root + "/anno/val.json",
    # ann_file=data_root + "annotations/instances_val2017.json",
    metric="bbox",
    backend_args=backend_args,
)
# val_evaluator = dict(ann_file=data_root + "/anno/val.json")
test_evaluator = val_evaluator

# training settings
max_epochs = 300
num_last_epochs = 15
interval = 10

train_cfg = dict(max_epochs=max_epochs, val_interval=interval)

# optimizer
# default 8 gpu
optim_wrapper = dict(
    type="OptimWrapper",
    optimizer=dict(
        type="SGD", lr=base_lr, momentum=0.9, weight_decay=5e-4, nesterov=True
    ),
    paramwise_cfg=dict(norm_decay_mult=0.0, bias_decay_mult=0.0),
)
# optim_wrapper = dict(
#     _delete_=True,
#     type="OptimWrapper",
#     optimizer=dict(type="AdamW", lr=base_lr, weight_decay=0.0001),
#     clip_grad=dict(max_norm=0.1, norm_type=2),
#     paramwise_cfg=dict(custom_keys={"backbone": dict(lr_mult=0.1)}),
# )
# # learning rate
param_scheduler = [
    dict(
        # use quadratic formula to warm up 5 epochs
        # and lr is updated by iteration
        # TODO: fix default scope in get function
        type="mmdet.QuadraticWarmupLR",
        by_epoch=True,
        begin=0,
        end=5,
        convert_to_iter_based=True,
    ),
    dict(
        # use cosine lr from 5 to 285 epoch
        type="CosineAnnealingLR",
        eta_min=base_lr * 0.05,
        begin=5,
        T_max=max_epochs - num_last_epochs,
        end=max_epochs - num_last_epochs,
        by_epoch=True,
        convert_to_iter_based=True,
    ),
    dict(
        # use fixed lr during last 15 epochs
        type="ConstantLR",
        by_epoch=True,
        factor=1,
        begin=max_epochs - num_last_epochs,
        end=max_epochs,
    ),
]

default_hooks = dict(
    # checkpoint=dict(
    #     interval=interval, max_keep_ckpts=3  # only keep latest 3 checkpoints
    # )
    checkpoint=dict(
        type="CheckpointHook",
        interval=save_epoch_intervals,
        save_best="auto",
        max_keep_ckpts=2,
    ),
)
# default_hooks = dict(
#     param_scheduler=dict(
#         type="PPYOLOEParamSchedulerHook",
#         warmup_min_iter=1000,
#         start_factor=0.0,
#         warmup_epochs=5,
#         min_lr_ratio=0.0,
#         total_epochs=int(max_epochs * 1.2),
#     ),
#     checkpoint=dict(
#         type="CheckpointHook",
#         interval=save_epoch_intervals,
#         save_best="auto",
#         max_keep_ckpts=3,
#     ),
# )

custom_hooks = [
    dict(type="YOLOXModeSwitchHook", num_last_epochs=num_last_epochs, priority=48),
    dict(type="SyncNormHook", priority=48),
    dict(
        type="EMAHook",
        ema_type="ExpMomentumEMA",
        momentum=0.0001,
        update_buffers=True,
        priority=49,
    ),
]

# NOTE: `auto_scale_lr` is for automatically scaling LR,
# USER SHOULD NOT CHANGE ITS VALUES.
# base_batch_size = (8 GPUs) x (8 samples per GPU)
auto_scale_lr = dict(base_batch_size=64)
