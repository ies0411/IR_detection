_base_ = "./ppyoloe_plus_s_fast_8xb8-80e_coco.py"

load_from = "../weights/ppyoloe/pretrained_model.pth"

deepen_factor = 1.33
widen_factor = 1.25

img_scale = (640, 640)  # width, height

data_root = "../datasets/"
classes = ("person", "car", "truck", "bus", "bicycle", "bike", "extra_vehicle", "dog")
num_classes = 8

persistent_workers = True
max_epochs = 135

save_epoch_intervals = 5
train_batch_size_per_gpu = 4
train_num_workers = 4
val_batch_size_per_gpu = 1
val_num_workers = 2
base_lr = 0.001
# base_lr = 0.0005
# base_lr = 0.0003


model = dict(
    backbone=dict(deepen_factor=deepen_factor, widen_factor=widen_factor),
    neck=dict(
        deepen_factor=deepen_factor,
        widen_factor=widen_factor,
    ),
    bbox_head=dict(head_module=dict(widen_factor=widen_factor)),
)


train_pipeline = [
    dict(type="LoadImageFromFile", backend_args=_base_.backend_args),
    dict(type="LoadAnnotations", with_bbox=True),
    dict(type="PPYOLOERandomDistort"),
    dict(type="mmdet.Expand", mean=(103.53, 116.28, 123.675)),
    dict(type="PPYOLOERandomCrop"),
    dict(type="mmdet.RandomFlip", direction="horizontal", prob=0.5),
    dict(
        type="mmdet.PackDetInputs",
        meta_keys=(
            "img_id",
            "img_path",
            "ori_shape",
            "img_shape",
            "flip",
            "flip_direction",
        ),
    ),
]


train_dataloader = dict(
    batch_size=train_batch_size_per_gpu,
    num_workers=train_num_workers,
    persistent_workers=persistent_workers,
    pin_memory=True,
    sampler=dict(type="DefaultSampler", shuffle=True),
    batch_sampler=dict(type="mmdet.AspectRatioBatchSampler"),
    collate_fn=dict(type="yolov5_collate", use_ms_training=True),
    dataset=dict(
        data_root=data_root,
        ann_file="anno/total.json",
        data_prefix=dict(img="total/"),
        metainfo=dict(classes=classes),
        filter_cfg=dict(filter_empty_gt=True, min_size=0),
        pipeline=train_pipeline,
    ),
)


test_pipeline = [
    dict(type="LoadImageFromFile", backend_args=_base_.backend_args),
    dict(
        type="mmdet.FixShapeResize",
        width=img_scale[0],
        height=img_scale[1],
        keep_ratio=False,
        interpolation="bicubic",
    ),
    dict(type="LoadAnnotations", with_bbox=True, _scope_="mmdet"),
    dict(
        type="mmdet.PackDetInputs",
        meta_keys=("img_id", "img_path", "ori_shape", "img_shape", "scale_factor"),
    ),
]

val_dataloader = dict(
    batch_size=val_batch_size_per_gpu,
    num_workers=val_num_workers,
    persistent_workers=persistent_workers,
    pin_memory=True,
    drop_last=False,
    sampler=dict(type="DefaultSampler", shuffle=False),
    dataset=dict(
        data_root=data_root,
        metainfo=dict(classes=classes),
        test_mode=True,
        ann_file="anno/val.json",
        data_prefix=dict(img="val/"),
        pipeline=test_pipeline,
        batch_shapes_cfg=None,
    ),
)

test_dataloader = val_dataloader

val_evaluator = dict(
    type="mmdet.CocoMetric",
    proposal_nums=(100, 1, 10),
    ann_file=data_root + "/anno/val.json",
    metric="bbox",
)
test_evaluator = val_evaluator

param_scheduler = None
optim_wrapper = dict(
    type="OptimWrapper",
    optimizer=dict(
        type="SGD", lr=base_lr, momentum=0.9, weight_decay=5e-4, nesterov=False
    ),
    paramwise_cfg=dict(norm_decay_mult=0.0),
)

default_hooks = dict(
    param_scheduler=dict(
        type="PPYOLOEParamSchedulerHook",
        warmup_min_iter=1000,
        start_factor=0.0,
        warmup_epochs=5,
        min_lr_ratio=0.0,
        total_epochs=int(max_epochs * 1.2),
    ),
    checkpoint=dict(
        type="CheckpointHook",
        interval=save_epoch_intervals,
        save_best="auto",
        max_keep_ckpts=3,
    ),
)

custom_hooks = [
    dict(
        type="EMAHook",
        ema_type="ExpMomentumEMA",
        momentum=0.0002,
        update_buffers=True,
        strict_load=False,
        priority=49,
    )
]


train_cfg = dict(
    type="EpochBasedTrainLoop", max_epochs=max_epochs, val_interval=save_epoch_intervals
)
val_cfg = dict(type="ValLoop")
test_cfg = dict(type="TestLoop")
