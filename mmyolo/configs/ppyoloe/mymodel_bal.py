_base_ = "./ppyoloe_plus_s_fast_8xb8-80e_coco.py"

# The pretrained model is geted and converted from official PPYOLOE.
# https://github.com/PaddlePaddle/PaddleDetection/blob/release/2.5/configs/ppyoloe/README.md
# load_from = "https://download.openmmlab.com/mmyolo/v0/ppyoloe/ppyoloe_pretrain/ppyoloe_plus_x_obj365_pretrained-43a8000d.pth"  # noqa
#
load_from = "work_dir/ppy/total/epoch_135.pth"
# resume = True

# deepen_factor = 1.33
# widen_factor = 1.25

deepen_factor = 1.88
widen_factor = 1.75

img_scale = (640, 640)  # width, height
# img_scale = (1024, 1024)

data_root = "data/all_dataset"
# dataset_type = "CocoDataset"
classes = ("person", "car", "truck", "bus", "bicycle", "bike", "extra_vehicle", "dog")
num_classes = 8

persistent_workers = True
max_epochs = 300

save_epoch_intervals = 5
train_batch_size_per_gpu = 4
train_num_workers = 4
val_batch_size_per_gpu = 1
val_num_workers = 2
base_lr = 0.001
# base_lr = 0.0005
# base_lr = 0.0003

dataset_type = "YOLOv5CocoDataset"

model = dict(
    backbone=dict(deepen_factor=deepen_factor, widen_factor=widen_factor),
    neck=dict(
        deepen_factor=deepen_factor,
        widen_factor=widen_factor,
    ),
    bbox_head=dict(head_module=dict(widen_factor=widen_factor)),
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
    dict(type="Cutout", num_holes=12, max_h_size=80, max_w_size=80, p=0.2),
    dict(
        type="OneOf",
        transforms=[
            dict(type="Blur", blur_limit=3, p=0.2),
            dict(type="MedianBlur", p=0.2),
            dict(type="CLAHE", p=0.2),
            dict(type="GaussNoise", var_limit=(10.0, 30.0), p=0.2),
        ],
        p=1.0,
    ),
]


train_pipeline = [
    dict(type="LoadImageFromFile", backend_args=_base_.backend_args),
    dict(type="LoadAnnotations", with_bbox=True),
    dict(
        type="mmdet.Albu",
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
# train_dataloader = dict(
#     # _delete_=True,
#     batch_size=train_batch_size_per_gpu,
#     num_workers=train_num_workers,
#     persistent_workers=persistent_workers,
#     pin_memory=True,
#     sampler=dict(type="DefaultSampler", shuffle=True),
#     batch_sampler=dict(type="mmdet.AspectRatioBatchSampler"),
#     # collate_fn=dict(type="yolov5_collate"),
#     # dataset=train_dataset,
#     dataset=dict(
#         type="mmdet.ClassBalancedDataset",
#         # type=dataset_type,
#         data_root=data_root,
#         ann_file="anno/total.json",
#         data_prefix=dict(img="total/"),
#         filter_cfg=dict(filter_empty_gt=False, min_size=0),
#         pipeline=train_pipeline,
#     ),
# )


# train_dataset = dict(
#     type="mmdet.ClassBalancedDataset",
#     oversample_thr=0.001,
#     dataset=dict(
#         # _delete_=True,
#         # type="CocoDataset",
#         data_root=data_root,
#         # ann_file="anno/total.json",
#         ann_file="anno/total.json",
#         data_prefix=dict(img="total/"),
#         metainfo=dict(classes=classes),
#         filter_cfg=dict(filter_empty_gt=True, min_size=0),
#         pipeline=train_pipeline,
#     ),
# )
train_dataset = dict(
    # dataset=dict(  # The actual dataset should handle `data_root` and `data_prefix`
    type=dataset_type,  # Ensure this is the correct dataset type (YOLOv5CocoDataset)
    data_root=data_root,
    ann_file="anno/total.json",
    data_prefix=dict(img="total/"),
    metainfo=dict(classes=classes),
    filter_cfg=dict(filter_empty_gt=False),
    backend_args=None,
    pipeline=train_pipeline,
    # ),
)

train_dataloader = dict(
    type="mmdet.ClassBalancedDataset",
    oversample_thr=0.001,
    batch_size=train_batch_size_per_gpu,
    num_workers=train_num_workers,
    persistent_workers=persistent_workers,
    pin_memory=True,
    sampler=dict(type="DefaultSampler", shuffle=True),
    batch_sampler=dict(type="mmdet.AspectRatioBatchSampler"),
    collate_fn=dict(type="yolov5_collate", use_ms_training=True),
    dataset=train_dataset,  # Reference the corrected train_dataset here
)


# train_dataloader = dict(
#     batch_size=train_batch_size_per_gpu,
#     num_workers=train_num_workers,
#     persistent_workers=persistent_workers,
#     pin_memory=True,
#     sampler=dict(type="DefaultSampler", shuffle=True),
#     batch_sampler=dict(type="mmdet.AspectRatioBatchSampler"),
#     collate_fn=dict(type="yolov5_collate", use_ms_training=True),
#     dataset=dict(
#         # type=dataset_type,
#         data_root=data_root,
#         # ann_file="anno/total.json",
#         ann_file="anno/total_bal.json",
#         data_prefix=dict(img="total/"),
#         metainfo=dict(classes=classes),
#         filter_cfg=dict(filter_empty_gt=True, min_size=0),
#         pipeline=train_pipeline,
#     ),
# )


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
        # type=dataset_type,
        data_root=data_root,
        metainfo=dict(classes=classes),
        test_mode=True,
        ann_file="anno/val.json",
        data_prefix=dict(img="val/"),
        # filter_cfg=dict(filter_empty_gt=True, min_size=0),
        pipeline=test_pipeline,
        batch_shapes_cfg=None,
    ),
)
# dataset=dict(
#     type=dataset_type,
#     data_root=data_root,
#     test_mode=True,
#     data_prefix=dict(img=val_data_prefix),
#     ann_file=val_ann_file,
#     pipeline=test_pipeline,
#     batch_shapes_cfg=batch_shapes_cfg))

test_dataloader = val_dataloader

val_evaluator = dict(
    type="mmdet.CocoMetric",
    proposal_nums=(100, 1, 10),
    ann_file=data_root + "/anno/val.json",
    # ann_file=data_root + "annotations/instances_val2017.json",
    metric="bbox",
    # backend_args=backend_args,
)
# val_evaluator = dict(ann_file=data_root + "/anno/val.json")
test_evaluator = val_evaluator

# val_evaluator = dict(
#     type="mmdet.CocoMetric",
#     proposal_nums=(100, 1, 10),
#     ann_file=data_root + val_ann_file,
#     metric="bbox",
# )
# test_evaluator = val_evaluator


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

# val_evaluator = dict(
#     type="mmdet.CocoMetric",
#     proposal_nums=(100, 1, 10),
#     ann_file=data_root + "annotations/instances_val2017.json",
#     metric="bbox",
# )
# test_evaluator = val_evaluator

train_cfg = dict(
    type="EpochBasedTrainLoop", max_epochs=max_epochs, val_interval=save_epoch_intervals
)
val_cfg = dict(type="ValLoop")
test_cfg = dict(type="TestLoop")
