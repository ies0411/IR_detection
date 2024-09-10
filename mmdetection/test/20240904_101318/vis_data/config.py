default_scope = 'mmdet'
default_hooks = dict(
    timer=dict(type='IterTimerHook', _scope_='mmdet'),
    logger=dict(type='LoggerHook', interval=50, _scope_='mmdet'),
    param_scheduler=dict(type='ParamSchedulerHook', _scope_='mmdet'),
    checkpoint=dict(
        type='CheckpointHook',
        interval=2000,
        by_epoch=False,
        _scope_='mmdet',
        max_keep_ckpts=3),
    sampler_seed=dict(type='DistSamplerSeedHook', _scope_='mmdet'),
    visualization=dict(type='DetVisualizationHook', _scope_='mmdet'))
env_cfg = dict(
    cudnn_benchmark=False,
    mp_cfg=dict(mp_start_method='fork', opencv_num_threads=0),
    dist_cfg=dict(backend='nccl'))
vis_backends = [dict(type='LocalVisBackend', _scope_='mmdet')]
visualizer = dict(
    type='DetLocalVisualizer',
    vis_backends=[dict(type='LocalVisBackend')],
    name='visualizer',
    _scope_='mmdet')
log_processor = dict(
    type='LogProcessor', window_size=50, by_epoch=True, _scope_='mmdet')
log_level = 'INFO'
load_from = 'checkpoints/co_dino_5scale_lsj_r50_3x_coco-fe5a6829.pth'
resume = False
dataset_type = 'CocoDataset'
data_root = 'data/datasets'
image_size = (1024, 1024)
backend_args = None
train_pipeline = [
    dict(type='LoadImageFromFile'),
    dict(type='LoadAnnotations', with_bbox=True),
    dict(
        type='Albu',
        transforms=[
            dict(
                type='ShiftScaleRotate',
                shift_limit=0.0625,
                scale_limit=0.1,
                rotate_limit=5,
                interpolation=1,
                p=0.1),
            dict(
                type='OneOf',
                transforms=[
                    dict(type='Blur', blur_limit=1, p=0.2),
                    dict(type='GaussNoise', var_limit=(10.0, 20.0), p=0.2)
                ],
                p=0.5)
        ],
        bbox_params=dict(
            type='BboxParams',
            format='pascal_voc',
            label_fields=['gt_bboxes_labels', 'gt_ignore_flags'],
            min_visibility=0.0,
            filter_lost_elements=True),
        keymap=dict(img='image', gt_bboxes='bboxes'),
        skip_img_without_anno=True),
    dict(type='RandomFlip', prob=0.3, direction='horizontal'),
    dict(
        type='RandomChoice',
        transforms=[[{
            'type':
            'RandomChoiceResize',
            'scales': [(480, 2048), (512, 2048), (544, 2048), (576, 2048),
                       (608, 2048), (640, 2048), (672, 2048), (704, 2048),
                       (736, 2048), (768, 2048), (800, 2048), (832, 2048),
                       (864, 2048), (896, 2048), (928, 2048), (960, 2048),
                       (992, 2048), (1024, 2048), (1056, 2048), (1088, 2048),
                       (1120, 2048), (1152, 2048), (1184, 2048), (1216, 2048),
                       (1248, 2048), (1280, 2048), (1312, 2048), (1344, 2048),
                       (1376, 2048), (1408, 2048), (1440, 2048), (1472, 2048),
                       (1504, 2048), (1536, 2048)],
            'keep_ratio':
            True
        }],
                    [{
                        'type': 'RandomChoiceResize',
                        'scales': [(400, 4200), (500, 4200), (600, 4200)],
                        'keep_ratio': True
                    }, {
                        'type': 'RandomCrop',
                        'crop_type': 'absolute_range',
                        'crop_size': (384, 600),
                        'allow_negative_crop': True
                    }, {
                        'type':
                        'RandomChoiceResize',
                        'scales': [(480, 2048), (512, 2048), (544, 2048),
                                   (576, 2048), (608, 2048), (640, 2048),
                                   (672, 2048), (704, 2048), (736, 2048),
                                   (768, 2048), (800, 2048), (832, 2048),
                                   (864, 2048), (896, 2048), (928, 2048),
                                   (960, 2048), (992, 2048), (1024, 2048),
                                   (1056, 2048), (1088, 2048), (1120, 2048),
                                   (1152, 2048), (1184, 2048), (1216, 2048),
                                   (1248, 2048), (1280, 2048), (1312, 2048),
                                   (1344, 2048), (1376, 2048), (1408, 2048),
                                   (1440, 2048), (1472, 2048), (1504, 2048),
                                   (1536, 2048)],
                        'keep_ratio':
                        True
                    }]]),
    dict(type='PackDetInputs')
]
test_pipeline = [
    dict(type='LoadImageFromFile'),
    dict(type='Resize', scale=(2048, 1280), keep_ratio=True),
    dict(type='LoadAnnotations', with_bbox=True),
    dict(
        type='PackDetInputs',
        meta_keys=('img_id', 'img_path', 'ori_shape', 'img_shape',
                   'scale_factor'))
]
train_dataloader = dict(
    batch_size=1,
    num_workers=1,
    persistent_workers=True,
    sampler=dict(type='DefaultSampler', _scope_='mmdet', shuffle=True),
    dataset=dict(
        type='CocoDataset',
        data_root='data/datasets',
        ann_file='anno/total_extra_2.json',
        data_prefix=dict(img='extra/'),
        filter_cfg=dict(filter_empty_gt=False, min_size=32),
        pipeline=[
            dict(type='LoadImageFromFile'),
            dict(type='LoadAnnotations', with_bbox=True),
            dict(
                type='Albu',
                transforms=[
                    dict(
                        type='ShiftScaleRotate',
                        shift_limit=0.0625,
                        scale_limit=0.1,
                        rotate_limit=5,
                        interpolation=1,
                        p=0.1),
                    dict(
                        type='OneOf',
                        transforms=[
                            dict(type='Blur', blur_limit=1, p=0.2),
                            dict(
                                type='GaussNoise',
                                var_limit=(10.0, 20.0),
                                p=0.2)
                        ],
                        p=0.5)
                ],
                bbox_params=dict(
                    type='BboxParams',
                    format='pascal_voc',
                    label_fields=['gt_bboxes_labels', 'gt_ignore_flags'],
                    min_visibility=0.0,
                    filter_lost_elements=True),
                keymap=dict(img='image', gt_bboxes='bboxes'),
                skip_img_without_anno=True),
            dict(type='RandomFlip', prob=0.3, direction='horizontal'),
            dict(
                type='RandomChoice',
                transforms=[[{
                    'type':
                    'RandomChoiceResize',
                    'scales': [(480, 2048), (512, 2048), (544, 2048),
                               (576, 2048), (608, 2048), (640, 2048),
                               (672, 2048), (704, 2048), (736, 2048),
                               (768, 2048), (800, 2048), (832, 2048),
                               (864, 2048), (896, 2048), (928, 2048),
                               (960, 2048), (992, 2048), (1024, 2048),
                               (1056, 2048), (1088, 2048), (1120, 2048),
                               (1152, 2048), (1184, 2048), (1216, 2048),
                               (1248, 2048), (1280, 2048), (1312, 2048),
                               (1344, 2048), (1376, 2048), (1408, 2048),
                               (1440, 2048), (1472, 2048), (1504, 2048),
                               (1536, 2048)],
                    'keep_ratio':
                    True
                }],
                            [{
                                'type': 'RandomChoiceResize',
                                'scales': [(400, 4200), (500, 4200),
                                           (600, 4200)],
                                'keep_ratio': True
                            }, {
                                'type': 'RandomCrop',
                                'crop_type': 'absolute_range',
                                'crop_size': (384, 600),
                                'allow_negative_crop': True
                            }, {
                                'type':
                                'RandomChoiceResize',
                                'scales':
                                [(480, 2048), (512, 2048), (544, 2048),
                                 (576, 2048), (608, 2048), (640, 2048),
                                 (672, 2048), (704, 2048), (736, 2048),
                                 (768, 2048), (800, 2048), (832, 2048),
                                 (864, 2048), (896, 2048), (928, 2048),
                                 (960, 2048), (992, 2048), (1024, 2048),
                                 (1056, 2048), (1088, 2048), (1120, 2048),
                                 (1152, 2048), (1184, 2048), (1216, 2048),
                                 (1248, 2048), (1280, 2048), (1312, 2048),
                                 (1344, 2048), (1376, 2048), (1408, 2048),
                                 (1440, 2048), (1472, 2048), (1504, 2048),
                                 (1536, 2048)],
                                'keep_ratio':
                                True
                            }]]),
            dict(type='PackDetInputs')
        ],
        backend_args=None,
        metainfo=dict(
            classes=('person', 'car', 'truck', 'bus', 'bicycle', 'bike',
                     'extra_vehicle', 'dog'))),
    batch_sampler=dict(type='AspectRatioBatchSampler'))
val_dataloader = dict(
    batch_size=1,
    num_workers=2,
    persistent_workers=True,
    drop_last=False,
    sampler=dict(type='DefaultSampler', shuffle=False, _scope_='mmdet'),
    dataset=dict(
        type='CocoDataset',
        data_root='data/datasets',
        ann_file='anno/val.json',
        data_prefix=dict(img='val/'),
        test_mode=True,
        pipeline=[
            dict(type='LoadImageFromFile'),
            dict(type='Resize', scale=(2048, 1280), keep_ratio=True),
            dict(type='LoadAnnotations', with_bbox=True),
            dict(
                type='PackDetInputs',
                meta_keys=('img_id', 'img_path', 'ori_shape', 'img_shape',
                           'scale_factor'))
        ],
        backend_args=None,
        _scope_='mmdet',
        metainfo=dict(
            classes=('person', 'car', 'truck', 'bus', 'bicycle', 'bike',
                     'extra_vehicle', 'dog'))))
test_dataloader = dict(
    batch_size=1,
    num_workers=2,
    persistent_workers=True,
    drop_last=False,
    sampler=dict(type='DefaultSampler', shuffle=False, _scope_='mmdet'),
    dataset=dict(
        type='CocoDataset',
        data_root='data/datasets',
        ann_file='anno/val.json',
        data_prefix=dict(img='val/'),
        test_mode=True,
        pipeline=[
            dict(type='LoadImageFromFile'),
            dict(type='Resize', scale=(2048, 1280), keep_ratio=True),
            dict(type='LoadAnnotations', with_bbox=True),
            dict(
                type='PackDetInputs',
                meta_keys=('img_id', 'img_path', 'ori_shape', 'img_shape',
                           'scale_factor'))
        ],
        backend_args=None,
        _scope_='mmdet',
        metainfo=dict(
            classes=('person', 'car', 'truck', 'bus', 'bicycle', 'bike',
                     'extra_vehicle', 'dog'))))
val_evaluator = dict(
    type='CocoMetric',
    ann_file='data/datasets/anno/val.json',
    metric='bbox',
    format_only=False,
    backend_args=None,
    _scope_='mmdet')
test_evaluator = dict(
    type='CocoMetric',
    ann_file='data/datasets/anno/val.json',
    metric='bbox',
    format_only=False,
    backend_args=None,
    _scope_='mmdet')
max_iters = 270000
train_cfg = dict(type='EpochBasedTrainLoop', max_epochs=15, val_interval=20)
val_cfg = dict(type='ValLoop', _scope_='mmdet')
test_cfg = dict(type='TestLoop', _scope_='mmdet')
optim_wrapper = dict(
    type='OptimWrapper',
    optimizer=dict(type='AdamW', lr=0.0001, weight_decay=0.0001),
    clip_grad=dict(max_norm=0.1, norm_type=2),
    paramwise_cfg=dict(custom_keys=dict(backbone=dict(lr_mult=0.1))))
param_scheduler = [
    dict(
        type='MultiStepLR',
        begin=0,
        end=15,
        by_epoch=True,
        milestones=[8],
        gamma=0.1)
]
auto_scale_lr = dict(base_batch_size=16)
load_pipeline = [
    dict(type='LoadImageFromFile'),
    dict(type='LoadAnnotations', with_bbox=True, with_mask=True),
    dict(
        type='RandomResize',
        scale=(1024, 1024),
        ratio_range=(0.1, 2.0),
        keep_ratio=True),
    dict(
        type='RandomCrop',
        crop_type='absolute_range',
        crop_size=(1024, 1024),
        recompute_bbox=True,
        allow_negative_crop=True),
    dict(type='FilterAnnotations', min_gt_bbox_wh=(0.01, 0.01)),
    dict(type='RandomFlip', prob=0.5),
    dict(type='Pad', size=(1024, 1024), pad_val=dict(img=(114, 114, 114)))
]
custom_imports = dict(
    imports=['projects.CO-DETR.codetr'], allow_failed_imports=False)
num_dec_layer = 6
loss_lambda = 2.0
num_classes = 8
batch_augments = [
    dict(type='BatchFixedSizePad', size=(1024, 1024), pad_mask=True)
]
model = dict(
    type='CoDETR',
    use_lsj=False,
    eval_module='detr',
    data_preprocessor=dict(
        type='DetDataPreprocessor',
        mean=[139.4229080324816, 139.4229080324816, 139.4229080324816],
        std=[56.34895042201581, 56.34895042201581, 56.34895042201581],
        bgr_to_rgb=True,
        pad_mask=False,
        batch_augments=None),
    backbone=dict(
        type='SwinTransformer',
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
        with_cp=True,
        convert_weights=True),
    neck=dict(
        type='ChannelMapper',
        in_channels=[192, 384, 768, 1536],
        kernel_size=1,
        out_channels=256,
        act_cfg=None,
        norm_cfg=dict(type='GN', num_groups=32),
        num_outs=5),
    query_head=dict(
        type='CoDINOHead',
        num_query=900,
        num_classes=8,
        in_channels=2048,
        as_two_stage=True,
        dn_cfg=dict(
            label_noise_scale=0.5,
            box_noise_scale=0.4,
            group_cfg=dict(dynamic=True, num_groups=None, num_dn_queries=500)),
        transformer=dict(
            type='CoDinoTransformer',
            with_coord_feat=False,
            num_co_heads=2,
            num_feature_levels=5,
            encoder=dict(
                type='DetrTransformerEncoder',
                num_layers=6,
                with_cp=6,
                transformerlayers=dict(
                    type='BaseTransformerLayer',
                    attn_cfgs=dict(
                        type='MultiScaleDeformableAttention',
                        embed_dims=256,
                        num_levels=5,
                        dropout=0.0),
                    feedforward_channels=2048,
                    ffn_dropout=0.0,
                    operation_order=('self_attn', 'norm', 'ffn', 'norm'))),
            decoder=dict(
                type='DinoTransformerDecoder',
                num_layers=6,
                return_intermediate=True,
                transformerlayers=dict(
                    type='DetrTransformerDecoderLayer',
                    attn_cfgs=[
                        dict(
                            type='MultiheadAttention',
                            embed_dims=256,
                            num_heads=8,
                            dropout=0.0),
                        dict(
                            type='MultiScaleDeformableAttention',
                            embed_dims=256,
                            num_levels=5,
                            dropout=0.0)
                    ],
                    feedforward_channels=2048,
                    ffn_dropout=0.0,
                    operation_order=('self_attn', 'norm', 'cross_attn', 'norm',
                                     'ffn', 'norm')))),
        positional_encoding=dict(
            type='SinePositionalEncoding',
            num_feats=128,
            temperature=20,
            normalize=True),
        loss_cls=dict(
            type='QualityFocalLoss',
            use_sigmoid=True,
            beta=2.0,
            loss_weight=1.0),
        loss_bbox=dict(type='L1Loss', loss_weight=5.0),
        loss_iou=dict(type='GIoULoss', loss_weight=2.0)),
    rpn_head=dict(
        type='RPNHead',
        in_channels=256,
        feat_channels=256,
        anchor_generator=dict(
            type='AnchorGenerator',
            octave_base_scale=4,
            scales_per_octave=3,
            ratios=[0.5, 1.0, 2.0],
            strides=[4, 8, 16, 32, 64, 128]),
        bbox_coder=dict(
            type='DeltaXYWHBBoxCoder',
            target_means=[0.0, 0.0, 0.0, 0.0],
            target_stds=[1.0, 1.0, 1.0, 1.0]),
        loss_cls=dict(
            type='CrossEntropyLoss', use_sigmoid=True, loss_weight=12.0),
        loss_bbox=dict(type='L1Loss', loss_weight=12.0)),
    roi_head=[
        dict(
            type='CoStandardRoIHead',
            bbox_roi_extractor=dict(
                type='SingleRoIExtractor',
                roi_layer=dict(
                    type='RoIAlign', output_size=7, sampling_ratio=0),
                out_channels=256,
                featmap_strides=[4, 8, 16, 32, 64],
                finest_scale=56),
            bbox_head=dict(
                type='Shared2FCBBoxHead',
                in_channels=256,
                fc_out_channels=1024,
                roi_feat_size=7,
                num_classes=8,
                bbox_coder=dict(
                    type='DeltaXYWHBBoxCoder',
                    target_means=[0.0, 0.0, 0.0, 0.0],
                    target_stds=[0.1, 0.1, 0.2, 0.2]),
                reg_class_agnostic=False,
                reg_decoded_bbox=True,
                loss_cls=dict(
                    type='CrossEntropyLoss',
                    use_sigmoid=False,
                    loss_weight=12.0),
                loss_bbox=dict(type='GIoULoss', loss_weight=120.0)))
    ],
    bbox_head=[
        dict(
            type='CoATSSHead',
            num_classes=8,
            in_channels=256,
            stacked_convs=1,
            feat_channels=256,
            anchor_generator=dict(
                type='AnchorGenerator',
                ratios=[1.0],
                octave_base_scale=8,
                scales_per_octave=1,
                strides=[4, 8, 16, 32, 64, 128]),
            bbox_coder=dict(
                type='DeltaXYWHBBoxCoder',
                target_means=[0.0, 0.0, 0.0, 0.0],
                target_stds=[0.1, 0.1, 0.2, 0.2]),
            loss_cls=dict(
                type='FocalLoss',
                use_sigmoid=True,
                gamma=2.0,
                alpha=0.25,
                loss_weight=12.0),
            loss_bbox=dict(type='GIoULoss', loss_weight=24.0),
            loss_centerness=dict(
                type='CrossEntropyLoss', use_sigmoid=True, loss_weight=12.0))
    ],
    train_cfg=[
        dict(
            assigner=dict(
                type='HungarianAssigner',
                match_costs=[
                    dict(type='FocalLossCost', weight=2.0),
                    dict(type='BBoxL1Cost', weight=5.0, box_format='xywh'),
                    dict(type='IoUCost', iou_mode='giou', weight=2.0)
                ])),
        dict(
            rpn=dict(
                assigner=dict(
                    type='MaxIoUAssigner',
                    pos_iou_thr=0.7,
                    neg_iou_thr=0.3,
                    min_pos_iou=0.3,
                    match_low_quality=True,
                    ignore_iof_thr=-1),
                sampler=dict(
                    type='RandomSampler',
                    num=256,
                    pos_fraction=0.5,
                    neg_pos_ub=-1,
                    add_gt_as_proposals=False),
                allowed_border=-1,
                pos_weight=-1,
                debug=False),
            rpn_proposal=dict(
                nms_pre=4000,
                max_per_img=1000,
                nms=dict(type='nms', iou_threshold=0.7),
                min_bbox_size=0),
            rcnn=dict(
                assigner=dict(
                    type='MaxIoUAssigner',
                    pos_iou_thr=0.5,
                    neg_iou_thr=0.5,
                    min_pos_iou=0.5,
                    match_low_quality=False,
                    ignore_iof_thr=-1),
                sampler=dict(
                    type='RandomSampler',
                    num=512,
                    pos_fraction=0.25,
                    neg_pos_ub=-1,
                    add_gt_as_proposals=True),
                pos_weight=-1,
                debug=False)),
        dict(
            assigner=dict(type='ATSSAssigner', topk=9),
            allowed_border=-1,
            pos_weight=-1,
            debug=False)
    ],
    test_cfg=[
        dict(max_per_img=300, nms=dict(type='soft_nms', iou_threshold=0.8)),
        dict(
            rpn=dict(
                nms_pre=1000,
                max_per_img=1000,
                nms=dict(type='nms', iou_threshold=0.7),
                min_bbox_size=0),
            rcnn=dict(
                score_thr=0.0,
                nms=dict(type='nms', iou_threshold=0.5),
                max_per_img=100)),
        dict(
            nms_pre=1000,
            min_bbox_size=0,
            score_thr=0.0,
            nms=dict(type='nms', iou_threshold=0.6),
            max_per_img=100)
    ])
max_epochs = 15
classes = ('person', 'car', 'truck', 'bus', 'bicycle', 'bike', 'extra_vehicle',
           'dog')
mean = [139.4229080324816, 139.4229080324816, 139.4229080324816]
std = [56.34895042201581, 56.34895042201581, 56.34895042201581]
tta_model = dict(
    type='DetTTAModel',
    tta_cfg=dict(nms=dict(type='nms', iou_threshold=0.5), max_per_img=100))
albu_train_transforms = [
    dict(
        type='ShiftScaleRotate',
        shift_limit=0.0625,
        scale_limit=0.1,
        rotate_limit=5,
        interpolation=1,
        p=0.1),
    dict(
        type='OneOf',
        transforms=[
            dict(type='Blur', blur_limit=1, p=0.2),
            dict(type='GaussNoise', var_limit=(10.0, 20.0), p=0.2)
        ],
        p=0.5)
]
scale = (2048, 1280)
img_scales = [(2048, 1280), (1333, 800), (666, 400), (2000, 1200), (640, 480)]
tta_pipeline = [
    dict(type='LoadImageFromFile'),
    dict(
        type='TestTimeAug',
        transforms=[[{
            'type': 'Resize',
            'scale': (2048, 1280),
            'keep_ratio': True
        }, {
            'type': 'Resize',
            'scale': (1333, 800),
            'keep_ratio': True
        }, {
            'type': 'Resize',
            'scale': (666, 400),
            'keep_ratio': True
        }, {
            'type': 'Resize',
            'scale': (2000, 1200),
            'keep_ratio': True
        }, {
            'type': 'Resize',
            'scale': (640, 480),
            'keep_ratio': True
        }],
                    [{
                        'type': 'RandomFlip',
                        'direction': 'horizontal',
                        'prob': 1.0
                    }, {
                        'type': 'RandomFlip',
                        'prob': 0.0
                    }],
                    [{
                        'type':
                        'PackDetInputs',
                        'meta_keys':
                        ('img_id', 'img_path', 'ori_shape', 'img_shape',
                         'scale_factor', 'flip', 'flip_direction')
                    }]])
]
train_pipeline_2 = [
    dict(type='LoadImageFromFile'),
    dict(type='LoadAnnotations', with_bbox=True),
    dict(type='PackDetInputs')
]
launcher = 'none'
work_dir = 'test'
