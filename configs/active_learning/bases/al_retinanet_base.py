mmdet_base = "../../_base_"
_base_ = [
    "models/retinanet_r50_fpn.py",
    f"{mmdet_base}/schedules/schedule_1x.py",
    f"{mmdet_base}/default_runtime.py",
]
dataset_type = 'ALVOCDataset'
data_root = 'data/VOC0712/'

CLASSES = ('aeroplane', 'bicycle', 'bird', 'boat', 'bottle', 'bus', 'car',
           'cat', 'chair', 'cow', 'diningtable', 'dog', 'horse', 'motorbike',
           'person', 'pottedplant', 'sheep', 'sofa', 'train', 'tvmonitor')

img_norm_cfg = dict(mean=[123.675, 116.28, 103.53], std=[58.395, 57.12, 57.375], to_rgb=True)

train_pipeline = [
    dict(type='LoadImageFromFile'),
    dict(type='LoadAnnotations', with_bbox=True),
    dict(type='Resize', img_scale=(1000, 600), keep_ratio=True),
    dict(type='RandomFlip', flip_ratio=0.5),
    dict(type='Normalize', **img_norm_cfg),
    dict(type='Pad', size_divisor=32),
    dict(type='DefaultFormatBundle'),
    dict(type='Collect', keys=['img', 'gt_bboxes', 'gt_labels']),
]

test_pipeline = [
    dict(type='LoadImageFromFile'),
    dict(
        type='MultiScaleFlipAug',
        img_scale=(1000, 600),
        flip=False,
        transforms=[
            dict(type='Resize', keep_ratio=True),
            dict(type='RandomFlip'),
            dict(type='Normalize', **img_norm_cfg),
            dict(type='Pad', size_divisor=32),
            dict(type='ImageToTensor', keys=['img']),
            dict(type='Collect', keys=['img']),
        ])
]

data = dict(
    samples_per_gpu=2,
    workers_per_gpu=4,
    train=dict(
        type='ALCocoDataset',
        ann_file=None,
        img_prefix='data/VOC0712/images',
        pipeline=train_pipeline,
        classes=CLASSES),
    val=dict(
        type='ALVOCDataset',
        ann_file='data/VOCdevkit/VOC2007_test/ImageSets/Main/test.txt',
        img_prefix='data/VOCdevkit/VOC2007_test/',
        pipeline=test_pipeline),
    test=dict(
        type='ALVOCDataset',
        ann_file='data/VOCdevkit/VOC2007_test/ImageSets/Main/test.txt',
        img_prefix='data/VOCdevkit/VOC2007_test/',
        pipeline=test_pipeline),
    
    test1=dict(
        type='ALCocoDataset',
        ann_file='work_dirs/retinanet_voc_7rounds_5percent_to_20percent/round1/annotations/labeled.json',
        img_prefix='data/VOC0712/images',
        pipeline=test_pipeline,
        classes=CLASSES),
    test2=dict(
        type='ALCocoDataset',
        ann_file='work_dirs/retinanet_voc_7rounds_5percent_to_20percent/round2/annotations/labeled.json',
        img_prefix='data/VOC0712/images',
        pipeline=test_pipeline,
        classes=CLASSES),
    test3=dict(
        type='ALCocoDataset',
        ann_file='work_dirs/retinanet_voc_7rounds_5percent_to_20percent/round3/annotations/labeled.json',
        img_prefix='data/VOC0712/images',
        pipeline=test_pipeline,
        classes=CLASSES),
    test4=dict(
        type='ALCocoDataset',
        ann_file='work_dirs/retinanet_voc_7rounds_5percent_to_20percent/round4/annotations/labeled.json',
        img_prefix='data/VOC0712/images',
        pipeline=test_pipeline,
        classes=CLASSES),
    test5=dict(
        type='ALCocoDataset',
        ann_file='work_dirs/retinanet_voc_7rounds_5percent_to_20percent/round5/annotations/labeled.json',
        img_prefix='data/VOC0712/images',
        pipeline=test_pipeline,
        classes=CLASSES),
    test6=dict(
        type='ALCocoDataset',
        ann_file='work_dirs/retinanet_voc_7rounds_5percent_to_20percent/round6/annotations/labeled.json',
        img_prefix='data/VOC0712/images',
        pipeline=test_pipeline,
        classes=CLASSES),
    test7=dict(
        type='ALCocoDataset',
        ann_file='work_dirs/retinanet_voc_7rounds_5percent_to_20percent/round7/annotations/labeled.json',
        img_prefix='data/VOC0712/images',
        pipeline=test_pipeline,
        classes=CLASSES),
    
    )
log_config = dict(
    interval=50,
    hooks=[
        dict(type="TextLoggerHook", by_epoch=False),
    ],
)
