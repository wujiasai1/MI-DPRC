
_base_ = "../bases/al_retinanet_base.py"

labeled_data = ''
unlabeled_data = ''

model = dict(
    bbox_head=dict(
        type='RetinaQualityEMAHead',
        base_momentum=0.99
    )
)
data = dict(
    samples_per_gpu=2,
    workers_per_gpu=4,
    train=dict(
        ann_file=None   
    ),
)

evaluation=dict(interval=26, metric='mAP')


optimizer = dict(type="SGD", lr=0.001, momentum=0.9, weight_decay=0.0001)
optimizer_config = dict(_delete_=True, grad_clip=dict(max_norm=35, norm_type=2))


lr_config = dict(
    policy='step',
    warmup='linear',
    step=[20])

runner = dict(type='EpochBasedRunner', max_epochs=26) 
checkpoint_config = dict(interval=26, max_keep_ckpts=1, by_epoch=True)



