data_root = '/data/tianyi/TF-OVCOS/data/official_mmseg/context459'
dataset_type = 'PascalContext459Dataset'
default_hooks = dict(
    checkpoint=dict(by_epoch=False, interval=2000, type='CheckpointHook'),
    logger=dict(interval=50, log_metric_by_epoch=False, type='LoggerHook'),
    param_scheduler=dict(type='ParamSchedulerHook'),
    sampler_seed=dict(type='DistSamplerSeedHook'),
    timer=dict(type='IterTimerHook'),
    visualization=dict(interval=5, type='SegVisualizationHook'))
default_scope = 'mmseg'
env_cfg = dict(
    cudnn_benchmark=True,
    dist_cfg=dict(backend='nccl'),
    mp_cfg=dict(mp_start_method='fork', opencv_num_threads=0))
launcher = 'none'
load_from = None
log_level = 'INFO'
log_processor = dict(by_epoch=False)
model = dict(
    clip_type='metaclip_fullcc',
    dino_type='dino_vitb8',
    instance_mask_path='data/region_masks/context',
    mask_generator=None,
    model_type='ViT-B-16-quickgelu',
    name_path='./configs/cls_context459.txt',
    type='CorrCLIPSegmentation')
resume = False
test_cfg = dict(type='TestLoop')
test_dataloader = dict(
    batch_size=1,
    dataset=dict(
        ann_file='ImageSets/SegmentationContext/val.txt',
        data_prefix=dict(
            img_path='JPEGImages',
            seg_map_path='annotations_detectron2/pc459_val'),
        data_root='/data/tianyi/TF-OVCOS/data/official_mmseg/context459',
        pipeline=[
            dict(type='LoadImageFromFile'),
            dict(keep_ratio=True, scale=(
                2048,
                448,
            ), type='Resize'),
            dict(reduce_zero_label=False, type='MyLoadAnnotations'),
            dict(type='PackSegInputs'),
        ],
        type='PascalContext459Dataset'),
    num_workers=4,
    persistent_workers=True,
    sampler=dict(shuffle=False, type='DefaultSampler'))
test_evaluator = dict(
    iou_metrics=[
        'mIoU',
    ],
    output_dir=
    '/data/tianyi/TF-OVCOS/runs/artifacts/official_predictions/corrclip/context459',
    type='IoUMetric')
test_pipeline = [
    dict(type='LoadImageFromFile'),
    dict(keep_ratio=True, scale=(
        2048,
        448,
    ), type='Resize'),
    dict(reduce_zero_label=False, type='MyLoadAnnotations'),
    dict(type='PackSegInputs'),
]
vis_backends = [
    dict(type='LocalVisBackend'),
]
visualizer = dict(
    alpha=1.0,
    name='visualizer',
    type='SegLocalVisualizer',
    vis_backends=[
        dict(type='LocalVisBackend'),
    ])
work_dir = './work_dirs/cfg_tfovos_context459_e2'
