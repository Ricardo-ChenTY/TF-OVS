_base_ = './base_config.py'

model = dict(name_path='./configs/cls_tfovos_voc20.txt',
    instance_mask_path='data/region_masks/voc',
)

dataset_type = 'PascalVOC20Dataset'
data_root = '/data/tianyi/TF-OVCOS/data/official_mmseg/voc20'

test_pipeline = [
    dict(type='LoadImageFromFile'),
    dict(type='Resize', scale=(2048, 336), keep_ratio=True),
    dict(type='LoadAnnotations'),
    dict(type='PackSegInputs')
]

test_dataloader = dict(
    batch_size=1,
    num_workers=4,
    persistent_workers=True,
    sampler=dict(type='DefaultSampler', shuffle=False),
    dataset=dict(
        type=dataset_type,
        data_root=data_root,
        data_prefix=dict(img_path='JPEGImages', seg_map_path='SegmentationClass'),
        ann_file='ImageSets/Segmentation/val.txt',
        pipeline=test_pipeline))


# Official mmseg IoUMetric output. This preserves prediction label maps for
# proposal exploratory metrics without changing model inference or mIoU logic.
test_evaluator = dict(type='IoUMetric', iou_metrics=['mIoU'], output_dir='/data/tianyi/TF-OVCOS/runs/artifacts/official_predictions/corrclip/voc20')

