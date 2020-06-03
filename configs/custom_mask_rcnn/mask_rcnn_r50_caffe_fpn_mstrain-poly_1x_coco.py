
_base_ = [
#    '../_base_/models/mask_rcnn_r50_fpn.py'
    #'../_base_/schedules/schedule_1x.py', '../_base_/default_runtime.py'
]
# use caffe img_norm
classes = ('vale', 'chipped', 'bend', 'NIck', 'DIstortion', 'Missing material', 'Bird strike', 'Blend', 'Blend repair', 'Tip curl', 'Scratch')

model = {'type': 'MaskRCNN', 'pretrained': 'torchvision://resnet50' , 'backbone': {'type': 'ResNet', 'depth': 50, 'num_stages': 4, 'out_indices': (0, 1, 2, 3), 'frozen_stages': 1, 'norm_cfg': {'type': 'BN', 'requires_grad': True}, 'norm_eval': True, 'style': 'pytorch'}, 'neck': {'type': 'FPN', 'in_channels': [256, 512, 1024, 2048], 'out_channels': 256, 'num_outs': 5}, 'rpn_head': {'type': 'RPNHead', 'in_channels': 256, 'feat_channels': 256, 'anchor_generator': {'type': 'AnchorGenerator', 'scales': [8], 'ratios': [0.5, 1.0, 2.0], 'strides': [4, 8, 16, 32, 64]}, 'bbox_coder': {'type': 'DeltaXYWHBBoxCoder', 'target_means': [0.0, 0.0, 0.0, 0.0], 'target_stds': [1.0, 1.0, 1.0, 1.0]}, 'loss_cls': {'type': 'CrossEntropyLoss', 'use_sigmoid': True, 'loss_weight': 1.0}, 'loss_bbox': {'type': 'L1Loss', 'loss_weight': 1.0}}, 'roi_head': {'type': 'StandardRoIHead', 'bbox_roi_extractor': {'type': 'SingleRoIExtractor', 'roi_layer': {'type': 'RoIAlign', 'out_size': 7, 'sample_num': 0}, 'out_channels': 256, 'featmap_strides': [4, 8, 16, 32]}, 'bbox_head': {'type': 'Shared2FCBBoxHead', 'in_channels': 256, 'fc_out_channels': 1024, 'roi_feat_size': 7, 'num_classes': len(classes), 'bbox_coder': {'type': 'DeltaXYWHBBoxCoder', 'target_means': [0.0, 0.0, 0.0, 0.0], 'target_stds': [0.1, 0.1, 0.2, 0.2]}, 'reg_class_agnostic': False, 'loss_cls': {'type': 'CrossEntropyLoss', 'use_sigmoid': False, 'loss_weight': 1.0}, 'loss_bbox': {'type': 'L1Loss', 'loss_weight': 1.0}}, 'mask_roi_extractor': {'type': 'SingleRoIExtractor', 'roi_layer': {'type': 'RoIAlign', 'out_size': 14, 'sample_num': 0}, 'out_channels': 256, 'featmap_strides': [4, 8, 16, 32]}, 'mask_head': {'type': 'FCNMaskHead', 'num_convs': 4, 'in_channels': 256, 'conv_out_channels': 256, 'num_classes': len(classes), 'loss_mask': {'type': 'CrossEntropyLoss', 'use_mask': True, 'loss_weight': 1.0}}}}
img_norm_cfg = dict(
    mean=[103.530, 116.280, 123.675], std=[1.0, 1.0, 1.0], to_rgb=False)
train_pipeline = [
    dict(type='LoadImageFromFile'),
    dict(
        type='LoadAnnotations',
        with_bbox=True,
        with_mask=True,
        poly2mask=False),
    dict(
        type='Resize',
        img_scale=[(1333, 640), (1333, 672), (1333, 704), (1333, 736),
                   (1333, 768), (1333, 800)],
        multiscale_mode='value',
        keep_ratio=True),
    dict(type='RandomFlip', flip_ratio=0.5),
    dict(type='Normalize', **img_norm_cfg),
    dict(type='Pad', size_divisor=32),
    dict(type='DefaultFormatBundle'),
    dict(type='Collect', keys=['img', 'gt_bboxes', 'gt_labels', 'gt_masks']),
]
test_pipeline = [
    dict(type='LoadImageFromFile'),
    dict(
        type='MultiScaleFlipAug',
        img_scale=(1333, 800),
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
test_cfg = dict(
    rpn=dict(
        nms_across_levels=False,
        nms_pre=1000,
        nms_post=1000,
        max_num=1000,
        nms_thr=0.7,
        min_bbox_size=0),
    rcnn=dict(
        score_thr=0.05,
        nms=dict(type='nms', iou_thr=0.5),
        max_per_img=100,
        mask_thr_binary=0.5))

dataset_type = 'CocoDataset'
data_root = '/content/mmdetection_aircraft/data/coco/'
classes = ('vale', 'chipped', 'bend', 'NIck', 'DIstortion', 'Missing material', 'Bird strike', 'Blend', 'Blend repair', 'Tip curl', 'Scratch')
data = dict(
    imgs_per_gpu=1,
    workers_per_gpu=1,
    train=dict(
        type=dataset_type,
        ann_file=data_root + 'annotations/instances.json',
        img_prefix=data_root + 'images/',
        pipeline=train_pipeline),
    val=dict(
        type=dataset_type,
        ann_file=data_root + 'annotations/instances.json',
        img_prefix=data_root + 'images/',
        pipeline=test_pipeline),
    test=dict(
        type=dataset_type,
        ann_file=data_root + 'annotations/instances.json',
        img_prefix=data_root + 'images/',
        pipeline=test_pipeline))
log_level = 'INFO'
train_cfg = {'rpn': {'assigner': {'type': 'MaxIoUAssigner', 'pos_iou_thr': 0.7, 'neg_iou_thr': 0.3, 'min_pos_iou': 0.3, 'match_low_quality': True, 'ignore_iof_thr': -1}, 'sampler': {'type': 'RandomSampler', 'num': 256, 'pos_fraction': 0.5, 'neg_pos_ub': -1, 'add_gt_as_proposals': False}, 'allowed_border': -1, 'pos_weight': -1, 'debug': False}, 'rpn_proposal': {'nms_across_levels': False, 'nms_pre': 2000, 'nms_post': 1000, 'max_num': 1000, 'nms_thr': 0.7, 'min_bbox_size': 0}, 'rcnn': {'assigner': {'type': 'MaxIoUAssigner', 'pos_iou_thr': 0.5, 'neg_iou_thr': 0.5, 'min_pos_iou': 0.5, 'match_low_quality': True, 'ignore_iof_thr': -1}, 'sampler': {'type': 'RandomSampler', 'num': 512, 'pos_fraction': 0.25, 'neg_pos_ub': -1, 'add_gt_as_proposals': True}, 'mask_size': 28, 'pos_weight': -1, 'debug': False}}
#load_from = '/home/maksim/mmdetection/checkpoints/mask_rcnn_r50_caffe_fpn_mstrain-poly_1x_coco-dbecf295.pth'
workflow = [('train', 1)]
checkpoint_config = {'interval': 1}
optimizer = {'type': 'SGD', 'lr': 0.02, 'momentum': 0.9, 'weight_decay': 0.0001}
optimizer_config = {'grad_clip': None}
lr_config = {'policy': 'step', 'warmup': 'linear', 'warmup_iters': 500, 'warmup_ratio': 0.001, 'step': [8, 11]}
log_config = {'interval': 50, 'hooks': [{'type': 'TextLoggerHook'}]}
resume_from = None
load_from = None
#load_from = '/content/mmdetection_aircraft/checkpoints/mask_rcnn_r50_caffe_fpn_mstrain-poly_1x_coco-dbecf295.pth'
total_epochs = 12