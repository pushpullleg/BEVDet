# WP1 fine-tune config: bevdet-r50-cbgs (unchanged, inherited) fine-tuned from its
# official checkpoint on the FULL nuScenes train split (v1.0-trainval infos).
# Everything not overridden here (model, pipelines, batch 8/GPU, CBGS, AdamW 2e-4,
# EMA hook, 20 epochs) comes from ../bevdet-r50-cbgs.py, which is not edited.
_base_ = ['../bevdet-r50-cbgs.py']

data_root = 'data/nuscenes/'
data = dict(
    train=dict(dataset=dict(ann_file=data_root + 'bevdetv3-trainval_infos_train.pkl')),
    val=dict(ann_file=data_root + 'bevdetv3-trainval_infos_val.pkl'),
    test=dict(ann_file=data_root + 'bevdetv3-trainval_infos_val.pkl'))

load_from = 'checkpoints/bevdet-dev2.1/bevdet-r50-cbgs.pth'
# Text log only: the bevdet env has no protobuf, so the default TensorboardLoggerHook
# fails to import (env deliberately left unchanged).
log_config = dict(interval=10, hooks=[dict(type='TextLoggerHook')])
