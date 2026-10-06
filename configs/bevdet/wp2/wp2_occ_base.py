# WP2 (representation level, reduced form; design spec 10-06): occupancy supervision of the BEV
# features during fine-tuning, oracle Occ3D teacher (an IDEALIZED setting).
#   start:  released bevdet-r50-cbgs; WP1 shared schedule + Arm A data (real only), via
#           ../wp1/wp1_armA_real.py (trainval ann_file override included).
#   model:  BEVDetWP2Occ (wp2_occ/model.py) = BEVDet + training-only occupancy head on the BEV
#           encoder output: crop 100x100 (offset 14), transpose to [x][y], 2x upsample, 3x3-3x3-1x1
#           conv -> 16 heights x 18 classes. Head width 128 (not fixed by the spec).
#   loss:   detection losses + 1.0 x cross-entropy over 18 Occ3D classes, camera mask only.
#   aug:    the WP1 schedule uses BEVAug (rot +-22.5 deg, scale 0.95-1.05, x/y flips); the model
#           moves the targets into the augmented frame with the same bda matrix.
#   lr:     head 10x (lr_mult 10 -> 2e-4); everything else as WP1. The cosine floor is written as
#           min_lr_ratio=0.1 (2e-5 -> 2e-6, identical to WP1's min_lr=2e-6 for every non-head
#           parameter) so the head anneals 2e-4 -> 2e-5 and stays 10x for the whole schedule.
#   test:   the head is never called; deploy = checkpoint without 'occ_head.' keys
#           (thesis-bevdet/scripts/wp2/strip_occ_head.py) + configs/bevdet/bevdet-r50-cbgs.py.
# Leaf configs: wp2_occ_s0.py, wp2_occ_s1.py (real targets), wp2_shuf_s0.py (shuffled targets).
# Run from the MAIN BEVDet tree (data/, checkpoints/, compiled ops) with this worktree on
# PYTHONPATH (for wp2_occ) and the config given by absolute path; see thesis-bevdet/scripts/wp2/.
_base_ = ['../wp1/wp1_armA_real.py']

custom_imports = dict(imports=['wp1_ema_hook', 'wp2_occ'], allow_failed_imports=False)

model = dict(
    type='BEVDetWP2Occ',
    occ_head=dict(in_channels=256, mid_channels=128),
    loss_occ_weight=1.0)

optimizer = dict(paramwise_cfg=dict(custom_keys={'occ_head': dict(lr_mult=10.0)}))
lr_config = dict(
    _delete_=True,
    policy='CosineAnnealing',
    by_epoch=False,
    min_lr_ratio=0.1,
    warmup='linear',
    warmup_iters=500,
    warmup_ratio=0.001,
    warmup_by_epoch=False)

class_names = [
    'car', 'truck', 'construction_vehicle', 'bus', 'trailer', 'barrier',
    'motorcycle', 'bicycle', 'pedestrian', 'traffic_cone'
]
point_cloud_range = [-51.2, -51.2, -5.0, 51.2, 51.2, 3.0]
# base train pipeline + LoadWP2OccTarget + the two target keys in Collect3D
train_pipeline = [
    dict(type='PrepareImageInputs', is_train=True, data_config={{_base_.data_config}}),
    dict(type='LoadWP2OccTarget'),
    dict(type='LoadAnnotations'),
    dict(type='BEVAug', bda_aug_conf={{_base_.bda_aug_conf}}, classes=class_names),
    dict(type='ObjectRangeFilter', point_cloud_range=point_cloud_range),
    dict(type='ObjectNameFilter', classes=class_names),
    dict(type='DefaultFormatBundle3D', class_names=class_names),
    dict(type='Collect3D',
         keys=['img_inputs', 'gt_bboxes_3d', 'gt_labels_3d', 'wp2_occ_sem', 'wp2_occ_mask'])
]
data = dict(train=dict(dataset=dict(pipeline=train_pipeline)))
