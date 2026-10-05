# WP3 training-free occupancy gate on bevdet-r50-cbgs (thesis). Same model, weights and test
# pipeline as ../bevdet-r50-cbgs.py, plus:
#   - model type BEVDetGate (wp3_gate/gate.py): F' = F * (alpha + (1 - alpha) * P);
#     gate OFF by default; switch with --cfg-options, e.g.
#       model.gate.enable=True model.gate.location=after_vt model.gate.alpha=0.5 model.gate.prior=real
#   - LoadBEVPrior: real + shuffled priors read by token from ../wp3_priors/val/{real,shuffled}/
#     (relative to $BEVDET_ROOT, built by thesis-bevdet/scripts/wp3/build_priors.py) and passed
#     in img_metas. The prior never comes from the images.
# Test only (priors exist for val frames). Run from $BEVDET_ROOT with $BEVDET_ROOT on sys.path
# (bevdet_compat.py adds the cwd). As for the base config, pass the FULL val ann_file:
#   --cfg-options data.test.ann_file=data/nuscenes/bevdetv3-trainval_infos_val.pkl
_base_ = ['../bevdet-r50-cbgs.py']

custom_imports = dict(imports=['wp3_gate'], allow_failed_imports=False)

model = dict(
    type='BEVDetGate',
    gate=dict(enable=False, location='after_vt', alpha=1.0, prior='real'))

prior_root = '../wp3_priors/val'
class_names = [
    'car', 'truck', 'construction_vehicle', 'bus', 'trailer', 'barrier',
    'motorcycle', 'bicycle', 'pedestrian', 'traffic_cone'
]
_collect_meta_keys = (
    'filename', 'ori_shape', 'img_shape', 'lidar2img', 'depth2img', 'cam2img', 'pad_shape',
    'scale_factor', 'flip', 'pcd_horizontal_flip', 'pcd_vertical_flip', 'box_mode_3d',
    'box_type_3d', 'img_norm_cfg', 'pcd_trans', 'sample_idx', 'pcd_scale_factor', 'pcd_rotation',
    'pcd_rotation_angle', 'pts_filename', 'transformation_3d_flow', 'trans_mat', 'affine_aug',
    'bev_prior_real', 'bev_prior_shuffled')

# base test pipeline + LoadBEVPrior + the two prior meta keys
test_pipeline = [
    dict(type='PrepareImageInputs', data_config={{_base_.data_config}}),
    dict(type='LoadBEVPrior', prior_root=prior_root),
    dict(type='LoadAnnotations'),
    dict(type='BEVAug',
         bda_aug_conf={{_base_.bda_aug_conf}},
         classes=class_names,
         is_train=False),
    dict(
        type='LoadPointsFromFile',
        coord_type='LIDAR',
        load_dim=5,
        use_dim=5,
        file_client_args=dict(backend='disk')),
    dict(
        type='MultiScaleFlipAug3D',
        img_scale=(1333, 800),
        pts_scale_ratio=1,
        flip=False,
        transforms=[
            dict(
                type='DefaultFormatBundle3D',
                class_names=class_names,
                with_label=False),
            dict(type='Collect3D', keys=['points', 'img_inputs'],
                 meta_keys=_collect_meta_keys)
        ])
]

data = dict(
    val=dict(pipeline=test_pipeline),
    test=dict(pipeline=test_pipeline))
