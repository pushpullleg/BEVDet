# WP3 step 2: choose the gate setting on a dev set of 500 TRAIN frames (gate OFF here = "no gate").
# Same as ../bevdet-r50-cbgs-gate.py, but the priors come from ../wp3_priors/dev (real only, built by
# thesis-bevdet/scripts/wp3/build_priors.py --real-only; frames in thesis-bevdet/scripts/wp3/dev_frames.json).
# Pass the dev ann_file: --ann-file data/nuscenes/wp3_dev500_infos.pkl (eval_adversarial.py) and
# evaluate through thesis-bevdet/scripts/wp1/eval_subset.py (GT restricted to these frames).
_base_ = ['../bevdet-r50-cbgs-gate.py']

prior_root = '../wp3_priors/dev'
class_names = [
    'car', 'truck', 'construction_vehicle', 'bus', 'trailer', 'barrier',
    'motorcycle', 'bicycle', 'pedestrian', 'traffic_cone'
]
_collect_meta_keys = (
    'filename', 'ori_shape', 'img_shape', 'lidar2img', 'depth2img', 'cam2img', 'pad_shape',
    'scale_factor', 'flip', 'pcd_horizontal_flip', 'pcd_vertical_flip', 'box_mode_3d',
    'box_type_3d', 'img_norm_cfg', 'pcd_trans', 'sample_idx', 'pcd_scale_factor', 'pcd_rotation',
    'pcd_rotation_angle', 'pts_filename', 'transformation_3d_flow', 'trans_mat', 'affine_aug',
    'bev_prior_real')

# base test pipeline + LoadBEVPrior + the two prior meta keys
test_pipeline = [
    dict(type='PrepareImageInputs', data_config={{_base_.data_config}}),
    dict(type='LoadBEVPrior', prior_root=prior_root, variants=('real',)),
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
