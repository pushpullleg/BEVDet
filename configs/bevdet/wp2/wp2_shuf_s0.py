# WP2 shuffled-target control, training seed 0: every frame is supervised with the Occ3D
# occupancy of a frame from a DIFFERENT scene (fixed pairing, np.random.default_rng(0);
# ../wp2_targets/shuffle_map_train.json relative to the main BEVDet tree, built by
# thesis-bevdet/scripts/wp2/build_shuffle_map.py). Everything else = ./wp2_occ_s0.py.
_base_ = ['./wp2_occ_s0.py']
train_pipeline = [
    dict(type='PrepareImageInputs', is_train=True, data_config={{_base_.data_config}}),
    dict(type='LoadWP2OccTarget', shuffle_map='../wp2_targets/shuffle_map_train.json'),
    dict(type='LoadAnnotations'),
    dict(type='BEVAug', bda_aug_conf={{_base_.bda_aug_conf}}, classes={{_base_.class_names}}),
    dict(type='ObjectRangeFilter', point_cloud_range={{_base_.point_cloud_range}}),
    dict(type='ObjectNameFilter', classes={{_base_.class_names}}),
    dict(type='DefaultFormatBundle3D', class_names={{_base_.class_names}}),
    dict(type='Collect3D',
         keys=['img_inputs', 'gt_bboxes_3d', 'gt_labels_3d', 'wp2_occ_sem', 'wp2_occ_mask'])
]
data = dict(train=dict(dataset=dict(pipeline=train_pipeline)))
