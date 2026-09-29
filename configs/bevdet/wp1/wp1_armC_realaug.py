# WP1 Arm C: real nuScenes train data + the same 3,473 kept frames as Arm B, but the images are
# REAL frames made appearance-matched to the rain images inside the paste-back region only
# (x 64-1536, y 356-900): bicubic 2x down/up + per-channel histogram match to the generated view.
# Real labels. Images: thesis-bevdet/scripts/wp1/build_armC_images.py; infos: build_arm_infos.py.
# Schedule: ./wp1_schedule.py (shared by all WP1 arms). Only the training set differs from Arm A.
_base_ = ['./wp1_schedule.py']

data = dict(train=dict(dataset=dict(ann_file='data/nuscenes/wp1_armC_realaug_v1_infos_train.pkl')))
