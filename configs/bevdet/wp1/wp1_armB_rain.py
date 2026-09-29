# WP1 Arm B: real nuScenes train data + kept MagicDrive rain frames (rain_v1, 3,473 frames with
# mean |gen - real| >= 15/255, pasted back to 1600x900), same 3D labels as their real frames.
# Infos built by thesis-bevdet/scripts/wp1/build_arm_infos.py (28,130 real + 3,473 synthetic).
# Schedule: ./wp1_schedule.py (shared by all WP1 arms). Only the training set differs from Arm A.
_base_ = ['./wp1_schedule.py']

data = dict(train=dict(dataset=dict(ann_file='data/nuscenes/wp1_armB_rain_v1_infos_train.pkl')))
