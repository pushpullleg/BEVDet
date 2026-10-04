# WP1 check (e) dose, Arm B_half: real nuScenes train data + every second kept MagicDrive rain
# keyframe within each scene (rain_v1, 1,753 of the 3,473 Arm B frames, all 91 scenes;
# thesis-bevdet/scripts/wp1/build_frames_half.py), same 3D labels as their real frames.
# Infos built by thesis-bevdet/scripts/wp1/build_arm_infos.py (28,130 real + 1,753 synthetic).
# Schedule: ./wp1_schedule.py (shared by all WP1 arms). Only the training set differs from Arm A.
_base_ = ['./wp1_schedule.py']

data = dict(train=dict(dataset=dict(ann_file='data/nuscenes/wp1_armBhalf_rain_v1_infos_train.pkl')))
