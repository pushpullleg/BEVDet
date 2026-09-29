# WP1 Arm D: real nuScenes train data + the same 3,473 pasted rain frames as Arm B, but each
# synthetic frame carries ALL labels of a different kept frame (fixed shuffle, seed 0, cyclic
# shift of a random permutation: no frame keeps its own labels). Mapping:
# thesis-bevdet/results/wp1/armD_label_donors.csv. Infos: build_arm_infos.py --label-shuffle-seed 0.
# Schedule: ./wp1_schedule.py (shared by all WP1 arms). Only the training set differs from Arm A.
_base_ = ['./wp1_schedule.py']

data = dict(train=dict(dataset=dict(ann_file='data/nuscenes/wp1_armD_rain_v1_scrambled_infos_train.pkl')))
