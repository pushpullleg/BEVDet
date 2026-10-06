# WP3 step 3 (full-val protocol): the pre-registered setting (chosen on the 500-frame train dev set,
# thesis-bevdet results/wp3/step2): gate ON, after_enc, alpha 0.5, real prior. Fixed file because
# attacks/eval_adversarial.py takes no --cfg-options. Pass --ann-file data/nuscenes/bevdetv3-trainval_infos_val.pkl.
_base_ = ['./bevdet-r50-cbgs-gate.py']
model = dict(gate=dict(enable=True, location='after_enc', alpha=0.5, prior='real'))
