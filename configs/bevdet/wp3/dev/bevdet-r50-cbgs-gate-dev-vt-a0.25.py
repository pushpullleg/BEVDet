# WP3 step 2 dev setting: gate ON, after_vt, alpha 0.25, real prior (fixed file: eval_adversarial.py takes no --cfg-options).
_base_ = ['./bevdet-r50-cbgs-gate-dev.py']
model = dict(gate=dict(enable=True, location='after_vt', alpha=0.25, prior='real'))
