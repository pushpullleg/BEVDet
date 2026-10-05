# WP3 gate ON: after_vt, alpha 0.5, real prior. A fixed file because attacks/eval_adversarial.py
# takes no --cfg-options (used for the step-1 PGD smoke test).
_base_ = ['./bevdet-r50-cbgs-gate.py']
model = dict(gate=dict(enable=True, location='after_vt', alpha=0.5, prior='real'))
