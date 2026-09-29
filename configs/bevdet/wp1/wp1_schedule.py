# WP1 SHARED FINE-TUNE SCHEDULE -- every WP1 arm (A: real only, B/C/D: real + synthetic)
# inherits this file unchanged and may only override data.train (the training set).
# Model, pipelines, CBGS, batch 8/GPU on 1 GPU, load_from = official bevdet-r50-cbgs.pth
# come from ./bevdet-r50-cbgs-wp1-ft.py.
#   lr:        AdamW 2e-5 (wd 1e-2, grad clip 5), linear warm-up 500 iters (from 2e-8),
#              cosine decay per iteration to 2e-6
#   length:    30,000 iterations (IterBasedRunner; ~1.94 CBGS epochs of the real train set)
#   seed:      0 (tools/train.py --seed 0, set by thesis-bevdet/scripts/wp1/run_arm.sh)
#   ckpt:      every 10,000 iters: raw iter_N.pth + EMA iter_N_ema.pth; evaluated = iter_30000_ema.pth
#   log:       text only, every 50 iters
#   EMA:       ON, as in the original recipe: the official checkpoint holds EMA weights
#              (keys epoch/state_dict/updates) from MEGVIIEMAHook. WP1IterEMAHook
#              (thesis-bevdet/scripts/wp1/wp1_ema_hook.py) is that hook unchanged (decay 0.999,
#              init_updates 10560, EMA starts from the loaded official weights) plus saving
#              iter_<N>_ema.pth every 10,000 iters (MEGVIIEMAHook only saves at epoch ends,
#              which IterBasedRunner never reaches). Every arm is EVALUATED ON THE EMA WEIGHTS
#              (iter_30000_ema.pth). Changed 2026-09-29; the first Arm A run (EMA off) is kept
#              as wp1_runs/armA_real_noEMA (mAP 0.3013 / NDS 0.3876).
_base_ = ['./bevdet-r50-cbgs-wp1-ft.py']

optimizer = dict(type='AdamW', lr=2e-5, weight_decay=1e-2)
optimizer_config = dict(grad_clip=dict(max_norm=5, norm_type=2))
lr_config = dict(
    _delete_=True,
    policy='CosineAnnealing',
    by_epoch=False,
    min_lr=2e-6,
    warmup='linear',
    warmup_iters=500,
    warmup_ratio=0.001,
    warmup_by_epoch=False)
runner = dict(_delete_=True, type='IterBasedRunner', max_iters=30000)
checkpoint_config = dict(interval=10000, by_epoch=False)
log_config = dict(interval=50, hooks=[dict(type='TextLoggerHook', by_epoch=False)])
custom_imports = dict(imports=['wp1_ema_hook'], allow_failed_imports=False)
custom_hooks = [dict(type='WP1IterEMAHook', init_updates=10560, interval=10000, priority='NORMAL')]
workflow = [('train', 1)]
