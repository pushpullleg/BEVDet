# WP1 SHARED FINE-TUNE SCHEDULE -- every WP1 arm (A: real only, B/C/D: real + synthetic)
# inherits this file unchanged and may only override data.train (the training set).
# Model, pipelines, CBGS, batch 8/GPU on 1 GPU, load_from = official bevdet-r50-cbgs.pth
# come from ./bevdet-r50-cbgs-wp1-ft.py.
#   lr:        AdamW 2e-5 (wd 1e-2, grad clip 5), linear warm-up 500 iters (from 2e-8),
#              cosine decay per iteration to 2e-6
#   length:    30,000 iterations (IterBasedRunner; ~1.94 CBGS epochs of the real train set)
#   seed:      0 (tools/train.py --seed 0, set by thesis-bevdet/scripts/wp1/run_arm.sh)
#   ckpt:      every 10,000 iters (iter_10000/20000/30000.pth); final = iter_30000.pth
#   log:       text only, every 50 iters
#   EMA:       OFF. The base config's MEGVIIEMAHook only saves its EMA weights in
#              after_train_epoch, which IterBasedRunner never calls, so it would be dead
#              compute. All arms train and evaluate the plain (non-EMA) weights.
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
custom_hooks = []
workflow = [('train', 1)]
