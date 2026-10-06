# WP2, real Occ3D targets, training seed 0 (tools/train.py --seed 0; WP2SeedCheckHook fails fast
# on a mismatch). See ./wp2_occ_base.py.
_base_ = ['./wp2_occ_base.py']
custom_hooks = [
    dict(type='WP1IterEMAHook', init_updates=10560, interval=10000, priority='NORMAL'),
    dict(type='WP2SeedCheckHook', seed=0),
]
