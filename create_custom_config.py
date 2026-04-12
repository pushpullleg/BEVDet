print("\n" + "="*70)
print("Creating a Custom BEVDet Config for Your RTX 4080")
print("="*70)

config_template = """
# Custom BEVDet config for RTX 4080
# Based on bevdet-r50.py but optimized for your hardware

# Inherit from base config
_base_ = ['./bevdet-r50.py']

# Optimize for RTX 4080 (16GB VRAM)
data = dict(
    samples_per_gpu=8,      # You can handle larger batches!
    workers_per_gpu=8,      # Parallel data loading
)

# Use mixed precision for faster training
fp16 = dict(loss_scale=512.)

# Learning rate (scale with batch size)
optimizer = dict(
    type='AdamW',
    lr=0.0002 * 2,  # 2x because 2x batch size
    weight_decay=0.01
)

# Save checkpoints
checkpoint_config = dict(interval=1)  # Save every epoch

# Where to save results
work_dir = './outputs/bevdet_rtx4080_experiment'
"""

print("📝 Here's a custom config optimized for your RTX 4080:")
print("="*70)
print(config_template)
print("="*70)

print("\n💡 What we changed:")
print("  • Batch size: 4 → 8 (use your 16GB VRAM!)")
print("  • Workers: 4 → 8 (faster data loading)")
print("  • Added FP16: 2x faster training")
print("  • Adjusted learning rate for larger batch")

print("\n🎯 To use this config:")
print("  1. Save it as: configs/bevdet/bevdet-r50-rtx4080.py")
print("  2. Train with: python tools/train.py configs/bevdet/bevdet-r50-rtx4080.py")

print("\n" + "="*70 + "\n")
