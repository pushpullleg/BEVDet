import os
import sys

print("\n" + "="*70)
print("BEVDet Configuration File Explorer")
print("="*70)

config_path = "configs/bevdet/bevdet-r50.py"

if os.path.exists(config_path):
    print(f"\n📄 Reading: {config_path}\n")
    
    with open(config_path, 'r') as f:
        lines = f.readlines()
    
    print("🔍 Key Sections in Config File:")
    print("-" * 70)
    
    # Find important sections
    sections = {
        'model': False,
        'data': False,
        'optimizer': False,
        'lr_config': False,
        'checkpoint': False
    }
    
    for i, line in enumerate(lines, 1):
        for section in sections.keys():
            if f'{section} = dict(' in line or f'{section}=dict(' in line:
                sections[section] = i
    
    for section, line_num in sections.items():
        if line_num:
            print(f"  ✓ Line {line_num:4d}: {section.upper()}")
    
    print("\n" + "="*70)
    print("💡 What These Sections Mean:")
    print("="*70)
    print("""
  📐 MODEL       : Architecture definition (backbone, neck, head)
  📊 DATA        : Dataset paths, batch size, augmentation
  🎯 OPTIMIZER   : Learning rate, weight decay, etc.
  📈 LR_CONFIG   : Learning rate schedule
  💾 CHECKPOINT  : Where to save/load model weights
    """)
    
    print("="*70)
    print("🎓 For Your Thesis:")
    print("="*70)
    print("""
  • MODEL section     → Change architecture (easy experiments)
  • DATA section      → Modify batch size, augmentation
  • OPTIMIZER section → Tune hyperparameters
    """)
    
else:
    print(f"❌ Config file not found: {config_path}")
    print("Make sure you're in the BEVDet directory!")

print("="*70 + "\n")
