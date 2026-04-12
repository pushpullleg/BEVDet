import os

print("\n" + "="*70)
print("BEVDet Tools Directory - Your Main Scripts")
print("="*70)

tools = {
    "train.py": {
        "purpose": "Train BEVDet models from scratch or fine-tune",
        "usage": "python tools/train.py configs/bevdet/bevdet-r50.py",
        "when": "When you want to train your own model"
    },
    "test.py": {
        "purpose": "Evaluate trained models on validation/test sets",
        "usage": "python tools/test.py configs/bevdet/bevdet-r50.py checkpoints/model.pth --eval bbox",
        "when": "After training, to see how well it performs"
    },
    "dist_train.sh": {
        "purpose": "Distributed training on multiple GPUs",
        "usage": "bash tools/dist_train.sh configs/bevdet/bevdet-r50.py 2",
        "when": "If you have multiple GPUs (you have 1 RTX 4080)"
    },
    "create_data.py": {
        "purpose": "Prepare dataset (create info files)",
        "usage": "python tools/create_data.py nuscenes --root-path ./data/nuscenes",
        "when": "First time setting up nuScenes dataset"
    }
}

for script, info in tools.items():
    print(f"\n📜 {script}")
    print(f"   Purpose: {info['purpose']}")
    print(f"   Usage:   {info['usage']}")
    print(f"   When:    {info['when']}")

print("\n" + "="*70)
print("🎓 For Your Research Journey:")
print("="*70)
print("""
Phase 1: Understanding
  → Use test.py with pretrained models
  → Visualize results
  → Understand what's happening

Phase 2: Experimentation
  → Modify config files
  → Use train.py to train variants
  → Use test.py to compare results

Phase 3: Research
  → Implement your modifications
  → Train multiple versions
  → Analyze and write thesis
""")

print("="*70 + "\n")
