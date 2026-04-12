import os

print("\n" + "="*70)
print("nuScenes Mini Dataset - Complete Verification")
print("="*70)

base_path = "data/nuscenes"

# Check directories
required_dirs = ['maps', 'samples', 'sweeps', 'v1.0-mini']
print(f"\n📂 Dataset Directories:")
all_dirs_ok = True
for dir_name in required_dirs:
    dir_path = os.path.join(base_path, dir_name)
    if os.path.exists(dir_path):
        file_count = len(os.listdir(dir_path))
        print(f"  ✓ {dir_name:15s} - {file_count:4d} items")
    else:
        print(f"  ✗ {dir_name:15s} - MISSING!")
        all_dirs_ok = False

# Check info files
print(f"\n📋 Data Info Files:")
info_files = [
    'nuscenes_infos_train.pkl',
    'nuscenes_infos_val.pkl'
]
all_info_ok = True
for info_file in info_files:
    info_path = os.path.join(base_path, info_file)
    if os.path.exists(info_path):
        size_mb = os.path.getsize(info_path) / (1024 * 1024)
        print(f"  ✓ {info_file:30s} - {size_mb:.1f} MB")
    else:
        print(f"  ✗ {info_file:30s} - MISSING!")
        all_info_ok = False

print("\n" + "="*70)

if all_dirs_ok and all_info_ok:
    print("🎉 Dataset is READY for BEVDet!")
    print("\n📊 Mini Dataset Contains:")
    print("  • 10 scenes from Boston & Singapore")
    print("  • ~400 annotated samples")
    print("  • 6 cameras per sample (360° coverage)")
    print("  • 3D bounding boxes for cars, pedestrians, etc.")
    
    print("\n🚀 Next Step: Download pretrained model & run inference!")
elif not all_info_ok:
    print("⚠️  Data info files missing!")
    print("\nRun this command:")
    print("  python tools/create_data.py nuscenes \\")
    print("      --root-path ./data/nuscenes \\")
    print("      --out-dir ./data/nuscenes \\")
    print("      --extra-tag nuscenes")
else:
    print("⚠️  Some directories missing - check extraction")

print("="*70 + "\n")
