import os
import glob

raw_dir = "datasets/lower_limb/raw"
skel_dir = "datasets/lower_limb/skeletons"

mp4_files = []
for cls in os.listdir(raw_dir):
    cls_path = os.path.join(raw_dir, cls)
    if os.path.isdir(cls_path):
        for f in os.listdir(cls_path):
            if f.endswith('.mp4'):
                mp4_files.append(f.replace('.mp4', ''))

npy_files = glob.glob(os.path.join(skel_dir, '*.npy'))
deleted_count = 0
for npy_path in npy_files:
    basename = os.path.basename(npy_path).replace('.npy', '')
    if basename not in mp4_files:
        print(f"Removing unmatched skeleton: {npy_path}")
        os.remove(npy_path)
        deleted_count += 1

print(f"Cleaned up {deleted_count} files.")
