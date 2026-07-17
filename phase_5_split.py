import os
import glob
import csv
from sklearn.model_selection import train_test_split

BASE_DIR = "datasets/lower_limb"
SKELETON_DIR = os.path.join(BASE_DIR, "skeletons")

CLASSES = [
    "ankle", "calf", "hamstring", "heel_slide",
    "hip", "knee", "leg_raise", "quadriceps", "toes"
]
CLASS_MAP = {c: i for i, c in enumerate(CLASSES)}

def split_dataset():
    npy_files = glob.glob(os.path.join(SKELETON_DIR, "*.npy"))
    
    dataset = []
    labels = []
    
    for f in npy_files:
        basename = os.path.basename(f)
        # Parse class from filename, e.g., ankle_0001.npy
        # Assumes format is <class_name>_<idx>.npy
        # Since class_name might have underscores (e.g. heel_slide), we must be careful.
        parts = basename.rsplit('_', 1) # Split at the last underscore
        if len(parts) == 2:
            class_name = parts[0]
            if class_name.endswith('_aug'):
                class_name = class_name[:-4]
            if class_name in CLASS_MAP:
                dataset.append({'filename': basename, 'class_id': CLASS_MAP[class_name]})
                labels.append(CLASS_MAP[class_name])
            
    if len(dataset) < 10:
        print("Not enough valid tensors to split.")
        return
        
    train_data, test_data = train_test_split(dataset, test_size=0.2, stratify=labels, random_state=42)
    
    train_path = os.path.join(BASE_DIR, 'train_labels.csv')
    with open(train_path, 'w', newline='') as f:
        writer = csv.DictWriter(f, fieldnames=['filename', 'class_id'])
        writer.writeheader()
        writer.writerows(train_data)
        
    test_path = os.path.join(BASE_DIR, 'test_labels.csv')
    with open(test_path, 'w', newline='') as f:
        writer = csv.DictWriter(f, fieldnames=['filename', 'class_id'])
        writer.writeheader()
        writer.writerows(test_data)
        
    print("\n--- Phase 5 Completed ---")
    print(f"Generated splits: {len(train_data)} train samples, {len(test_data)} test samples.")

if __name__ == "__main__":
    split_dataset()
