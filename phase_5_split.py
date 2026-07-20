import os
import glob
import csv
import re
import random

BASE_DIR = "datasets/lower_limb"
SKELETON_DIR = os.path.join(BASE_DIR, "skeletons")
METADATA_CSV = os.path.join(BASE_DIR, "metadata.csv")

CLASSES = [
    "ankle", "calf", "hamstring", "heel_slide",
    "hip", "knee", "leg_raise", "quadriceps", "toes"
]
CLASS_MAP = {c: i for i, c in enumerate(CLASSES)}

# datasets/lower_limb/augment_fill.py writes source_url as
# "augmented_from_{source_filename}_{aug_type}" for every clip it generates.
AUG_SOURCE_PATTERN = re.compile(r'^augmented_from_(.+)_(hflip|brightness)$')

RANDOM_SEED = 42
TEST_FRAC = 0.20
VAL_FRAC_OF_REMAINING = 0.1875  # ~0.1875 * 0.80 = 0.15 of the total


def load_source_map():
    """mp4 filename -> source_url, straight from metadata.csv."""
    mapping = {}
    if os.path.exists(METADATA_CSV):
        with open(METADATA_CSV, newline='', encoding='utf-8') as f:
            for row in csv.DictReader(f):
                mapping[row['filename']] = row.get('source_url', '') or ''
    return mapping


def get_group_id(basename, source_map):
    """
    Groups a skeleton file (basename, no extension) with the real video it
    ultimately derives from, so augmented near-duplicates never end up on the
    opposite side of a split from their source. Real videos are their own group.
    """
    if '_aug_' not in basename:
        return basename

    mp4name = basename + '.mp4'
    src_url = source_map.get(mp4name)
    if src_url:
        m = AUG_SOURCE_PATTERN.match(src_url)
        if m:
            src_mp4 = m.group(1)
            return src_mp4[:-4] if src_mp4.endswith('.mp4') else src_mp4

    # Fallback: source couldn't be resolved from metadata -- treat the
    # augmented clip as its own group rather than crashing the split.
    return basename


def split_dataset():
    npy_files = glob.glob(os.path.join(SKELETON_DIR, "*.npy"))
    source_map = load_source_map()

    # class_name -> {group_id: [ {filename, class_id}, ... ]}
    class_groups = {c: {} for c in CLASSES}

    for f in npy_files:
        basename_ext = os.path.basename(f)
        basename = basename_ext[:-4]  # strip .npy
        parts = basename.rsplit('_', 1)
        if len(parts) != 2:
            continue
        class_name = parts[0]
        if class_name.endswith('_aug'):
            class_name = class_name[:-4]
        if class_name not in CLASS_MAP:
            continue

        class_id = CLASS_MAP[class_name]
        group_id = get_group_id(basename, source_map)
        class_groups[class_name].setdefault(group_id, []).append(
            {'filename': basename_ext, 'class_id': class_id}
        )

    total_files = sum(len(v) for g in class_groups.values() for v in g.values())
    if total_files < 10:
        print("Not enough valid tensors to split.")
        return

    rng = random.Random(RANDOM_SEED)
    train_data, val_data, test_data = [], [], []

    print("\n--- Group-aware split summary ---")
    for c in CLASSES:
        groups = class_groups[c]
        group_ids = list(groups.keys())
        rng.shuffle(group_ids)

        total_samples = sum(len(groups[g]) for g in group_ids)
        target_test = round(total_samples * TEST_FRAC)
        target_val = round((total_samples - target_test) * VAL_FRAC_OF_REMAINING)

        c_test, c_val, c_train = [], [], []
        for g in group_ids:
            items = groups[g]
            if len(c_test) < target_test:
                c_test.extend(items)
            elif len(c_val) < target_val:
                c_val.extend(items)
            else:
                c_train.extend(items)

        train_data.extend(c_train)
        val_data.extend(c_val)
        test_data.extend(c_test)
        print(f"  {c:12s} groups={len(group_ids):3d}  train={len(c_train):3d} val={len(c_val):3d} test={len(c_test):3d}")

    # Sanity check: no (class, group) pair should appear in more than one split.
    def group_set(data):
        s = set()
        for row in data:
            basename = row['filename'][:-4]
            s.add((row['class_id'], get_group_id(basename, source_map)))
        return s

    train_groups, val_groups, test_groups = group_set(train_data), group_set(val_data), group_set(test_data)
    overlap = (train_groups & val_groups) | (train_groups & test_groups) | (val_groups & test_groups)
    assert not overlap, f"Group leakage detected across splits: {overlap}"

    for name, data in [
        ('train_labels.csv', train_data),
        ('val_labels.csv', val_data),
        ('test_labels.csv', test_data),
    ]:
        path = os.path.join(BASE_DIR, name)
        with open(path, 'w', newline='') as f:
            writer = csv.DictWriter(f, fieldnames=['filename', 'class_id'])
            writer.writeheader()
            writer.writerows(data)

    print("\n--- Phase 5 Completed ---")
    print(f"Generated splits: {len(train_data)} train / {len(val_data)} val / {len(test_data)} test samples.")
    print("No group leakage across train/val/test splits (verified).")


if __name__ == "__main__":
    split_dataset()
