"""
Phase 5: Split.

Usage:
    python phase_5_split.py --domain lower_limb
    python phase_5_split.py --domain upper_body

Splits at the SOURCE-VIDEO level, not the file level. An augmented clip
(hflip/brightness) is a near-duplicate of the video it was derived from --
letting one land in train and the other in test leaks information from test
into train and inflates the reported accuracy. Every augmented clip is
grouped with its source video via metadata.csv's source_url field
(format: augmented_from_<source_filename>_<aug_type>) and the whole group is
assigned to exactly one split.

Produces train/val/test (70/15/15) so the training script has a real
validation set instead of selecting its best checkpoint using the test set.
"""
import os
import csv
import glob
import argparse
from collections import defaultdict

from sklearn.model_selection import train_test_split

from domains import get_domain, DOMAIN_NAMES


def parse_class(basename, class_names):
    class_set = set(class_names)
    parts = basename.rsplit('_', 1)
    if len(parts) != 2:
        return None
    class_name = parts[0]
    if class_name.endswith('_aug'):
        class_name = class_name[:-4]
    return class_name if class_name in class_set else None


def load_augmentation_sources(metadata_csv):
    """filename (mp4) -> source filename (mp4), for augmented clips only."""
    sources = {}
    if not os.path.exists(metadata_csv):
        return sources
    with open(metadata_csv, 'r', encoding='utf-8') as f:
        for row in csv.DictReader(f):
            src = row.get('source_url', '') or ''
            if src.startswith('augmented_from_'):
                rest = src[len('augmented_from_'):]
                idx = rest.find('.mp4')
                if idx != -1:
                    sources[row['filename']] = rest[:idx + 4]
    return sources


def split_dataset(domain):
    base_dir = os.path.join("datasets", domain.name)
    skeleton_dir = os.path.join(base_dir, "skeletons")
    class_to_id = {c: i for i, c in enumerate(domain.classes)}

    npy_files = glob.glob(os.path.join(skeleton_dir, "*.npy.enc"))
    aug_sources = load_augmentation_sources(os.path.join(base_dir, "metadata.csv"))

    groups = defaultdict(list)   # group_key -> [{'filename':..., 'class_id':...}]
    group_class = {}

    for f in npy_files:
        basename = os.path.basename(f)[:-len('.npy.enc')]
        class_name = parse_class(basename, domain.classes)
        if class_name is None:
            continue
        class_id = class_to_id[class_name]

        mp4_name = f"{basename}.mp4"
        source_mp4 = aug_sources.get(mp4_name)
        group_key = os.path.splitext(source_mp4)[0] if source_mp4 else basename

        groups[group_key].append({'filename': os.path.basename(f), 'class_id': class_id})
        group_class[group_key] = class_id

    group_keys = list(groups.keys())
    group_labels = [group_class[k] for k in group_keys]

    class_counts = defaultdict(int)
    for c in group_labels:
        class_counts[c] += 1
    too_small = [domain.classes[c] for c, n in class_counts.items() if n < 3]
    if too_small:
        print(f"Warning: these classes have fewer than 3 independent source videos "
              f"and will produce an unreliable split: {too_small}")

    if len(group_keys) < 10:
        print(f"[{domain.name}] Not enough independent source videos to split.")
        return

    train_keys, temp_keys, train_y, temp_y = train_test_split(
        group_keys, group_labels, test_size=0.30, stratify=group_labels, random_state=42
    )
    val_keys, test_keys, _, _ = train_test_split(
        temp_keys, temp_y, test_size=0.50, stratify=temp_y, random_state=42
    )

    def write_split(path, keys):
        rows = [item for k in keys for item in groups[k]]
        with open(path, 'w', newline='') as f:
            writer = csv.DictWriter(f, fieldnames=['filename', 'class_id'])
            writer.writeheader()
            writer.writerows(rows)
        return len(rows)

    n_train = write_split(os.path.join(base_dir, 'train_labels.csv'), train_keys)
    n_val = write_split(os.path.join(base_dir, 'val_labels.csv'), val_keys)
    n_test = write_split(os.path.join(base_dir, 'test_labels.csv'), test_keys)

    print(f"\n--- Phase 5 Completed ({domain.name}) ---")
    print(f"Independent source videos: {len(group_keys)} "
          f"(train groups={len(train_keys)}, val groups={len(val_keys)}, test groups={len(test_keys)})")
    print(f"Generated splits: {n_train} train samples, {n_val} val samples, {n_test} test samples.")


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="Phase 5: Train/val/test split")
    parser.add_argument("--domain", required=True, choices=DOMAIN_NAMES)
    args = parser.parse_args()
    split_dataset(get_domain(args.domain))
