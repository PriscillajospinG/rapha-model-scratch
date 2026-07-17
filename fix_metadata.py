import os
import csv

BASE_DIR = "datasets/lower_limb"
RAW_DIR = os.path.join(BASE_DIR, "raw")
METADATA_CSV = os.path.join(BASE_DIR, "metadata.csv")

if not os.path.exists(METADATA_CSV):
    print("No metadata.csv found.")
    exit(0)

with open(METADATA_CSV, 'r') as f:
    reader = csv.DictReader(f)
    fieldnames = reader.fieldnames
    rows = list(reader)

for row in rows:
    if row['status'] != 'accept':
        continue
    c = row['class']
    fname = row['filename']
    filepath = os.path.join(RAW_DIR, c, fname)
    if not os.path.exists(filepath):
        print(f"Marking as duplicate: {fname}")
        row['status'] = 'duplicate'
        row['reason'] = 'Removed by strict 3-frame duplicate checker'

with open(METADATA_CSV, 'w', newline='') as f:
    writer = csv.DictWriter(f, fieldnames=fieldnames)
    writer.writeheader()
    writer.writerows(rows)

print("Fixed metadata.csv")
