import csv
import os

metadata_file = "datasets/lower_limb/metadata.csv"
temp_file = "datasets/lower_limb/metadata_temp.csv"
fieldnames = ['filename', 'class', 'duration_seconds', 'fps', 'width', 'height', 'source_url', 'hash', 'download_date', 'status', 'reason']

rows_to_keep = []
with open(metadata_file, 'r', encoding='utf-8') as f:
    reader = csv.reader(f)
    header = next(reader)
    for row in reader:
        # Map row to standard fieldnames, padding with empty strings or ignoring extras
        row_dict = {}
        for i, field in enumerate(fieldnames):
            if i < len(row):
                row_dict[field] = row[i]
            else:
                row_dict[field] = ""
        rows_to_keep.append(row_dict)

with open(temp_file, 'w', encoding='utf-8', newline='') as f:
    writer = csv.DictWriter(f, fieldnames=fieldnames)
    writer.writeheader()
    for row in rows_to_keep:
        writer.writerow(row)

os.replace(temp_file, metadata_file)
print("Metadata repaired.")
