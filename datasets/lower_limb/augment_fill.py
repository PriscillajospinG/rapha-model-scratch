import os
import glob
import random
import csv
import hashlib
from datetime import datetime
import cv2
import numpy as np

BASE_DIR = "datasets/lower_limb"
RAW_DIR = os.path.join(BASE_DIR, "raw")
METADATA_CSV = os.path.join(BASE_DIR, "metadata.csv")
TARGET = 50
CLASSES = ["ankle", "calf", "hamstring", "heel_slide", "hip", "knee", "leg_raise", "quadriceps", "toes"]

def get_file_hash(filepath):
    hasher = hashlib.md5()
    try:
        with open(filepath, 'rb') as afile:
            buf = afile.read(65536)
            while len(buf) > 0:
                hasher.update(buf)
                buf = afile.read(65536)
        return hasher.hexdigest()
    except Exception:
        return None

def augment_video_cv2(input_path, output_path):
    aug_type = random.choice(["hflip", "brightness"])
    print(f"Applying {aug_type} to {os.path.basename(input_path)} -> {os.path.basename(output_path)}")
    
    cap = cv2.VideoCapture(input_path)
    if not cap.isOpened():
        print(f"Failed to open {input_path}")
        return None
        
    fps = cap.get(cv2.CAP_PROP_FPS)
    w = int(cap.get(cv2.CAP_PROP_FRAME_WIDTH))
    h = int(cap.get(cv2.CAP_PROP_FRAME_HEIGHT))
    total_frames = int(cap.get(cv2.CAP_PROP_FRAME_COUNT))
    
    fourcc = cv2.VideoWriter_fourcc(*'mp4v')
    out = cv2.VideoWriter(output_path, fourcc, fps, (w, h))
    
    while True:
        ret, frame = cap.read()
        if not ret:
            break
            
        if aug_type == "hflip":
            frame = cv2.flip(frame, 1)
        elif aug_type == "brightness":
            frame = cv2.convertScaleAbs(frame, alpha=1.05, beta=15)
            
        out.write(frame)
        
    cap.release()
    out.release()
    
    duration = total_frames / fps if fps > 0 else 10.0
    return aug_type, duration, fps, w, h

# Load existing metadata to append to it
with open(METADATA_CSV, 'a', newline='', encoding='utf-8') as f:
    writer = csv.DictWriter(f, fieldnames=['filename', 'class', 'duration_seconds', 'fps', 'width', 'height', 'source_url', 'hash', 'download_date', 'status', 'reason'])
    
    for c in CLASSES:
        class_dir = os.path.join(RAW_DIR, c)
        videos = glob.glob(os.path.join(class_dir, "*.mp4"))
        count = len(videos)
        
        if count >= TARGET:
            continue
            
        needed = TARGET - count
        print(f"Class {c} needs {needed} more videos.")
        
        if count == 0:
            print(f"Warning: No videos to augment for {c}!")
            continue
            
        for i in range(needed):
            src_video = random.choice(videos)
            idx = count + i + 1
            new_filename = f"{c}_aug_{idx:04d}.mp4"
            dest_video = os.path.join(class_dir, new_filename)
            
            result = augment_video_cv2(src_video, dest_video)
            if not result: continue
            
            aug_type, duration, fps, w, h = result
            new_hash = get_file_hash(dest_video)
            
            writer.writerow({
                'filename': new_filename,
                'class': c,
                'duration_seconds': round(duration, 2),
                'fps': round(fps, 2),
                'width': w,
                'height': h,
                'source_url': f"augmented_from_{os.path.basename(src_video)}_{aug_type}",
                'hash': new_hash,
                'download_date': datetime.now().isoformat(),
                'status': 'accept',
                'reason': 'Augmented to fill dataset quota'
            })
            
print("Augmentation complete!")
