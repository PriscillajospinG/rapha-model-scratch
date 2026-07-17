import os
import glob
import cv2
import numpy as np
from PIL import Image
import imagehash
from collections import defaultdict

BASE_DIR = "datasets/lower_limb/raw"
CLASSES = ["ankle", "calf", "hamstring", "heel_slide", "hip", "knee", "leg_raise", "quadriceps", "toes"]

def get_3_frame_hash(filepath):
    cap = cv2.VideoCapture(filepath)
    if not cap.isOpened():
        return None
        
    frames = int(cap.get(cv2.CAP_PROP_FRAME_COUNT))
    if frames < 3:
        cap.release()
        return None
        
    positions = [int(frames*0.25), int(frames*0.5), int(frames*0.75)]
    hashes = []
    
    for pos in positions:
        cap.set(cv2.CAP_PROP_POS_FRAMES, pos)
        ret, frame = cap.read()
        if ret:
            frame = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)
            img = Image.fromarray(frame)
            hashes.append(str(imagehash.phash(img)))
            
    cap.release()
    if len(hashes) == 3:
        return "-".join(hashes)
    return None

deleted = 0
for c in CLASSES:
    class_dir = os.path.join(BASE_DIR, c)
    if not os.path.exists(class_dir): continue
    
    videos = glob.glob(os.path.join(class_dir, "*.mp4"))
    seen = defaultdict(list)
    
    for v in videos:
        h = get_3_frame_hash(v)
        if h:
            seen[h].append(v)
            
    for h, v_list in seen.items():
        if len(v_list) > 1:
            for v in v_list[1:]:
                print(f"Deleting duplicate: {v}")
                os.remove(v)
                deleted += 1

print(f"Deleted {deleted} duplicates.")
