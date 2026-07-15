import os
import sys
import json
import csv
import hashlib
import cv2
import yt_dlp
import numpy as np
from PIL import Image
import imagehash
from datetime import datetime
from ultralytics import YOLO
import argparse
import shutil
import glob

CLASSES = [
    "ankle_pumps", "calf_stretch", "hamstring_stretch", "heel_slide",
    "hip_abduction", "knee_extension", "straight_leg_raise", "quadriceps_set", "toe_raise"
]

QUERIES = {
    "ankle_pumps": [
        "ankle pump physiotherapy exercise",
        "ankle dorsiflexion rehabilitation exercise",
        "ankle mobility exercise"
    ],
    "calf_stretch": [
        "calf stretch physiotherapy",
        "gastrocnemius stretch exercise",
        "standing calf stretch rehabilitation"
    ],
    "hamstring_stretch": [
        "hamstring stretch physiotherapy",
        "seated hamstring stretch",
        "hamstring rehabilitation exercise"
    ],
    "heel_slide": [
        "heel slide exercise",
        "heel slide physiotherapy",
        "post surgery heel slide exercise"
    ],
    "hip_abduction": [
        "hip abduction exercise",
        "hip strengthening physiotherapy",
        "side lying hip abduction"
    ],
    "knee_extension": [
        "knee extension exercise",
        "knee rehabilitation exercise",
        "quadriceps knee strengthening"
    ],
    "straight_leg_raise": [
        "straight leg raise exercise",
        "SLR physiotherapy exercise",
        "supine straight leg raise"
    ],
    "quadriceps_set": [
        "quadriceps set exercise",
        "quad set physiotherapy",
        "isometric quadriceps exercise"
    ],
    "toe_raise": [
        "toe raise exercise",
        "toe strengthening physiotherapy",
        "toe mobility exercise"
    ]
}

BASE_DIR = "dataset"
RAW_DIR = os.path.join(BASE_DIR, "raw")
REJECTED_DIR = os.path.join(BASE_DIR, "rejected")
REVIEW_DIR = os.path.join(BASE_DIR, "review")
METADATA_DIR = os.path.join(BASE_DIR, "metadata")
REPORTS_DIR = os.path.join(BASE_DIR, "reports")

def setup_directories():
    for d in [REJECTED_DIR, REVIEW_DIR, METADATA_DIR, REPORTS_DIR]:
        os.makedirs(d, exist_ok=True)
    for c in CLASSES:
        os.makedirs(os.path.join(RAW_DIR, c), exist_ok=True)

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

def get_video_phash(filepath):
    try:
        cap = cv2.VideoCapture(filepath)
        if not cap.isOpened():
            return None
        total_frames = int(cap.get(cv2.CAP_PROP_FRAME_COUNT))
        if total_frames <= 0:
            return None
        cap.set(cv2.CAP_PROP_POS_FRAMES, total_frames // 2)
        ret, frame = cap.read()
        cap.release()
        if ret:
            frame_rgb = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)
            img = Image.fromarray(frame_rgb)
            return str(imagehash.phash(img))
    except Exception as e:
        print(f"Error computing phash: {e}")
    return None

def analyze_video(filepath, person_model, pose_model):
    cap = cv2.VideoCapture(filepath)
    if not cap.isOpened():
        return 'reject', 'Cannot open video file'

    fps = cap.get(cv2.CAP_PROP_FPS)
    total_frames = int(cap.get(cv2.CAP_PROP_FRAME_COUNT))
    
    if fps == 0 or total_frames == 0:
        return 'reject', 'Invalid video metadata'

    duration = total_frames / fps
    if duration < 5 or duration > 120:
        return 'reject', f'Duration {duration:.1f}s not in 5-120s range'

    sample_rate = max(1, int(fps * 0.5))
    max_people_seen = 0
    leg_keypoints_found = False
    
    prev_gray = None
    motion_magnitudes = []
    
    while True:
        ret, frame = cap.read()
        if not ret:
            break
            
        frame_idx = int(cap.get(cv2.CAP_PROP_POS_FRAMES))
        if frame_idx % sample_rate != 0:
            continue
            
        # 1. Person Counting using YOLO
        results = person_model(frame, verbose=False, classes=[0])
        persons_in_frame = len(results[0].boxes)
        max_people_seen = max(max_people_seen, persons_in_frame)
        
        # 2. Pose estimation for leg visibility
        if not leg_keypoints_found and persons_in_frame > 0:
            pose_results = pose_model(frame, verbose=False)
            if len(pose_results[0].keypoints) > 0:
                kpts = pose_results[0].keypoints.data[0]
                if len(kpts) >= 17:
                    lower_body_conf = kpts[11:17, 2] 
                    if (lower_body_conf > 0.5).sum() >= 2:
                        leg_keypoints_found = True
                        
        # 3. Camera Motion
        gray = cv2.cvtColor(frame, cv2.COLOR_BGR2GRAY)
        gray = cv2.resize(gray, (320, 240))
        if prev_gray is not None:
            flow = cv2.calcOpticalFlowFarneback(prev_gray, gray, None, 0.5, 3, 15, 3, 5, 1.2, 0)
            mag, _ = cv2.cartToPolar(flow[..., 0], flow[..., 1])
            motion_magnitudes.append(np.mean(mag))
        prev_gray = gray

    cap.release()
    
    if max_people_seen > 1:
        return 'reject', 'Multiple people detected'
    if max_people_seen == 0:
        return 'reject', 'No person detected'
    if not leg_keypoints_found:
        return 'review', 'Lower body keypoints not clearly visible'
        
    avg_motion = np.mean(motion_magnitudes) if motion_magnitudes else 0
    if avg_motion > 5.0:
        return 'review', f'High camera motion ({avg_motion:.2f})'
        
    return 'accept', 'Passed basic heuristics'

def download_and_process(target_per_class):
    setup_directories()
    
    print("Loading YOLO models...")
    person_model = YOLO("yolov8n.pt")
    pose_model = YOLO("yolov8n-pose.pt")
    
    metadata_file = os.path.join(METADATA_DIR, "metadata.csv")
    file_exists = os.path.exists(metadata_file)
    
    seen_hashes = set()
    seen_phashes = set()
    
    if file_exists:
        with open(metadata_file, 'r', encoding='utf-8') as f:
            reader = csv.DictReader(f)
            for row in reader:
                pass
                
    csv_file = open(metadata_file, 'a', newline='', encoding='utf-8')
    fieldnames = ['filename', 'class', 'duration_seconds', 'fps', 'width', 'height', 'source_url', 'download_date', 'status', 'reason']
    writer = csv.DictWriter(csv_file, fieldnames=fieldnames)
    if not file_exists:
        writer.writeheader()
        
    stats = {c: {'collected': 0, 'new_collected': 0, 'rejected': 0, 'review': 0} for c in CLASSES}
    
    for class_name, queries in QUERIES.items():
        print(f"\n--- Processing class: {class_name} ---")
        
        existing = len(glob.glob(os.path.join(RAW_DIR, class_name, "*.mp4")))
        stats[class_name]['collected'] = existing
        
        needed = target_per_class - existing
        if needed <= 0:
            print(f"Already have enough for {class_name} ({existing}/{target_per_class})")
            continue
            
        print(f"Need {needed} more videos for {class_name} (Current: {existing})")
        
        ydl_opts = {
            'format': 'best[height<=720][ext=mp4]/best[ext=mp4]',
            'outtmpl': os.path.join(BASE_DIR, 'temp_%(id)s.%(ext)s'),
            'noplaylist': True,
            'quiet': True,
            'match_filter': yt_dlp.utils.match_filter_func("duration >= 5 & duration <= 120"),
            'extract_flat': False,
        }
        
        for query in queries:
            if stats[class_name]['collected'] >= target_per_class:
                break
                
            needed = target_per_class - stats[class_name]['collected']
            search_query = f"ytsearch{needed * 3}:{query}"
            
            try:
                with yt_dlp.YoutubeDL(ydl_opts) as ydl:
                    print(f"Query: '{query}' -> Searching for candidates...")
                    info = ydl.extract_info(search_query, download=True)
            except Exception as e:
                print(f"yt-dlp error: {e}")
                continue
                
            temp_files = glob.glob(os.path.join(BASE_DIR, "temp_*.mp4"))
            
            for temp_file in temp_files:
                if stats[class_name]['collected'] >= target_per_class:
                    os.remove(temp_file)
                    continue
                    
                fhash = get_file_hash(temp_file)
                phash = get_video_phash(temp_file)
                
                if (fhash and fhash in seen_hashes) or (phash and phash in seen_phashes):
                    print(f"Duplicate found, skipping...")
                    os.remove(temp_file)
                    continue
                    
                if fhash: seen_hashes.add(fhash)
                if phash: seen_phashes.add(phash)
                
                cap = cv2.VideoCapture(temp_file)
                if cap.isOpened():
                    duration = cap.get(cv2.CAP_PROP_FRAME_COUNT) / max(1, cap.get(cv2.CAP_PROP_FPS))
                    fps = cap.get(cv2.CAP_PROP_FPS)
                    width = cap.get(cv2.CAP_PROP_FRAME_WIDTH)
                    height = cap.get(cv2.CAP_PROP_FRAME_HEIGHT)
                    cap.release()
                else:
                    os.remove(temp_file)
                    continue
                    
                status, reason = analyze_video(temp_file, person_model, pose_model)
                print(f"Analysis result: {status} ({reason})")
                
                vid_id = temp_file.split('temp_')[-1].replace('.mp4', '')
                source_url = f"https://www.youtube.com/watch?v={vid_id}"
                
                if status == 'accept':
                    stats[class_name]['collected'] += 1
                    stats[class_name]['new_collected'] += 1
                    idx = stats[class_name]['collected']
                    final_name = f"{class_name}_{idx:03d}.mp4"
                    dest_path = os.path.join(RAW_DIR, class_name, final_name)
                    shutil.move(temp_file, dest_path)
                elif status == 'review':
                    stats[class_name]['review'] += 1
                    idx = stats[class_name]['review']
                    final_name = f"{class_name}_review_{idx:03d}_{vid_id}.mp4"
                    dest_path = os.path.join(REVIEW_DIR, final_name)
                    shutil.move(temp_file, dest_path)
                else:
                    stats[class_name]['rejected'] += 1
                    idx = stats[class_name]['rejected']
                    final_name = f"{class_name}_reject_{idx:03d}_{vid_id}.mp4"
                    dest_path = os.path.join(REJECTED_DIR, final_name)
                    shutil.move(temp_file, dest_path)
                    
                writer.writerow({
                    'filename': final_name,
                    'class': class_name,
                    'duration_seconds': round(duration, 2),
                    'fps': round(fps, 2),
                    'width': width,
                    'height': height,
                    'source_url': source_url,
                    'download_date': datetime.now().isoformat(),
                    'status': status,
                    'reason': reason
                })
                csv_file.flush()
                
            for f in glob.glob(os.path.join(BASE_DIR, "temp_*")):
                os.remove(f)
                
    csv_file.close()
    
    report = {
        'total_target': target_per_class * len(CLASSES),
        'videos_per_class': {c: stats[c]['collected'] for c in CLASSES},
        'rejected_videos': sum(stats[c]['rejected'] for c in CLASSES),
        'videos_pending_manual_review': sum(stats[c]['review'] for c in CLASSES),
        'duplicate_count': len(seen_hashes),
        'missing_videos_per_class': {c: target_per_class - stats[c]['collected'] for c in CLASSES}
    }
    
    report_path = os.path.join(REPORTS_DIR, 'dataset_report.json')
    with open(report_path, 'w') as f:
        json.dump(report, f, indent=4)
        
    print("\n--- FINAL OUTPUT ---")
    print("Current videos per class:")
    for c in CLASSES:
        print(f"  {c}: {stats[c]['collected']}")
    print(f"\nVideos downloaded this session: {sum(stats[c]['new_collected'] for c in CLASSES)}")
    print(f"Rejected videos: {sum(stats[c]['rejected'] for c in CLASSES)}")
    print(f"Videos requiring manual review: {sum(stats[c]['review'] for c in CLASSES)}")
    print("\nRemaining videos needed per class:")
    for c in CLASSES:
        print(f"  {c}: {max(0, target_per_class - stats[c]['collected'])}")

if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="Collect Physiotherapy Video Dataset")
    parser.add_argument("--target", type=int, default=150, help="Target number of videos per class")
    args = parser.parse_args()
    
    download_and_process(args.target)
