"""
Phase 1 & 2: Collection.

Usage:
    python phase_1_2_collect.py --domain lower_limb --target 50
    python phase_1_2_collect.py --domain upper_body --target 50

This does NOT auto-accept videos into the training set. Every downloaded
candidate is:
  1. Deduplicated (file hash + 3-frame perceptual hash).
  2. Screened by an automated heuristic (quality_filter.py) that rejects
     obvious junk (no person / multiple people) and flags borderline clips,
     checking the body-region keypoints relevant to this domain.
  3. Anything not auto-rejected is moved to datasets/<domain>/pending_review/
     for a HUMAN to confirm or correct the label before it ever reaches the
     training set. Run `python review_app.py --domain <domain>` to work
     through the queue.

The label attached at download time is only a *hypothesis* based on the
search query used to find the video -- it is not trustworthy on its own.
"""
import os
import json
import csv
import hashlib
import argparse
import glob
import shutil
from datetime import datetime

import cv2
import yt_dlp
from PIL import Image
import imagehash

from domains import get_domain, DOMAIN_NAMES
from quality_filter import heuristic_screen


def get_file_hash(filepath):
    hasher = hashlib.md5()
    try:
        with open(filepath, 'rb') as afile:
            for buf in iter(lambda: afile.read(65536), b''):
                hasher.update(buf)
        return hasher.hexdigest()
    except Exception:
        return None


def get_video_phash(filepath):
    """Perceptual hash from 3 frames (25%/50%/75% through the clip) rather
    than just the midpoint -- a single-frame hash misses near-duplicate
    videos that share a middle frame but differ elsewhere."""
    try:
        cap = cv2.VideoCapture(filepath)
        if not cap.isOpened():
            return None
        total_frames = int(cap.get(cv2.CAP_PROP_FRAME_COUNT))
        if total_frames < 3:
            cap.release()
            return None
        hashes = []
        for frac in (0.25, 0.5, 0.75):
            cap.set(cv2.CAP_PROP_POS_FRAMES, int(total_frames * frac))
            ret, frame = cap.read()
            if ret:
                img = Image.fromarray(cv2.cvtColor(frame, cv2.COLOR_BGR2RGB))
                hashes.append(str(imagehash.phash(img)))
        cap.release()
        return "-".join(hashes) if len(hashes) == 3 else None
    except Exception:
        return None


def download_and_process(domain, target_per_class, run_heuristic_screen=True):
    base_dir = os.path.join("datasets", domain.name)
    pending_dir = os.path.join(base_dir, "pending_review")
    for c in domain.classes:
        os.makedirs(os.path.join(pending_dir, c), exist_ok=True)

    print(f"[{domain.name}] Starting collection (all candidates go to pending_review, "
          f"none are auto-accepted)...")

    metadata_file = os.path.join(base_dir, "metadata.csv")
    file_exists = os.path.exists(metadata_file)

    hash_registry, url_registry = set(), set()
    if file_exists:
        with open(metadata_file, 'r', encoding='utf-8') as f:
            for row in csv.DictReader(f):
                if row.get('hash'):
                    hash_registry.add(row['hash'])
                if row.get('source_url'):
                    url_registry.add(row['source_url'])

    csv_file = open(metadata_file, 'a', newline='', encoding='utf-8')
    fieldnames = ['filename', 'query_class', 'duration_seconds', 'fps', 'width', 'height',
                  'source_url', 'hash', 'download_date', 'status', 'reason']
    writer = csv.DictWriter(csv_file, fieldnames=fieldnames)
    if not file_exists:
        writer.writeheader()

    stats = {c: {'pending': len(glob.glob(os.path.join(pending_dir, c, "*.mp4"))),
                 'new': 0, 'auto_rejected': 0} for c in domain.classes}

    for class_name in domain.classes:
        queries = domain.queries[class_name]
        print(f"\n--- Processing class: {class_name} ---")
        needed = target_per_class - stats[class_name]['pending']
        if needed <= 0:
            print(f"Already have {stats[class_name]['pending']}/{target_per_class} pending for {class_name}")
            continue

        ydl_opts = {
            'format': 'best[height<=720][ext=mp4]/best[ext=mp4]',
            'outtmpl': os.path.join(base_dir, 'temp_%(id)s.%(ext)s'),
            'noplaylist': True,
            'quiet': True,
            'match_filter': yt_dlp.utils.match_filter_func("duration >= 3 & duration <= 300"),
            'download_archive': os.path.join(base_dir, 'download_archive.txt'),
        }

        for query in queries:
            if stats[class_name]['pending'] >= target_per_class:
                break
            needed = target_per_class - stats[class_name]['pending']
            search_query = f"ytsearch{needed * 2}:{query}"

            try:
                with yt_dlp.YoutubeDL(ydl_opts) as ydl:
                    print(f"Query: '{query}' -> searching for {needed} candidates...")
                    ydl.extract_info(search_query, download=True)
            except Exception:
                continue

            for temp_file in glob.glob(os.path.join(base_dir, "temp_*.mp4")):
                if stats[class_name]['pending'] >= target_per_class:
                    os.remove(temp_file)
                    continue

                vid_id = temp_file.split('temp_')[-1].replace('.mp4', '')
                source_url = f"https://www.youtube.com/watch?v={vid_id}"
                if source_url in url_registry:
                    os.remove(temp_file)
                    continue

                fhash = get_file_hash(temp_file)
                phash = get_video_phash(temp_file)
                if (fhash and fhash in hash_registry) or (phash and phash in hash_registry):
                    os.remove(temp_file)
                    continue
                if fhash:
                    hash_registry.add(fhash)
                if phash:
                    hash_registry.add(phash)

                cap = cv2.VideoCapture(temp_file)
                if not cap.isOpened():
                    os.remove(temp_file)
                    continue
                duration = cap.get(cv2.CAP_PROP_FRAME_COUNT) / max(1, cap.get(cv2.CAP_PROP_FPS))
                fps = cap.get(cv2.CAP_PROP_FPS)
                width = cap.get(cv2.CAP_PROP_FRAME_WIDTH)
                height = cap.get(cv2.CAP_PROP_FRAME_HEIGHT)
                cap.release()

                if run_heuristic_screen:
                    try:
                        status, reason = heuristic_screen(temp_file, domain)
                    except Exception as e:
                        status, reason = 'candidate', f'Heuristic screen failed ({e}), sending to human review'
                else:
                    status, reason = 'candidate', 'Heuristic screening disabled'

                if status == 'reject':
                    os.remove(temp_file)
                    stats[class_name]['auto_rejected'] += 1
                    writer.writerow({
                        'filename': f'{vid_id}.mp4', 'query_class': class_name,
                        'duration_seconds': round(duration, 2), 'fps': round(fps, 2),
                        'width': width, 'height': height, 'source_url': source_url,
                        'hash': fhash, 'download_date': datetime.now().isoformat(),
                        'status': 'auto_rejected', 'reason': reason,
                    })
                    csv_file.flush()
                    continue

                # candidate or low_confidence -> goes to a human, video is kept
                # only until reviewed (see review_app.py for deletion/promotion).
                stats[class_name]['pending'] += 1
                stats[class_name]['new'] += 1
                idx = stats[class_name]['pending']
                final_name = f"{class_name}_{idx:04d}_{vid_id}.mp4"
                dest_path = os.path.join(pending_dir, class_name, final_name)
                shutil.move(temp_file, dest_path)
                url_registry.add(source_url)

                writer.writerow({
                    'filename': final_name, 'query_class': class_name,
                    'duration_seconds': round(duration, 2), 'fps': round(fps, 2),
                    'width': width, 'height': height, 'source_url': source_url,
                    'hash': fhash, 'download_date': datetime.now().isoformat(),
                    'status': f'pending_review:{status}', 'reason': reason,
                })
                csv_file.flush()

            for f in glob.glob(os.path.join(base_dir, "temp_*")):
                os.remove(f)

    csv_file.close()

    report = {
        'domain': domain.name,
        'total_target': target_per_class * len(domain.classes),
        'pending_review_per_class': {c: stats[c]['pending'] for c in domain.classes},
        'newly_downloaded': sum(stats[c]['new'] for c in domain.classes),
        'auto_rejected': sum(stats[c]['auto_rejected'] for c in domain.classes),
        'note': 'Nothing here is in the training set yet. Run review_app.py to confirm labels.',
    }
    with open(os.path.join(base_dir, 'dataset_report.json'), 'w') as f:
        json.dump(report, f, indent=4)

    print(f"\n--- Phase 1 & 2 Completed ({domain.name}) ---")
    print(f"New candidates awaiting human review: {sum(stats[c]['new'] for c in domain.classes)}")
    print(f"Auto-rejected by heuristic screen: {sum(stats[c]['auto_rejected'] for c in domain.classes)}")
    print(f"Run `python review_app.py --domain {domain.name}` to confirm labels before training.")


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="Phase 1 & 2: Dataset Collection")
    parser.add_argument("--domain", required=True, choices=DOMAIN_NAMES)
    parser.add_argument("--target", type=int, default=150,
                         help="Target number of pending-review candidates per class")
    parser.add_argument("--no-heuristic-screen", action="store_true",
                         help="Skip the automated YOLO triage (not recommended -- sends everything to human review)")
    args = parser.parse_args()
    download_and_process(get_domain(args.domain), args.target,
                          run_heuristic_screen=not args.no_heuristic_screen)
