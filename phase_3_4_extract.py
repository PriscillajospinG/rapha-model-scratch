"""
Phase 3 & 4: Extraction.

Usage:
    python phase_3_4_extract.py --domain lower_limb
    python phase_3_4_extract.py --domain upper_body

Reads ONLY from datasets/<domain>/raw/<class>/ -- i.e. only videos a human
already confirmed via review_app.py. For each video:
  1. Run MediaPipe PoseLandmarker to get this domain's joints per frame.
  2. Interpolate gaps, resample to a fixed frame count, normalize
     (center/scale per the domain's config) via pipeline_common.
  3. Quality-gate on visibility -- clips where the relevant body region was
     barely visible get dropped instead of silently entering training.
  4. Encrypt the resulting tensor and write it to skeletons/ as .npy.enc.
  5. Delete the source video. Raw video is never retained past extraction --
     only the encrypted skeleton tensor is.
"""
import os
import glob
import json
import argparse

import cv2
import numpy as np
import mediapipe as mp
from mediapipe.tasks import python
from mediapipe.tasks.python import vision

from domains import get_domain, DOMAIN_NAMES
from pipeline_common import build_tensor, validate_tensor, MIN_VISIBLE_FRAME_FRACTION
from crypto_utils import save_encrypted_npy

POSE_MODEL_ASSET = "pose_landmarker_heavy.task"


def extract_landmarks(video_path, detector, joint_indices):
    cap = cv2.VideoCapture(video_path)
    frames_data = []

    while cap.isOpened():
        ret, frame = cap.read()
        if not ret:
            break

        frame_rgb = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)
        mp_image = mp.Image(image_format=mp.ImageFormat.SRGB, data=frame_rgb)
        results = detector.detect(mp_image)

        if results.pose_landmarks and len(results.pose_landmarks) > 0:
            landmarks = results.pose_landmarks[0]
            frame_joints = [
                [landmarks[idx].x, landmarks[idx].y, landmarks[idx].z,
                 landmarks[idx].visibility if landmarks[idx].visibility is not None else 0.0]
                for idx in joint_indices
            ]
            frames_data.append(frame_joints)
        else:
            frames_data.append([[np.nan, np.nan, np.nan, 0.0]] * len(joint_indices))

    cap.release()
    return np.array(frames_data) if frames_data else None


def process_dataset(domain):
    base_dir = os.path.join("datasets", domain.name)
    raw_dir = os.path.join(base_dir, "raw")
    skeleton_dir = os.path.join(base_dir, "skeletons")
    os.makedirs(skeleton_dir, exist_ok=True)

    print(f"[{domain.name}] Loading MediaPipe PoseLandmarker model...")
    base_options = python.BaseOptions(model_asset_path=POSE_MODEL_ASSET)
    options = vision.PoseLandmarkerOptions(
        base_options=base_options,
        running_mode=vision.RunningMode.IMAGE,
        output_segmentation_masks=False)

    stats = {'total': 0, 'successful': 0, 'needs_review': 0, 'corrupted': 0}
    validation_report = []

    with vision.PoseLandmarker.create_from_options(options) as detector:
        for c in domain.classes:
            videos = glob.glob(os.path.join(raw_dir, c, "*.mp4"))
            for vid in videos:
                stats['total'] += 1
                filename = os.path.basename(vid)
                base_name = os.path.splitext(filename)[0]
                out_file = os.path.join(skeleton_dir, f"{base_name}.npy.enc")

                if os.path.exists(out_file):
                    continue

                print(f"Extracting: {filename}")
                raw_data = extract_landmarks(vid, detector, domain.joint_indices)

                if raw_data is None:
                    stats['corrupted'] += 1
                    validation_report.append({'file': filename, 'status': 'corrupted',
                                               'reason': 'No frames read from video'})
                    os.remove(vid)
                    continue

                tensor, qc = build_tensor(raw_data, domain)
                is_valid, msg = validate_tensor(tensor, domain)

                if not is_valid:
                    stats['corrupted'] += 1
                    validation_report.append({'file': filename, 'status': 'corrupted',
                                               'reason': msg, **qc})
                    os.remove(vid)
                    continue

                if qc['visible_fraction'] < MIN_VISIBLE_FRAME_FRACTION:
                    stats['needs_review'] += 1
                    validation_report.append({
                        'file': filename, 'status': 'needs_review',
                        'reason': f"Relevant joints visible in only {qc['visible_fraction']:.0%} "
                                  f"of frames (threshold {MIN_VISIBLE_FRAME_FRACTION:.0%})",
                        **qc,
                    })
                    os.remove(vid)
                    continue

                save_encrypted_npy(tensor, out_file)
                stats['successful'] += 1
                validation_report.append({'file': filename, 'status': 'accepted', **qc})
                os.remove(vid)  # skeleton-only retention: raw video deleted after extraction

    with open(os.path.join(base_dir, 'tensor_statistics.json'), 'w') as f:
        json.dump(stats, f, indent=4)

    with open(os.path.join(base_dir, 'validation_report.json'), 'w') as f:
        json.dump(validation_report, f, indent=4)

    print(f"\n--- Phase 3 & 4 Completed ({domain.name}) ---")
    print(f"Generated {stats['successful']} valid encrypted skeleton tensors.")
    print(f"Flagged {stats['needs_review']} clips as insufficient visibility (dropped).")
    print(f"Corrupted: {stats['corrupted']}")


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="Phase 3 & 4: Skeleton extraction")
    parser.add_argument("--domain", required=True, choices=DOMAIN_NAMES)
    args = parser.parse_args()
    process_dataset(get_domain(args.domain))
