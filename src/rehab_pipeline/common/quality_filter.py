"""
Automated pre-filter for freshly downloaded videos, run BEFORE a human ever
looks at them. This existed in an earlier version of this pipeline
(collect_dataset.py) but was disabled in the version that replaced it in
favor of an unconditional auto-accept. It's restored here as a triage step:
it narrows down what a human reviewer has to look at, it never makes the
final accept/reject call by itself.

Heuristics:
  - reject outright if 0 or >1 person is visible (wrong content / not solo)
  - flag for review if the domain's body-region keypoints aren't clearly visible
  - flag for review if camera motion is high (handheld/shaky footage that
    will make skeleton extraction unreliable)
  - otherwise: pass through to human review as a normal candidate
"""
import cv2
import numpy as np
from ultralytics import YOLO

_person_model = None
_pose_model = None


def _get_models():
    global _person_model, _pose_model
    if _person_model is None:
        _person_model = YOLO("yolov8n.pt")
        _pose_model = YOLO("yolov8n-pose.pt")
    return _person_model, _pose_model


def heuristic_screen(filepath, domain, sample_rate=5):
    """Returns (status, reason) where status is one of:
    'candidate'  -- passed triage, send to human review as normal priority
    'low_confidence' -- send to human review but flagged as likely borderline
    'reject'     -- discard automatically, never shown to a reviewer

    `domain` is a domains.base.DomainConfig -- which body-region keypoints
    we check for visibility (legs for lower_limb, arms for upper_body, etc.)
    comes from domain.yolo_pose_keypoint_range (YOLO pose uses COCO ordering).
    """
    person_model, pose_model = _get_models()
    kp_lo, kp_hi = domain.yolo_pose_keypoint_range

    cap = cv2.VideoCapture(filepath)
    if not cap.isOpened():
        return 'reject', 'Cannot open video file'

    max_people_seen = 0
    region_keypoints_found = False
    prev_gray = None
    motion_magnitudes = []
    frame_idx = 0

    while True:
        ret, frame = cap.read()
        if not ret:
            break
        frame_idx += 1
        if frame_idx % sample_rate != 0:
            continue

        results = person_model(frame, verbose=False, classes=[0])
        persons_in_frame = len(results[0].boxes)
        max_people_seen = max(max_people_seen, persons_in_frame)

        if not region_keypoints_found and persons_in_frame > 0:
            pose_results = pose_model(frame, verbose=False)
            if len(pose_results[0].keypoints) > 0:
                kpts = pose_results[0].keypoints.data[0]
                if len(kpts) >= kp_hi:
                    region_conf = kpts[kp_lo:kp_hi, 2]
                    if (region_conf > 0.5).sum() >= 2:
                        region_keypoints_found = True

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
    if not region_keypoints_found:
        return 'low_confidence', f'{domain.name} keypoints not clearly visible'

    avg_motion = float(np.mean(motion_magnitudes)) if motion_magnitudes else 0.0
    if avg_motion > 5.0:
        return 'low_confidence', f'High camera motion ({avg_motion:.2f})'

    return 'candidate', 'Passed automated triage'
