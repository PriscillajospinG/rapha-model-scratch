"""
Real-time exercise classification from a live camera feed, for one domain.

Usage:
    python realtime_infer.py --domain lower_limb --source 0 --window-seconds 6
    python realtime_infer.py --domain upper_body --once   # single capture, print result, exit

Design:
  - Pose is extracted per-frame with MediaPipe in IMAGE mode (the codebase's
    VIDEO/LIVE_STREAM mode previously crashed on timestamp handling -- see
    git history -- so this deliberately matches the same mode used at
    training time for consistency, not just to dodge that bug).
  - Frames are kept in a rolling buffer covering the last `--window-seconds`
    of wall-clock time (default 6s -- check datasets/<domain>/metadata.csv
    duration_seconds for your actual training clips and set this to match;
    a mismatched window length is a real source of train/serve skew that
    nothing in this script can detect for you).
  - Every `--interval` seconds (continuous mode), or once (--once, used by
    stroke_assessment.py), the buffer is classified via inference_common,
    which applies the exact preprocessing and calibrated confidence gating
    used at training/export time. Predictions below threshold are reported
    as "uncertain" rather than forced to a class.
  - Nothing is written to disk by default. Raw video frames live only in
    memory for the duration of the sliding window. Pass --capture-dir to
    opt in to saving encrypted skeleton windows (NOT video) for future
    retraining -- only do this with the subject's informed consent.
"""
import argparse
import os
import time
import uuid
from collections import deque, Counter

import cv2
import mediapipe as mp
from mediapipe.tasks import python
from mediapipe.tasks.python import vision

from domains import get_domain, DOMAIN_NAMES
from inference_common import load_session, extract_frame_joints, classify_window, DEFAULT_POSE_MODEL_ASSET


def run(args):
    domain = get_domain(args.domain)
    inf = load_session(domain, model_dir=args.model_dir)
    print(f"[{domain.name}] Loaded model. Classes={inf.classes}")
    print(f"Calibration temperature={inf.temperature:.3f}  confidence_threshold={inf.threshold:.2f}")

    base_options = python.BaseOptions(model_asset_path=args.pose_model)
    options = vision.PoseLandmarkerOptions(
        base_options=base_options,
        running_mode=vision.RunningMode.IMAGE,
        output_segmentation_masks=False,
    )
    detector = vision.PoseLandmarker.create_from_options(options)

    cap = cv2.VideoCapture(args.source if args.source != "0" else 0)
    if not cap.isOpened():
        raise RuntimeError(f"Could not open video source: {args.source}")

    buffer = deque()  # (timestamp, joints)
    recent_predictions = deque(maxlen=args.smoothing_window)
    last_infer_time = 0.0
    last_label, last_conf = "warming up", 0.0

    if args.capture_dir:
        os.makedirs(args.capture_dir, exist_ok=True)
        from crypto_utils import save_encrypted_npy
        print(f"Consented capture mode ON -- saving encrypted skeleton windows to {args.capture_dir}")

    if not args.display:
        print("Press Ctrl+C to quit." if not args.once else "")
    else:
        print("Press 'q' to quit.")

    try:
        while True:
            ret, frame = cap.read()
            if not ret:
                break
            now = time.time()

            frame_rgb = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)
            mp_image = mp.Image(image_format=mp.ImageFormat.SRGB, data=frame_rgb)
            joints = extract_frame_joints(mp_image, detector, domain)

            buffer.append((now, joints))
            while buffer and now - buffer[0][0] > args.window_seconds:
                buffer.popleft()

            ready = args.once and len(buffer) >= args.min_frames and (now - buffer[0][0]) >= args.window_seconds
            due = (not args.once) and (now - last_infer_time >= args.interval) and len(buffer) >= args.min_frames

            if ready or due:
                last_infer_time = now
                frames_only = [b[1] for b in buffer]
                result = classify_window(frames_only, inf)

                if result["uncertain"]:
                    last_label, last_conf = "uncertain", result["confidence"]
                    recent_predictions.clear()
                else:
                    recent_predictions.append(result["label"])
                    smoothed = Counter(recent_predictions).most_common(1)[0][0]
                    last_label, last_conf = smoothed, result["confidence"]

                if args.capture_dir:
                    import numpy as np
                    from pipeline_common import build_tensor
                    tensor, _ = build_tensor(np.asarray(frames_only), domain)
                    fname = os.path.join(args.capture_dir, f"{domain.name}_{uuid.uuid4().hex}.npy.enc")
                    save_encrypted_npy(tensor, fname)

                if args.once:
                    print(f"{last_label} ({last_conf:.2f})")
                    return last_label, last_conf

            if args.display:
                overlay = f"{last_label} ({last_conf:.2f})"
                cv2.putText(frame, overlay, (10, 30), cv2.FONT_HERSHEY_SIMPLEX, 0.9, (0, 255, 0), 2)
                cv2.imshow(f"Real-time {domain.name} classification", frame)
                if cv2.waitKey(1) & 0xFF == ord('q'):
                    break
            else:
                print(f"\r{last_label} ({last_conf:.2f})", end="", flush=True)
    finally:
        cap.release()
        detector.close()
        if args.display:
            cv2.destroyAllWindows()

    return last_label, last_conf


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="Real-time exercise classification")
    parser.add_argument("--domain", required=True, choices=DOMAIN_NAMES)
    parser.add_argument("--source", default="0", help="Camera index or video path/stream URL")
    parser.add_argument("--model-dir", default=None, help="Defaults to models/<domain>")
    parser.add_argument("--pose-model", default=DEFAULT_POSE_MODEL_ASSET,
                         help="Use a lighter MediaPipe model asset (e.g. pose_landmarker_lite.task) "
                              "if this can't keep up in real time on your hardware")
    parser.add_argument("--window-seconds", type=float, default=6.0,
                         help="Sliding window duration -- should match the typical clip length "
                              "in your training data (check metadata.csv duration_seconds)")
    parser.add_argument("--interval", type=float, default=1.0, help="Seconds between inferences (continuous mode)")
    parser.add_argument("--min-frames", type=int, default=30, help="Minimum buffered frames before inferring")
    parser.add_argument("--smoothing-window", type=int, default=3,
                         help="Majority-vote smoothing over the last N accepted predictions")
    parser.add_argument("--no-display", dest="display", action="store_false",
                         help="Run headless (print to console instead of opening a window)")
    parser.add_argument("--once", action="store_true",
                         help="Capture one window, classify once, print result, and exit "
                              "(used by stroke_assessment.py)")
    parser.add_argument("--capture-dir", default=None,
                         help="If set, save encrypted skeleton windows here for future retraining. "
                              "Only use with the subject's informed consent.")
    args = parser.parse_args()
    if args.once:
        args.display = False
    run(args)
