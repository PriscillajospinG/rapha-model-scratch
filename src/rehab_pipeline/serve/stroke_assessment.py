"""
Combined stroke rehab assessment: an ORCHESTRATION layer over the
per-domain models (upper_body, lower_limb), not a separate 5th class list.
Stroke assessment/rehab isn't its own body region -- clinically it's an
integrated read across face (droop), arm (weakness), and leg (gait), similar
in spirit to the FAST test. This runs whichever domain models are available
in sequence and combines them into one report.

The face domain is intentionally not included yet -- see README.md's
"Known limitations" for why (facial droop assessment needs a different
model, MediaPipe FaceLandmarker + a symmetry score, not the same CTR-GCN
classifier pattern as the limb domains). Add it to the domain registry
(domains/face.py + domains/__init__.py) once it exists.

Usage:
    rehab-stroke-assessment --domains upper_body lower_limb
    rehab-stroke-assessment --domains upper_body lower_limb face  # face reported as unavailable

This produces a screening report, not a diagnosis. See the disclaimer field
in its own output.
"""
import argparse
import json
from datetime import datetime

from ..domains import DOMAIN_NAMES
from .realtime import run as run_realtime_once
from ..common.inference import DEFAULT_POSE_MODEL_ASSET

DISCLAIMER = (
    "This is an automated screening report generated from short webcam captures "
    "of a single exercise per body region. It is NOT a validated clinical stroke "
    "assessment. Labels reflect a movement-classification model trained on "
    "YouTube exercise-demo footage (see README.md 'Known limitations') and "
    "should be reviewed by a clinician, not acted on directly."
)


class _Args:
    def __init__(self, domain, source, window_seconds, pose_model):
        self.domain = domain
        self.source = source
        self.model_dir = None
        self.pose_model = pose_model
        self.window_seconds = window_seconds
        self.interval = 1.0
        self.min_frames = 30
        self.smoothing_window = 3
        self.display = False
        self.once = True
        self.capture_dir = None


def run_assessment(domains, source="0", window_seconds=6.0, pose_model=DEFAULT_POSE_MODEL_ASSET,
                    interactive=True):
    report = {"timestamp": datetime.now().isoformat(), "disclaimer": DISCLAIMER, "results": {}}

    requested = list(domains)
    if "face" in requested:
        report["results"]["face"] = {
            "status": "not_available",
            "reason": "face domain is not built yet -- see README.md Known limitations",
        }
        requested = [d for d in requested if d != "face"]

    for domain_name in requested:
        if domain_name not in DOMAIN_NAMES:
            report["results"][domain_name] = {"status": "error", "reason": "unknown domain"}
            continue

        if interactive:
            input(f"\nReady to assess {domain_name.replace('_', ' ')}. Perform the requested "
                  f"movement now, then press Enter to start a {window_seconds:.0f}s capture...")

        try:
            label, confidence = run_realtime_once(_Args(domain_name, source, window_seconds, pose_model))
            report["results"][domain_name] = {"status": "ok", "label": label, "confidence": confidence}
        except FileNotFoundError as e:
            report["results"][domain_name] = {"status": "error", "reason": str(e)}

    return report


def main():
    parser = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    parser.add_argument("--domains", nargs="+", default=["upper_body", "lower_limb"],
                         choices=DOMAIN_NAMES + ["face"])
    parser.add_argument("--source", default="0")
    parser.add_argument("--window-seconds", type=float, default=6.0)
    parser.add_argument("--pose-model", default=DEFAULT_POSE_MODEL_ASSET)
    parser.add_argument("--non-interactive", dest="interactive", action="store_false",
                         help="Don't wait for Enter between domains (assumes you're already ready)")
    parser.add_argument("--output", default=None, help="Optional path to save the JSON report")
    args = parser.parse_args()

    report = run_assessment(args.domains, args.source, args.window_seconds,
                             args.pose_model, args.interactive)
    print("\n=== Stroke assessment report ===")
    print(json.dumps(report, indent=2))
    if args.output:
        with open(args.output, "w") as f:
            json.dump(report, f, indent=2)
        print(f"Saved to {args.output}")


if __name__ == "__main__":
    main()
