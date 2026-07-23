"""
Phase 8 & 9: Export best trained CTR-GCN model to ONNX + deployment metadata
(including calibration temperature + the confidence threshold below which
serve/realtime.py should report "uncertain" instead of forcing a class).

Usage:
    rehab-export --domain lower_limb
    rehab-export --domain upper_body
"""
import os
import json
import argparse
import hashlib
import subprocess
import torch

from ..domains import get_domain, DOMAIN_NAMES
from .train import build_model


def _file_hash(path):
    if not os.path.exists(path):
        return None
    h = hashlib.sha256()
    with open(path, 'rb') as f:
        for chunk in iter(lambda: f.read(65536), b''):
            h.update(chunk)
    return h.hexdigest()[:16]


def _git_commit():
    try:
        return subprocess.check_output(
            ["git", "rev-parse", "--short", "HEAD"], stderr=subprocess.DEVNULL
        ).decode().strip()
    except Exception:
        return "unknown"


def export_and_report(domain):
    print(f"\n--- Phase 8 & 9: Export and Report ({domain.name}) ---")

    base_dir = os.path.join("datasets", domain.name)
    model_dir = os.path.join("models", domain.name)
    eval_dir = os.path.join("evaluation", domain.name)

    model_path = os.path.join(model_dir, "best_model.pth")
    if not os.path.exists(model_path):
        print(f"Error: {model_path} not found. Train the model first "
              f"(rehab-train --domain {domain.name}).")
        return

    device = torch.device("cpu")
    model = build_model(domain).to(device)
    model.load_state_dict(torch.load(model_path, map_location=device))
    model.eval()
    print(f"Loaded weights from {model_path}")

    pth_size = os.path.getsize(model_path) / (1024 * 1024)
    print(f"PTH file: {model_path}  ({pth_size:.2f} MB)")

    dummy_input = torch.randn(1, 4, domain.target_frames, domain.num_joints, 1)
    onnx_path = os.path.join(model_dir, "best_model.onnx")

    print("Exporting to ONNX...")
    torch.onnx.export(
        model, dummy_input, onnx_path,
        export_params=True, opset_version=17, do_constant_folding=True,
        input_names=["skeleton_input"], output_names=["class_logits"],
        dynamic_axes={"skeleton_input": {0: "batch_size"}, "class_logits": {0: "batch_size"}},
        verbose=False,
    )
    onnx_size = os.path.getsize(onnx_path) / (1024 * 1024)
    print(f"ONNX file: {onnx_path}  ({onnx_size:.2f} MB)")

    try:
        import onnx
        onnx_model = onnx.load(onnx_path)
        onnx.checker.check_model(onnx_model)
        print("ONNX model validation passed.")
    except ImportError:
        print("onnx package not installed - skipping validation.")
    except Exception as e:
        print(f"ONNX validation failed: {e}")

    try:
        import onnxruntime as ort
        import numpy as np
        sess = ort.InferenceSession(onnx_path, providers=["CPUExecutionProvider"])
        ort_out = sess.run(None, {"skeleton_input": dummy_input.numpy()})[0]
        with torch.no_grad():
            pt_out = model(dummy_input).numpy()
        max_diff = float(np.abs(ort_out - pt_out).max())
        if max_diff < 1e-4:
            print(f"ONNX vs PyTorch output match (max_diff={max_diff:.2e})")
        else:
            print(f"WARNING: output mismatch (max_diff={max_diff:.2e}) - inspect before deploying.")
    except ImportError:
        print("onnxruntime not installed - skipping output verification.")
    except Exception as e:
        print(f"ONNX runtime check failed: {e}")

    metrics_path = os.path.join(eval_dir, "metrics.json")
    metrics = json.load(open(metrics_path)) if os.path.exists(metrics_path) else {}

    calibration_path = os.path.join(eval_dir, "calibration.json")
    calibration = json.load(open(calibration_path)) if os.path.exists(calibration_path) else {}
    if not calibration:
        print("WARNING: no calibration.json found - deploying without a calibrated confidence "
              "threshold means realtime inference cannot safely reject low-confidence predictions.")

    dataset_fingerprint = {
        split: _file_hash(os.path.join(base_dir, f"{split}_labels.csv"))
        for split in ("train", "val", "test")
    }

    metadata = {
        "model_name": f"{domain.name}_ctrgcn",
        "domain": domain.name,
        "architecture": "CTR-GCN (Channel-wise Topology Refinement Graph Convolutional Network)",
        "model_version": {
            "git_commit": _git_commit(),
            "dataset_split_fingerprint": dataset_fingerprint,
        },
        "num_classes": len(domain.classes),
        "class_mapping": {str(i): c for i, c in enumerate(domain.classes)},
        "input": {
            "name": "skeleton_input",
            "shape": [1, 4, domain.target_frames, domain.num_joints, 1],
            "layout": "(N, C, T, V, M) = (batch, channels, frames, joints, persons)",
            "channels": {"0": "x (center-joint normalized, scale normalized)",
                         "1": "y (center-joint normalized, scale normalized)",
                         "2": "z (center-joint normalized, scale normalized)",
                         "3": "visibility"},
            "note": "Inputs MUST be normalized identically to training: see "
                    "common.preprocessing.normalize_skeleton with this domain's center_joints/"
                    "scale_joints. Do not feed raw MediaPipe coordinates directly.",
        },
        "output": {
            "name": "class_logits",
            "shape": [1, len(domain.classes)],
            "note": "Divide logits by 'calibration.temperature' before softmax to get "
                    "calibrated probabilities. Treat max probability below "
                    "'calibration.recommended_confidence_threshold' as 'uncertain', not as "
                    "the argmax class -- see serve/realtime.py.",
        },
        "calibration": calibration,
        "joint_mapping": {str(i): name for i, name in enumerate(domain.joint_names)},
        "preprocessing": f"MediaPipe PoseLandmarker (heavy model) -> {domain.num_joints} "
                          f"{domain.name} joints -> interpolate to {domain.target_frames} frames "
                          f"-> center/scale normalize (common.preprocessing.build_tensor)",
        "data_handling": "Only encrypted skeleton tensors are retained; source video is "
                          "deleted after extraction. Every training label was confirmed by a "
                          "human reviewer via rehab-review (see review_log.csv) rather than "
                          "trusted from collection-time search queries.",
        "domain_notes": domain.notes,
        "performance": metrics,
        "files": {"pth": "best_model.pth", "onnx": "best_model.onnx"},
    }
    meta_path = os.path.join(model_dir, "deployment_metadata.json")
    with open(meta_path, "w") as f:
        json.dump(metadata, f, indent=4)
    print(f"Metadata saved: {meta_path}")

    num_train = sum(1 for _ in open(os.path.join(base_dir, "train_labels.csv"))) - 1
    num_val = sum(1 for _ in open(os.path.join(base_dir, "val_labels.csv"))) - 1
    num_test = sum(1 for _ in open(os.path.join(base_dir, "test_labels.csv"))) - 1

    acc = metrics.get("test_accuracy", 0)
    f1 = metrics.get("macro_f1", 0)
    top3 = metrics.get("top_3_accuracy", 0)
    goals_met = acc >= 0.80 and f1 >= 0.75
    status_line = "**Status: Target metrics achieved**" if goals_met else \
                  "**Status: Below target - see recommendations below**"

    report_md = f"""# Final Training Report — {domain.name}

## Dataset
| Split | Samples |
|-------|---------|
| Train | {num_train} |
| Val   | {num_val} |
| Test  | {num_test} |
| Total | {num_train + num_val + num_test} |

Labels were human-confirmed via rehab-review (see datasets/{domain.name}/review_log.csv),
not trusted from the YouTube search query used to find each video.
Splits are grouped by source video, so augmented (flip/brightness) clips never
leak across train/val/test.

## Model Performance (held-out test set, evaluated once)
| Metric | Value |
|--------|-------|
| Test Accuracy | {acc:.4f} |
| Macro F1 | {f1:.4f} |
| Top-3 Accuracy | {top3:.4f} |

## Calibration
| Setting | Value |
|---------|-------|
| Temperature | {calibration.get('temperature', 'N/A')} |
| Recommended confidence threshold | {calibration.get('recommended_confidence_threshold', 'N/A')} |

Predictions below the recommended threshold should be surfaced as "uncertain",
not forced to a class -- see evaluation/{domain.name}/calibration.json for the
full accuracy-vs-coverage sweep this was chosen from.

{status_line}

## Exported Files
| File | Location |
|------|----------|
| PyTorch weights | `{model_path}` ({pth_size:.2f} MB) |
| ONNX model | `{onnx_path}` ({onnx_size:.2f} MB) |
| Deployment metadata | `{meta_path}` |
| Confusion matrix | `{os.path.join(eval_dir, 'confusion_matrix.png')}` |
"""
    if domain.notes:
        report_md += f"\n## Domain-specific caveats\n{domain.notes}\n"

    if not goals_met:
        report_md += """
## Recommendations for Improvement
1. **Increase real (non-augmented) dataset size**, especially for classes flagged
   as thin in dataset_report.json.
2. **Check confusion_matrix.png** to see which specific movement pairs the model
   confuses -- that tells you where to prioritize new data collection.
3. **Re-run collection + rehab-review** for underrepresented classes.
4. **Tune hyperparameters** -- try lr=0.0005, deeper model, or a two-stream
   (joint + velocity) variant, which is standard for skeleton action recognition.
5. This dataset is still YouTube-demo footage. If the deployment population is
   real patients (impaired movement, assistive devices, non-ideal camera angles),
   validate error rates on a small sample of *that* population before trusting
   this number for anything patient-facing.
"""

    report_path = f"training_report_{domain.name}.md"
    with open(report_path, "w") as f:
        f.write(report_md)
    print(f"Training report saved: {report_path}")
    print(f"\n=== Phase 8 & 9 Complete ({domain.name}) ===")


def main():
    parser = argparse.ArgumentParser(description="Phase 8 & 9: Export + report")
    parser.add_argument("--domain", required=True, choices=DOMAIN_NAMES)
    args = parser.parse_args()
    export_and_report(get_domain(args.domain))


if __name__ == "__main__":
    main()
