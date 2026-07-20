"""
Phase 8 & 9: Export best trained CTR-GCN model to:
  - best_model.pth        (PyTorch weights)
  - best_model.onnx       (ONNX for deployment)
  - deployment_metadata.json

Run this AFTER training is complete on the L4 GPU:
  python phase_8_9_export.py
"""
import os
import json
import torch
import torch.nn as nn
from phase_6_7_train import CTRGCN, CLASSES

BASE_DIR   = "datasets/lower_limb"
MODEL_DIR  = "models/lower_limb"
EVAL_DIR   = "evaluation/lower_limb"
EXPORT_DIR = "models/lower_limb"

os.makedirs(EXPORT_DIR, exist_ok=True)

def export_and_report():
    print("\n--- Phase 8 & 9: Export and Report ---")

    model_path = os.path.join(MODEL_DIR, "best_model.pth")
    if not os.path.exists(model_path):
        print("❌  Error: best_model.pth not found. Train the model first (python phase_6_7_train.py).")
        return

    # ── Load model on CPU for portable export ──────────────────────────
    device = torch.device("cpu")
    model  = CTRGCN(num_class=len(CLASSES), in_channels=7).to(device)
    model.load_state_dict(torch.load(model_path, map_location=device))
    model.eval()
    print(f"✅  Loaded weights from {model_path}")

    # ── 1. PTH: already exists – just confirm and print size ───────────
    pth_size = os.path.getsize(model_path) / (1024 * 1024)
    print(f"✅  PTH file: {model_path}  ({pth_size:.2f} MB)")

    # ── 2. ONNX Export ─────────────────────────────────────────────────
    # Input shape: (N, C, T, V, M) = (1, 7, 300, 10, 1)
    # Channels 0-3 are root-centered/scale-normalized [x, y, z, visibility],
    # channels 4-6 are frame-to-frame velocity [vx, vy, vz] of the position
    # channels. See normalize_position()/add_velocity() in phase_6_7_train.py.
    dummy_input = torch.randn(1, 7, 300, 10, 1)
    onnx_path   = os.path.join(EXPORT_DIR, "best_model.onnx")

    print("⏳  Exporting to ONNX …")
    torch.onnx.export(
        model,
        dummy_input,
        onnx_path,
        export_params=True,
        opset_version=17,               # Opset 17 is widely supported
        do_constant_folding=True,       # Fold constants for smaller/faster model
        input_names=["skeleton_input"],
        output_names=["class_logits"],
        dynamic_axes={
            "skeleton_input": {0: "batch_size"},
            "class_logits":   {0: "batch_size"},
        },
        verbose=False,
    )
    onnx_size = os.path.getsize(onnx_path) / (1024 * 1024)
    print(f"✅  ONNX file: {onnx_path}  ({onnx_size:.2f} MB)")

    # ── 3. Validate ONNX file ──────────────────────────────────────────
    try:
        import onnx
        onnx_model = onnx.load(onnx_path)
        onnx.checker.check_model(onnx_model)
        print("✅  ONNX model validation passed.")
    except ImportError:
        print("⚠️   onnx package not installed – skipping validation. Install with: pip install onnx")
    except Exception as e:
        print(f"❌  ONNX validation failed: {e}")

    # ── 4. Optional: verify ONNX output matches PyTorch output ─────────
    try:
        import onnxruntime as ort
        import numpy as np
        sess    = ort.InferenceSession(onnx_path, providers=["CPUExecutionProvider"])
        inp     = dummy_input.numpy()
        ort_out = sess.run(None, {"skeleton_input": inp})[0]
        with torch.no_grad():
            pt_out = model(dummy_input).numpy()
        max_diff = float(np.abs(ort_out - pt_out).max())
        if max_diff < 1e-4:
            print(f"✅  ONNX vs PyTorch output match (max_diff={max_diff:.2e})")
        else:
            print(f"⚠️   Output mismatch detected (max_diff={max_diff:.2e}) – inspect before deploying.")
    except ImportError:
        print("ℹ️   onnxruntime not installed – skipping output verification.")
    except Exception as e:
        print(f"⚠️   ONNX runtime check failed: {e}")

    # ── 5. Deployment metadata ─────────────────────────────────────────
    metrics_path = os.path.join(EVAL_DIR, "metrics.json")
    metrics      = {}
    if os.path.exists(metrics_path):
        with open(metrics_path) as f:
            metrics = json.load(f)

    metadata = {
        "model_name": "lower_limb_ctrgcn",
        "architecture": "CTR-GCN (Channel-wise Topology Refinement Graph Convolutional Network)",
        "num_classes": len(CLASSES),
        "class_mapping": {str(i): c for i, c in enumerate(CLASSES)},
        "input": {
            "name": "skeleton_input",
            "shape": [1, 7, 300, 10, 1],
            "layout": "(N, C, T, V, M) = (batch, channels, frames, joints, persons)",
            "channels": {
                "0": "x coordinate (root-centered on mid-hip, scale-normalized by hip-to-ankle body length)",
                "1": "y coordinate (root-centered, scale-normalized)",
                "2": "z coordinate (root-centered, scale-normalized)",
                "3": "visibility",
                "4": "x velocity (frame-to-frame delta of normalized x)",
                "5": "y velocity (frame-to-frame delta of normalized y)",
                "6": "z velocity (frame-to-frame delta of normalized z)"
            }
        },
        "output": {
            "name": "class_logits",
            "shape": [1, len(CLASSES)],
            "note": "Apply softmax to get probabilities"
        },
        "joint_mapping": {
            "0": "Left Hip (MP landmark 23)",
            "1": "Right Hip (MP landmark 24)",
            "2": "Left Knee (MP landmark 25)",
            "3": "Right Knee (MP landmark 26)",
            "4": "Left Ankle (MP landmark 27)",
            "5": "Right Ankle (MP landmark 28)",
            "6": "Left Heel (MP landmark 29)",
            "7": "Right Heel (MP landmark 30)",
            "8": "Left Foot Index (MP landmark 31)",
            "9": "Right Foot Index (MP landmark 32)"
        },
        "preprocessing": "MediaPipe PoseLandmarker (heavy model) → 10 lower-limb joints → interpolated to 300 frames → root-centered on mid-hip, scale-normalized by hip-to-ankle body length, velocity channels appended",
        "training": {
            "dataset": "450 balanced videos (50 per class, 9 classes), group-aware split (a video and its augmented derivatives never cross a split boundary)",
            "train_samples": sum(1 for _ in open(os.path.join(BASE_DIR, "train_labels.csv"))) - 1,
            "val_samples": sum(1 for _ in open(os.path.join(BASE_DIR, "val_labels.csv"))) - 1,
            "test_samples": sum(1 for _ in open(os.path.join(BASE_DIR, "test_labels.csv"))) - 1,
            "optimizer": "AdamW (lr=0.001, weight_decay=1e-4)",
            "scheduler": "CosineAnnealingLR (T_max=200)",
            "loss": "CrossEntropyLoss (label_smoothing=0.1)",
            "regularization": "Dropout(p=0.3) before classifier head; online skeleton augmentation (rotation, scale jitter, joint jitter, temporal crop) on train split only",
        },
        "performance": {
            "best_accuracy": metrics.get("best_accuracy", "N/A"),
            "best_val_accuracy": metrics.get("best_val_accuracy", "N/A"),
            "macro_f1":      metrics.get("macro_f1", "N/A"),
            "top3_accuracy": metrics.get("top_3_accuracy", "N/A"),
        },
        "files": {
            "pth":  "best_model.pth",
            "onnx": "best_model.onnx",
        }
    }
    meta_path = os.path.join(EXPORT_DIR, "deployment_metadata.json")
    with open(meta_path, "w") as f:
        json.dump(metadata, f, indent=4)
    print(f"✅  Metadata saved: {meta_path}")

    # ── 6. Final training report markdown ─────────────────────────────
    num_train = sum(1 for _ in open(os.path.join(BASE_DIR,"train_labels.csv"))) - 1
    num_val   = sum(1 for _ in open(os.path.join(BASE_DIR,"val_labels.csv")))   - 1
    num_test  = sum(1 for _ in open(os.path.join(BASE_DIR,"test_labels.csv")))  - 1

    acc      = metrics.get("best_accuracy", 0)
    val_acc  = metrics.get("best_val_accuracy", 0)
    f1       = metrics.get("macro_f1", 0)
    top3     = metrics.get("top_3_accuracy", 0)

    goals_met = acc >= 0.80 and f1 >= 0.75
    status_line = "**Status: Target metrics achieved ✅**" if goals_met else \
                  "**Status: Below target — see recommendations below ⚠️**"

    report_md = f"""# Final Training Report — Lower Limb CTR-GCN

## Dataset
| Split | Samples |
|-------|---------|
| Train | {num_train} |
| Val   | {num_val}   |
| Test  | {num_test}  |
| Total | {num_train + num_val + num_test} |

Split is group-aware: a real video and any augmented clips derived from it
always land in the same split, so the test set contains no near-duplicates
of training data.

## Model Performance
| Metric | Value |
|--------|-------|
| Best Top-1 Accuracy (test, untouched during training) | {acc:.4f} |
| Best Val Accuracy (used for checkpoint selection) | {val_acc:.4f} |
| Macro F1 (test) | {f1:.4f} |
| Top-3 Accuracy (test) | {top3:.4f} |

{status_line}

## Exported Files
| File | Location |
|------|----------|
| PyTorch weights | `{model_path}` ({pth_size:.2f} MB) |
| ONNX model | `{onnx_path}` ({onnx_size:.2f} MB) |
| Deployment metadata | `{meta_path}` |
"""
    if not goals_met:
        report_md += """
## Recommendations for Improvement
1. **Increase dataset size** — collect more real videos per class, especially for weaker classes (toes, knee, leg_raise, heel_slide)
2. **Tune hyperparameters** — try lr=0.0005, or a shallower/wider model given the small dataset
3. **Check confusion matrix** — identify which class pairs confuse the model most (previously knee/quadriceps, hip/leg_raise)
4. **Review low-visibility joints** — MediaPipe visibility scores for toe/heel landmarks are often lower quality on short clips; consider filtering or flagging low-confidence frames
"""

    with open("training_report.md", "w", encoding="utf-8") as f:
        f.write(report_md)
    print("✅  Training report saved: training_report.md")

    print("\n=== Phase 8 & 9 Complete ===")
    print(f"   PTH  → {model_path}")
    print(f"   ONNX → {onnx_path}")
    print("   Ready for deployment! 🚀")

if __name__ == "__main__":
    export_and_report()
