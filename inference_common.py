"""
Shared inference logic used by both realtime_infer.py (continuous live
webcam display, single domain) and stroke_assessment.py (one capture window
per domain, combined into a multi-domain report). Keeping this in one place
means both call paths apply the exact same preprocessing, calibration, and
confidence gating.
"""
import os
import json
from dataclasses import dataclass

import numpy as np
import onnxruntime as ort

from pipeline_common import build_tensor

DEFAULT_POSE_MODEL_ASSET = "pose_landmarker_heavy.task"


@dataclass
class InferenceSession:
    domain: object          # domains.base.DomainConfig
    session: object         # onnxruntime.InferenceSession
    input_name: str
    classes: list
    temperature: float
    threshold: float


def load_session(domain, model_dir=None):
    model_dir = model_dir or os.path.join("models", domain.name)
    meta_path = os.path.join(model_dir, "deployment_metadata.json")
    if not os.path.exists(meta_path):
        raise FileNotFoundError(
            f"{meta_path} not found. Run `python phase_8_9_export.py --domain {domain.name}` "
            f"first so calibration/threshold info is available -- running without it means "
            f"unreliable raw softmax confidence."
        )
    with open(meta_path) as f:
        meta = json.load(f)

    classes = [meta["class_mapping"][str(i)] for i in range(len(meta["class_mapping"]))]
    temperature = float(meta.get("calibration", {}).get("temperature", 1.0))
    threshold = float(meta.get("calibration", {}).get("recommended_confidence_threshold", 0.5))

    onnx_path = os.path.join(model_dir, "best_model.onnx")
    sess = ort.InferenceSession(onnx_path, providers=["CPUExecutionProvider"])
    input_name = sess.get_inputs()[0].name

    return InferenceSession(domain=domain, session=sess, input_name=input_name,
                             classes=classes, temperature=temperature, threshold=threshold)


def _softmax(x):
    e = np.exp(x - np.max(x))
    return e / e.sum()


def extract_frame_joints(mp_image, detector, domain):
    """Run the pose detector on one frame and return this domain's joints
    (x, y, z, visibility), or NaNs if nobody was detected."""
    result = detector.detect(mp_image)
    if result.pose_landmarks and len(result.pose_landmarks) > 0:
        lm = result.pose_landmarks[0]
        return [[lm[i].x, lm[i].y, lm[i].z,
                 lm[i].visibility if lm[i].visibility is not None else 0.0]
                for i in domain.joint_indices]
    return [[np.nan, np.nan, np.nan, 0.0]] * domain.num_joints


def classify_window(frames, inference_session, min_visible_fraction=0.4):
    """
    frames: list/array of shape (N, num_joints, 4), as accumulated over a
    capture window.
    Returns dict: {label, confidence, uncertain: bool, qc}
    label is one of inference_session.classes, or "uncertain".
    """
    tensor, qc = build_tensor(np.asarray(frames), inference_session.domain)

    if qc["visible_fraction"] < min_visible_fraction:
        return {"label": "uncertain", "confidence": 0.0,
                "uncertain": True, "reason": "person/region not clearly visible", "qc": qc}

    onnx_input = tensor[None, ...].astype(np.float32)  # (1, 4, T, V, 1)
    logits = inference_session.session.run(None, {inference_session.input_name: onnx_input})[0][0]
    probs = _softmax(logits / inference_session.temperature)
    pred_idx = int(np.argmax(probs))
    confidence = float(probs[pred_idx])

    if confidence < inference_session.threshold:
        return {"label": "uncertain", "confidence": confidence,
                "uncertain": True, "reason": "confidence below calibrated threshold", "qc": qc}

    return {"label": inference_session.classes[pred_idx], "confidence": confidence,
            "uncertain": False, "reason": None, "qc": qc}
