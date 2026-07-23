"""
Shared preprocessing used by every stage of the pipeline (collection,
extraction, training, real-time inference), parameterized by a DomainConfig
(see domains/base.py) so the same code works for lower_limb, upper_body, or
any future domain. Keeping this in one place is what guarantees the exact
same normalization is applied at training time and at inference time -- a
mismatch here is a silent accuracy killer.
"""
import numpy as np
from scipy.interpolate import interp1d

MIN_VISIBLE_FRAME_FRACTION = 0.5   # below this, a clip is too occluded to trust
MIN_VISIBILITY_THRESHOLD = 0.3     # per-joint-per-frame visibility floor


def interpolate_frames(data_array, target_length):
    """Fill NaN gaps (missed detections) and resample to a fixed frame count."""
    data_array = data_array.copy()
    N, V, C = data_array.shape

    for v in range(V):
        for c in range(C):
            series = data_array[:, v, c]
            nans = np.isnan(series)
            if np.all(nans):
                data_array[:, v, c] = 0.0
            elif np.any(nans):
                idx = np.arange(N)
                series[nans] = np.interp(idx[nans], idx[~nans], series[~nans])
                data_array[:, v, c] = series

    original_indices = np.linspace(0, N - 1, num=N)
    target_indices = np.linspace(0, N - 1, num=target_length)
    interpolator = interp1d(original_indices, data_array, axis=0, kind='linear')
    return interpolator(target_indices)


def normalize_skeleton(frames_xyz, visibility, center_joints, scale_joints):
    """
    Make the skeleton translation- and scale-invariant so the model learns
    movement patterns instead of where the patient stood relative to the
    camera.

    frames_xyz: (T, V, 3) raw MediaPipe x/y/z
    visibility: (T, V) visibility scores in [0, 1]
    center_joints / scale_joints: pair of joint indices (within this
      domain's joint list) used as the translation center and the scale
      reference respectively. Pick a pair that stays roughly rigid across
      the domain's movements (e.g. hip width for leg exercises, shoulder
      width for arm exercises) -- never a pair whose distance IS the motion
      being classified, or normalization erases the signal.
    Returns (normalized_xyz, scale_used, degenerate: bool)
    """
    c0, c1 = center_joints
    s0, s1 = scale_joints

    center = (frames_xyz[:, c0, :] + frames_xyz[:, c1, :]) / 2.0
    centered = frames_xyz - center[:, None, :]

    ref_vis_ok = (visibility[:, s0] > MIN_VISIBILITY_THRESHOLD) & \
                 (visibility[:, s1] > MIN_VISIBILITY_THRESHOLD)
    ref_dist = np.linalg.norm(frames_xyz[:, s0, :2] - frames_xyz[:, s1, :2], axis=1)
    valid_dists = ref_dist[ref_vis_ok]

    degenerate = False
    if len(valid_dists) == 0 or np.median(valid_dists) < 1e-4:
        scale = 1.0
        degenerate = True
    else:
        scale = float(np.median(valid_dists))

    normalized = centered / scale
    return normalized, scale, degenerate


def build_tensor(raw_frames_data, domain):
    """
    raw_frames_data: list/array of shape (N, num_joints, 4) as produced by
    MediaPipe extraction (x, y, z, visibility per joint per frame), in the
    joint order given by domain.joint_indices.
    Returns tensor of shape (4, domain.target_frames, domain.num_joints, 1),
    plus QC info.
    """
    data = np.asarray(raw_frames_data, dtype=np.float32)
    n_frames = data.shape[0]

    visible_mask = data[:, :, 3] > MIN_VISIBILITY_THRESHOLD
    frame_has_signal = visible_mask.any(axis=1)
    visible_fraction = float(frame_has_signal.mean()) if n_frames else 0.0

    interpolated = interpolate_frames(data, domain.target_frames)  # (T, V, 4)
    xyz = interpolated[:, :, :3]
    vis = np.clip(interpolated[:, :, 3], 0.0, 1.0)

    norm_xyz, scale, degenerate = normalize_skeleton(
        xyz, vis, domain.center_joints, domain.scale_joints
    )

    full = np.concatenate([norm_xyz, vis[:, :, None]], axis=2)  # (T, V, 4)
    tensor = np.transpose(full, (2, 0, 1))  # (4, T, V)
    tensor = np.expand_dims(tensor, axis=-1).astype(np.float32)  # (4, T, V, 1)

    qc = {
        "visible_fraction": visible_fraction,
        "scale_used": scale,
        "degenerate_scale": degenerate,
        "source_frame_count": n_frames,
    }
    return tensor, qc


def tensor_shape_for(domain):
    return (4, domain.target_frames, domain.num_joints, 1)


def validate_tensor(tensor, domain):
    expected = tensor_shape_for(domain)
    if tensor.shape != expected:
        return False, f"Invalid shape (got {tensor.shape}, expected {expected})"
    if np.isnan(tensor).any() or np.isinf(tensor).any():
        return False, "NaN or Inf values present"
    return True, "Valid"
