import os
import glob
import json
import cv2
import numpy as np
import mediapipe as mp
from scipy.interpolate import interp1d

CLASSES = [
    "ankle", "calf", "hamstring", "heel_slide",
    "hip", "knee", "leg_raise", "quadriceps", "toes"
]

LOWER_LIMB_INDICES = [23, 24, 25, 26, 27, 28, 29, 30, 31, 32]
TARGET_FRAMES = 300

BASE_DIR = "datasets/lower_limb"
RAW_DIR = os.path.join(BASE_DIR, "raw")
SKELETON_DIR = os.path.join(BASE_DIR, "skeletons")
CORRUPTED_DIR = os.path.join(SKELETON_DIR, "corrupted")

def setup_dirs():
    os.makedirs(SKELETON_DIR, exist_ok=True)
    os.makedirs(CORRUPTED_DIR, exist_ok=True)

def extract_landmarks(video_path):
    mp_pose = mp.solutions.pose
    pose = mp_pose.Pose(
        static_image_mode=False,
        model_complexity=2,
        enable_segmentation=False,
        min_detection_confidence=0.5
    )
    
    cap = cv2.VideoCapture(video_path)
    frames_data = []
    
    while cap.isOpened():
        ret, frame = cap.read()
        if not ret:
            break
            
        frame_rgb = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)
        results = pose.process(frame_rgb)
        
        if results.pose_landmarks:
            landmarks = results.pose_landmarks.landmark
            frame_joints = []
            for idx in LOWER_LIMB_INDICES:
                lm = landmarks[idx]
                frame_joints.append([lm.x, lm.y, lm.z, lm.visibility])
            frames_data.append(frame_joints)
        else:
            # If no person detected, append NaNs to interpolate later
            frames_data.append([[np.nan, np.nan, np.nan, 0.0]] * 10)
            
    cap.release()
    pose.close()
    
    if not frames_data:
        return None
        
    return np.array(frames_data) # Shape: (N, 10, 4)

def interpolate_frames(data_array, target_length):
    N, V, C = data_array.shape
    
    # Handle NaNs using linear interpolation over time
    for v in range(V):
        for c in range(C):
            series = data_array[:, v, c]
            nans = np.isnan(series)
            if np.all(nans):
                data_array[:, v, c] = 0.0 # Fallback if entirely NaN
            elif np.any(nans):
                def get_x(z): return z.nonzero()[0]
                series[nans] = np.interp(get_x(nans), get_x(~nans), series[~nans])
                data_array[:, v, c] = series
                
    # Interpolate sequence length to target_length
    original_indices = np.linspace(0, N - 1, num=N)
    target_indices = np.linspace(0, N - 1, num=target_length)
    
    interpolator = interp1d(original_indices, data_array, axis=0, kind='linear')
    resampled_data = interpolator(target_indices)
    
    return resampled_data # Shape: (300, 10, 4)

def validate_and_repair(tensor):
    # tensor shape should be (4, 300, 10, 1)
    if tensor.shape != (4, TARGET_FRAMES, 10, 1):
        return False, "Invalid shape"
        
    if np.isnan(tensor).any() or np.isinf(tensor).any():
        return False, "NaN or Inf values present"
        
    # Check visibility bounds
    vis = tensor[3, :, :, 0]
    if np.any(vis < 0.0) or np.any(vis > 1.0):
        # Auto-repair visibility
        tensor[3, :, :, 0] = np.clip(vis, 0.0, 1.0)
        
    return True, "Valid"

def process_dataset():
    setup_dirs()
    
    stats = {'total': 0, 'successful': 0, 'corrupted': 0}
    validation_report = []
    
    for c in CLASSES:
        videos = glob.glob(os.path.join(RAW_DIR, c, "*.mp4"))
        for vid in videos:
            stats['total'] += 1
            filename = os.path.basename(vid)
            base_name = os.path.splitext(filename)[0]
            
            out_file = os.path.join(SKELETON_DIR, f"{base_name}.npy")
            if os.path.exists(out_file) or os.path.exists(os.path.join(CORRUPTED_DIR, f"{base_name}.npy")):
                continue
                
            print(f"Extracting: {filename}")
            raw_data = extract_landmarks(vid)
            
            if raw_data is None:
                stats['corrupted'] += 1
                validation_report.append({'file': filename, 'status': 'corrupted', 'reason': 'No landmarks extracted'})
                continue
                
            norm_data = interpolate_frames(raw_data, TARGET_FRAMES)
            
            # Rearrange from (T, V, C) to (C, T, V, M)
            # T=300, V=10, C=4
            tensor = np.transpose(norm_data, (2, 0, 1)) # -> (4, 300, 10)
            tensor = np.expand_dims(tensor, axis=-1) # -> (4, 300, 10, 1)
            
            is_valid, msg = validate_and_repair(tensor)
            
            save_path = out_file if is_valid else os.path.join(CORRUPTED_DIR, f"{base_name}.npy")
            np.save(save_path, tensor)
            
            if is_valid:
                stats['successful'] += 1
            else:
                stats['corrupted'] += 1
                validation_report.append({'file': filename, 'status': 'corrupted', 'reason': msg})
                
    with open(os.path.join(BASE_DIR, 'tensor_statistics.json'), 'w') as f:
        json.dump(stats, f, indent=4)
        
    with open(os.path.join(BASE_DIR, 'validation_report.json'), 'w') as f:
        json.dump(validation_report, f, indent=4)
        
    print("\n--- Phase 3 & 4 Completed ---")
    print(f"Generated {stats['successful']} valid skeleton tensors.")

if __name__ == "__main__":
    process_dataset()
