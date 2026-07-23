from .base import DomainConfig

# Starting-point PT exercise set (confirmed by product owner as a first pass,
# not clinician-validated). Revisit against the confusion matrix once real
# data is collected -- wrist_curl and pronation_supination in particular may
# not be reliably separable (see note below).
CLASSES = [
    "shoulder_flexion", "shoulder_abduction", "elbow_flexion", "wrist_curl",
    "pronation_supination", "arm_raise", "external_rotation",
    "bicep_curl", "tricep_extension",
]

QUERIES = {
    "shoulder_flexion": [
        "shoulder flexion exercise", "shoulder flexion physiotherapy",
        "shoulder flexion rehab", "shoulder flexion range of motion",
        "shoulder pain flexion exercise", "frozen shoulder flexion stretch",
    ],
    "shoulder_abduction": [
        "shoulder abduction exercise", "shoulder abduction physiotherapy",
        "lateral raise shoulder abduction", "shoulder abduction rehab",
        "rotator cuff abduction exercise",
    ],
    "elbow_flexion": [
        "elbow flexion exercise", "elbow flexion physiotherapy",
        "elbow range of motion exercise", "elbow flexion rehab",
        "elbow contracture stretch",
    ],
    "wrist_curl": [
        "wrist curl exercise", "wrist curl physiotherapy",
        "wrist flexor strengthening", "wrist curl rehab",
        "wrist strengthening exercise",
    ],
    "pronation_supination": [
        "forearm pronation supination exercise", "pronation supination physiotherapy",
        "forearm rotation exercise rehab", "wrist pronation supination stretch",
    ],
    "arm_raise": [
        "arm raise exercise physiotherapy", "front arm raise rehab",
        "shoulder arm raise exercise", "straight arm raise physical therapy",
    ],
    "external_rotation": [
        "shoulder external rotation exercise", "external rotation physiotherapy",
        "rotator cuff external rotation", "external rotation rehab band",
    ],
    "bicep_curl": [
        "bicep curl exercise physiotherapy", "bicep curl rehab",
        "elbow flexor strengthening bicep curl", "bicep curl physical therapy",
    ],
    "tricep_extension": [
        "tricep extension exercise physiotherapy", "tricep extension rehab",
        "elbow extensor strengthening", "tricep extension physical therapy",
    ],
}

# MediaPipe Pose landmark indices.
JOINT_INDICES = [11, 12, 13, 14, 15, 16, 23, 24]
JOINT_NAMES = [
    "left_shoulder", "right_shoulder", "left_elbow", "right_elbow",
    "left_wrist", "right_wrist", "left_hip", "right_hip",
]

CONFIG = DomainConfig(
    name="upper_body",
    classes=CLASSES,
    queries=QUERIES,
    joint_indices=JOINT_INDICES,
    joint_names=JOINT_NAMES,
    center_joints=(0, 1),   # left_shoulder, right_shoulder
    scale_joints=(0, 1),    # shoulder width
    graph_edges=[
        (0, 1),             # shoulder-shoulder
        (0, 2), (1, 3),     # shoulder-elbow
        (2, 4), (3, 5),     # elbow-wrist
        (0, 6), (1, 7),     # shoulder-hip (torso anchor, gives the graph posture context)
        (6, 7),             # hip-hip
    ],
    yolo_pose_keypoint_range=(5, 11),  # COCO: shoulders(5,6), elbows(7,8), wrists(9,10)
    notes=(
        "MediaPipe Pose gives a single positional point per wrist -- it does not "
        "capture forearm rotation. 'wrist_curl' and 'pronation_supination' are "
        "kinematically similar from that signal alone and may need MediaPipe "
        "Hands landmarks fused in, or may need to be merged/dropped, if the "
        "confusion matrix shows them being confused with each other."
    ),
)
