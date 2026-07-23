from .base import DomainConfig

CLASSES = [
    "ankle", "calf", "hamstring", "heel_slide",
    "hip", "knee", "leg_raise", "quadriceps", "toes"
]

QUERIES = {
    "ankle": [
        "ankle rehabilitation exercise", "ankle physiotherapy exercise",
        "ankle mobility exercise", "ankle ROM exercise",
        "ankle strengthening exercise", "sprained ankle exercises",
        "ankle physical therapy", "ankle stability workout",
        "how to strengthen weak ankles",
    ],
    "calf": [
        "calf rehabilitation exercise", "calf physiotherapy exercise",
        "calf mobility exercise", "calf strengthening exercise",
        "calf muscle rehab", "calf raises variations",
        "soleus stretch and strengthen", "gastrocnemius exercises",
        "lower leg workout for runners",
    ],
    "hamstring": [
        "hamstring rehabilitation exercise", "hamstring physiotherapy exercise",
        "hamstring mobility exercise", "hamstring physical therapy",
        "pulled hamstring exercises", "hamstring curls at home",
        "how to strengthen hamstrings", "nordic hamstring curl tutorial",
    ],
    "heel_slide": [
        "heel slide rehabilitation exercise", "heel slide physiotherapy exercise",
        "heel slide mobility exercise", "knee replacement heel slides",
        "heel slide physical therapy", "acl rehab heel slides",
        "supine heel slides exercise",
    ],
    "hip": [
        "hip rehabilitation exercise", "hip physiotherapy exercise",
        "hip mobility exercise", "hip strengthening exercise",
        "hip flexor exercises", "glute and hip workout",
        "hip physical therapy exercises", "hip replacement rehab",
        "hip abductor strengthening",
    ],
    "knee": [
        "knee rehabilitation exercise", "knee physiotherapy exercise",
        "knee mobility exercise", "knee strengthening exercise",
        "knee pain relief workout", "how to strengthen knees",
        "knee stability exercises", "knee physical therapy at home",
        "exercises for bad knees", "vmo strengthening exercises",
        "acl recovery exercises", "patellar tracking exercises",
    ],
    "leg_raise": [
        "leg raise rehabilitation exercise", "leg raise physiotherapy exercise",
        "straight leg raise exercise", "supine straight leg raise",
        "side lying leg raise", "slr physical therapy",
        "hip flexor straight leg raise",
    ],
    "quadriceps": [
        "quadriceps rehabilitation exercise", "quadriceps physiotherapy exercise",
        "quadriceps strengthening exercise", "quad exercises at home",
        "how to build quad muscles", "quadriceps physical therapy",
        "isometric quad exercises", "terminal knee extension tke",
    ],
    "toes": [
        "toes rehabilitation exercise", "toes physiotherapy exercise",
        "toe strengthening exercise", "toe yoga exercises",
        "plantar fasciitis toe stretches", "foot and toe mobility",
        "intrinsic foot muscle exercises", "toe curl exercises",
    ],
}

# MediaPipe Pose landmark indices, in the order below.
JOINT_INDICES = [23, 24, 25, 26, 27, 28, 29, 30, 31, 32]
JOINT_NAMES = [
    "left_hip", "right_hip", "left_knee", "right_knee",
    "left_ankle", "right_ankle", "left_heel", "right_heel",
    "left_foot_index", "right_foot_index",
]

CONFIG = DomainConfig(
    name="lower_limb",
    classes=CLASSES,
    queries=QUERIES,
    joint_indices=JOINT_INDICES,
    joint_names=JOINT_NAMES,
    center_joints=(0, 1),   # left_hip, right_hip
    scale_joints=(0, 1),    # hip width -- stable across leg movement, unlike e.g. hip-to-ankle distance
    graph_edges=[
        (0, 1), (0, 2), (1, 3), (2, 4), (3, 5),
        (4, 6), (5, 7), (4, 8), (5, 9), (6, 8), (7, 9),
    ],
    yolo_pose_keypoint_range=(11, 17),  # COCO: hips(11,12), knees(13,14), ankles(15,16)
)
