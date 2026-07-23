"""
A domain is one body region / rehab focus this pipeline can be trained for
(lower_limb, upper_body, ...). Everything that used to be hardcoded for
lower-limb exercises lives in a DomainConfig now, so the same collection /
review / extraction / training / inference code works for any of them.
"""
from dataclasses import dataclass, field
from typing import Dict, List, Tuple


@dataclass(frozen=True)
class DomainConfig:
    name: str
    classes: List[str]
    queries: Dict[str, List[str]]           # class -> YouTube search queries (collection stage hypotheses)
    joint_indices: List[int]                # indices into the landmarker's full point set
    joint_names: List[str]
    center_joints: Tuple[int, int]          # indices (within joint_indices) averaged for translation-centering
    scale_joints: Tuple[int, int]           # indices whose distance is used as the scale reference
    graph_edges: List[Tuple[int, int]]      # skeleton connectivity for the GCN adjacency matrix
    yolo_pose_keypoint_range: Tuple[int, int]  # COCO keypoint slice quality_filter.py should check visibility of
    landmarker: str = "pose"                # which MediaPipe landmarker this domain reads from ("pose" for now)
    target_frames: int = 300
    notes: str = ""

    @property
    def num_joints(self) -> int:
        return len(self.joint_indices)

    def __post_init__(self):
        assert len(self.joint_indices) == len(self.joint_names)
        assert set(self.classes) == set(self.queries.keys()), \
            f"{self.name}: classes and queries keys must match exactly"
