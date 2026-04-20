from __future__ import annotations

from dataclasses import dataclass, field
from pathlib import Path
from typing import Literal

import yaml


TargetKind = Literal["human", "dog"]


HUMAN_KEYPOINT_NAMES = (
    "nose",
    "left_eye",
    "right_eye",
    "left_ear",
    "right_ear",
    "left_shoulder",
    "right_shoulder",
    "left_elbow",
    "right_elbow",
    "left_wrist",
    "right_wrist",
    "left_hip",
    "right_hip",
    "left_knee",
    "right_knee",
    "left_ankle",
    "right_ankle",
)

DOG_KEYPOINT_NAMES = (
    "front_left_paw",
    "front_left_knee",
    "front_left_elbow",
    "rear_left_paw",
    "rear_left_knee",
    "rear_left_elbow",
    "front_right_paw",
    "front_right_knee",
    "front_right_elbow",
    "rear_right_paw",
    "rear_right_knee",
    "rear_right_elbow",
    "tail_start",
    "tail_end",
    "left_ear_base",
    "right_ear_base",
    "nose",
    "chin",
    "left_ear_tip",
    "right_ear_tip",
    "left_eye",
    "right_eye",
    "withers",
    "throat",
)

HUMAN_HEAD_WEIGHTS = {
    "nose": 1.25,
    "left_eye": 1.0,
    "right_eye": 1.0,
    "left_ear": 0.8,
    "right_ear": 0.8,
}

DOG_HEAD_WEIGHTS = {
    "left_ear_base": 0.7,
    "right_ear_base": 0.7,
    "nose": 1.5,
    "chin": 0.9,
    "left_ear_tip": 0.3,
    "right_ear_tip": 0.3,
    "left_eye": 1.2,
    "right_eye": 1.2,
}

DOG_BODY_FALLBACK_WEIGHTS = {
    "withers": 1.0,
    "throat": 0.8,
}


@dataclass(frozen=True, slots=True)
class PoseSchema:
    target_kind: TargetKind
    keypoint_names: tuple[str, ...]
    head_keypoint_indices: tuple[int, ...]
    head_keypoint_weights: dict[int, float]
    head_blend_alpha: float
    min_keypoint_visibility: float
    bbox_fallback_y_ratio: float
    source: str
    body_fallback_indices: tuple[int, ...] = ()
    body_fallback_weights: dict[int, float] = field(default_factory=dict)
    nose_keypoint_index: int | None = None

    @property
    def keypoint_count(self) -> int:
        return len(self.keypoint_names)


def get_pose_schema(
    target_kind: TargetKind, dog_schema_path: str | Path | None = None
) -> PoseSchema:
    if target_kind == "human":
        return _build_schema(
            target_kind="human",
            keypoint_names=HUMAN_KEYPOINT_NAMES,
            head_weights=HUMAN_HEAD_WEIGHTS,
            head_blend_alpha=1.0,
            min_keypoint_visibility=0.3,
            bbox_fallback_y_ratio=0.15,
            nose_keypoint_name="nose",
            source="builtin-human",
        )
    return load_dog_pose_schema(dog_schema_path)


def load_dog_pose_schema(schema_path: str | Path | None = None) -> PoseSchema:
    path = Path(schema_path) if schema_path is not None else None
    if path is not None and path.exists():
        payload = yaml.safe_load(path.read_text(encoding="utf-8"))
        keypoint_names = _extract_dog_keypoint_names(payload)
        kpt_shape = payload.get("kpt_shape")
        if not isinstance(kpt_shape, list) or len(kpt_shape) < 1:
            raise ValueError(f"Invalid dog pose schema in {path}: missing kpt_shape")
        if int(kpt_shape[0]) != len(keypoint_names):
            raise ValueError(
                f"Invalid dog pose schema in {path}: kpt_shape count"
                f" {kpt_shape[0]} != {len(keypoint_names)} keypoint names"
            )
        return _build_schema(
            target_kind="dog",
            keypoint_names=tuple(keypoint_names),
            head_weights=DOG_HEAD_WEIGHTS,
            head_blend_alpha=0.55,
            min_keypoint_visibility=0.4,
            bbox_fallback_y_ratio=0.35,
            nose_keypoint_name="nose",
            body_weights=DOG_BODY_FALLBACK_WEIGHTS,
            source=str(path),
        )

    return _build_schema(
        target_kind="dog",
        keypoint_names=DOG_KEYPOINT_NAMES,
        head_weights=DOG_HEAD_WEIGHTS,
        head_blend_alpha=0.55,
        min_keypoint_visibility=0.4,
        bbox_fallback_y_ratio=0.35,
        nose_keypoint_name="nose",
        body_weights=DOG_BODY_FALLBACK_WEIGHTS,
        source="builtin-dog",
    )


def _build_schema(
    *,
    target_kind: TargetKind,
    keypoint_names: tuple[str, ...],
    head_weights: dict[str, float],
    head_blend_alpha: float,
    min_keypoint_visibility: float,
    bbox_fallback_y_ratio: float,
    source: str,
    nose_keypoint_name: str | None = None,
    body_weights: dict[str, float] | None = None,
) -> PoseSchema:
    name_to_index = {name: index for index, name in enumerate(keypoint_names)}
    missing_names = [name for name in head_weights if name not in name_to_index]
    if missing_names:
        raise ValueError(
            f"Pose schema source {source} missing required"
            f" head keypoints: {', '.join(missing_names)}"
        )

    head_indices = tuple(name_to_index[name] for name in head_weights)
    head_index_weights = {name_to_index[name]: weight for name, weight in head_weights.items()}

    nose_idx = name_to_index.get(nose_keypoint_name) if nose_keypoint_name else None

    body_indices: tuple[int, ...] = ()
    body_index_weights: dict[int, float] = {}
    if body_weights:
        valid_body = {n: w for n, w in body_weights.items() if n in name_to_index}
        body_indices = tuple(name_to_index[n] for n in valid_body)
        body_index_weights = {name_to_index[n]: w for n, w in valid_body.items()}

    return PoseSchema(
        target_kind=target_kind,
        keypoint_names=keypoint_names,
        head_keypoint_indices=head_indices,
        head_keypoint_weights=head_index_weights,
        head_blend_alpha=head_blend_alpha,
        min_keypoint_visibility=min_keypoint_visibility,
        bbox_fallback_y_ratio=bbox_fallback_y_ratio,
        source=source,
        body_fallback_indices=body_indices,
        body_fallback_weights=body_index_weights,
        nose_keypoint_index=nose_idx,
    )


def _extract_dog_keypoint_names(payload: object) -> list[str]:
    if not isinstance(payload, dict):
        raise ValueError("Dog pose schema must be a mapping")

    keypoint_sets = payload.get("kpt_names")
    if not isinstance(keypoint_sets, dict):
        raise ValueError("Dog pose schema missing kpt_names mapping")

    names = keypoint_sets.get(0, keypoint_sets.get("0"))
    if not isinstance(names, list) or not names:
        raise ValueError("Dog pose schema missing class 0 keypoint names")
    if not all(isinstance(name, str) and name for name in names):
        raise ValueError("Dog pose schema contains invalid keypoint names")
    return names
