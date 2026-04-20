from pathlib import Path

import pytest

from src.shared.pose_schema import get_pose_schema, load_dog_pose_schema


@pytest.mark.unit
def test_load_dog_pose_schema_reads_official_yaml_order(tmp_path: Path) -> None:
    schema_path = tmp_path / "dog-pose.yaml"
    schema_path.write_text(
        """
kpt_shape: [24, 3]
kpt_names:
  0:
    - front_left_paw
    - front_left_knee
    - front_left_elbow
    - rear_left_paw
    - rear_left_knee
    - rear_left_elbow
    - front_right_paw
    - front_right_knee
    - front_right_elbow
    - rear_right_paw
    - rear_right_knee
    - rear_right_elbow
    - tail_start
    - tail_end
    - left_ear_base
    - right_ear_base
    - nose
    - chin
    - left_ear_tip
    - right_ear_tip
    - left_eye
    - right_eye
    - withers
    - throat
""".strip(),
        encoding="utf-8",
    )

    schema = load_dog_pose_schema(schema_path)

    assert schema.keypoint_count == 24
    assert schema.keypoint_names[16] == "nose"
    assert schema.head_keypoint_indices == (14, 15, 16, 17, 18, 19, 20, 21)
    assert schema.source == str(schema_path)


@pytest.mark.unit
def test_load_dog_pose_schema_rejects_missing_head_keypoints(tmp_path: Path) -> None:
    schema_path = tmp_path / "bad-dog-pose.yaml"
    schema_path.write_text(
        """
kpt_shape: [24, 3]
kpt_names:
  0:
    - a0
    - a1
    - a2
    - a3
    - a4
    - a5
    - a6
    - a7
    - a8
    - a9
    - a10
    - a11
    - a12
    - a13
    - a14
    - a15
    - a16
    - a17
    - a18
    - a19
    - a20
    - a21
    - a22
    - a23
""".strip(),
        encoding="utf-8",
    )

    with pytest.raises(ValueError, match="missing required head keypoints"):
        load_dog_pose_schema(schema_path)


@pytest.mark.unit
def test_get_pose_schema_falls_back_to_builtin_dog_schema() -> None:
    schema = get_pose_schema("dog", Path("c:/does-not-exist/dog-pose.yaml"))

    assert schema.keypoint_count == 24
    assert schema.keypoint_names[20] == "left_eye"
    assert schema.source == "builtin-dog"
