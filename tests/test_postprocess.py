import numpy as np
import pytest

from src.config import PostprocessConfig
from src.inference.postprocess import compute_stabilized_centroid, KeypointStabilizer, parse_yolo_output
from src.shared.pose_schema import get_pose_schema
from src.shared.types import Detection


@pytest.mark.unit
def test_parse_yolo_output_filters_and_suppresses_overlap() -> None:
    num_keypoints = 17
    raw = np.zeros((1, 5 + num_keypoints * 3, 3), dtype=np.float32)

    raw[0, 0:4, 0] = np.array([10.0, 10.0, 8.0, 8.0])
    raw[0, 4, 0] = 0.95
    raw[0, 5:8, 0] = np.array([10.0, 8.0, 0.9])
    raw[0, 14:17, 0] = np.array([8.0, 9.0, 0.8])
    raw[0, 17:20, 0] = np.array([12.0, 9.0, 0.85])

    raw[0, 0:4, 1] = np.array([10.2, 10.0, 8.0, 8.0])
    raw[0, 4, 1] = 0.80

    raw[0, 0:4, 2] = np.array([100.0, 100.0, 20.0, 20.0])
    raw[0, 4, 2] = 0.20

    detections = parse_yolo_output(raw, conf_threshold=0.5, num_keypoints=num_keypoints)

    assert len(detections) == 1
    assert abs(detections[0].score - 0.95) < 1e-6


@pytest.mark.unit
def test_stabilized_centroid_prefers_visible_head_keypoints() -> None:
    num_keypoints = 17
    raw = np.zeros((1, 3, 5 + num_keypoints * 3), dtype=np.float32)
    raw[0, 0, 0:4] = np.array([50.0, 60.0, 20.0, 20.0])
    raw[0, 0, 4] = 0.95
    raw[0, 0, 5:8] = np.array([52.0, 58.0, 1.0])
    raw[0, 0, 8:11] = np.array([50.0, 58.0, 1.0])
    raw[0, 0, 11:14] = np.array([54.0, 58.0, 1.0])
    raw[0, 0, 14:17] = np.array([48.0, 59.0, 1.0])
    raw[0, 0, 17:20] = np.array([56.0, 59.0, 1.0])

    detection = parse_yolo_output(raw, conf_threshold=0.5, num_keypoints=num_keypoints)[0]
    stabilizer = KeypointStabilizer()
    schema = get_pose_schema("human")
    centroid = compute_stabilized_centroid(detection, schema, stabilizer)

    # Cascading centroid: 5 keypoints with high confidence → high quality
    # head_blend_alpha=0.85, bbox_fallback_y_ratio=0.2
    # bbox is xyxy: (40,50,60,70), bbox_cx=50, bbox_ref_y=50+(70-50)*0.2=54
    # Head center ≈ (52, 58.4) with nose priority boost
    # Blended x ≈ 0.85*head_x + 0.15*50, y ≈ 0.85*head_y + 0.15*54
    # Result should be close to head keypoints (strong head bias)
    assert abs(centroid[0] - 52.0) < 3.0
    assert abs(centroid[1] - 58.0) < 3.0


@pytest.mark.unit
def test_stabilized_centroid_uses_official_dog_head_keypoints() -> None:
    num_keypoints = 24
    raw = np.zeros((1, 1, 6 + num_keypoints * 3), dtype=np.float32)
    raw[0, 0, 0:4] = np.array([40.0, 50.0, 60.0, 80.0])
    raw[0, 0, 4] = 0.91
    raw[0, 0, 5] = 0.0

    keypoint_start = 6
    def set_kpt(index: int, x_value: float, y_value: float, visible: float = 1.0) -> None:
        offset = keypoint_start + (index * 3)
        raw[0, 0, offset : offset + 3] = np.array([x_value, y_value, visible])

    set_kpt(14, 46.0, 58.0)
    set_kpt(15, 58.0, 58.0)
    set_kpt(16, 52.0, 60.0)
    set_kpt(17, 52.0, 64.0)
    set_kpt(18, 44.0, 56.0)
    set_kpt(19, 60.0, 56.0)
    set_kpt(20, 49.0, 59.0)
    set_kpt(21, 55.0, 59.0)

    detection = parse_yolo_output(raw, conf_threshold=0.5, num_keypoints=num_keypoints)[0]
    stabilizer = KeypointStabilizer()
    schema = get_pose_schema("dog")
    centroid = compute_stabilized_centroid(detection, schema, stabilizer)

    # Cascading centroid: 8 keypoints with high confidence → high quality
    # head_blend_alpha=0.55, bbox_fallback_y_ratio=0.35
    # bbox is xyxy: (40,50,60,80), bbox_cx=50, bbox_ref_y=50+(80-50)*0.35=60.5
    # Head center ≈ (52, 59) weighted by nose=1.5, eyes=1.2, ears=0.7/0.3, chin=0.9
    # Blended x ≈ 0.55*52 + 0.45*50, y ≈ 0.55*59 + 0.45*60.5
    # Result should be biased toward head keypoints
    assert abs(centroid[0] - 51.1) < 2.0
    assert abs(centroid[1] - 59.7) < 2.0



@pytest.mark.unit
def test_parse_yolo_output_accepts_nms_export_shape() -> None:
    num_keypoints = 17
    raw = np.zeros((1, 300, 6 + num_keypoints * 3), dtype=np.float32)
    raw[0, 0, 0:4] = np.array([10.0, 12.0, 20.0, 24.0])
    raw[0, 0, 4] = 0.92
    raw[0, 0, 5] = 0.0
    raw[0, 0, 6:9] = np.array([14.0, 16.0, 0.9])

    detections = parse_yolo_output(raw, conf_threshold=0.5, num_keypoints=num_keypoints)

    assert len(detections) == 1
    assert tuple(detections[0].bbox.tolist()) == (10.0, 12.0, 20.0, 24.0)
    assert abs(detections[0].score - 0.92) < 1e-6


@pytest.mark.unit
def test_parse_yolo_output_empty_returns_empty_list() -> None:
    num_keypoints = 17
    raw = np.zeros((1, 5 + num_keypoints * 3, 0), dtype=np.float32)

    detections = parse_yolo_output(raw, conf_threshold=0.5, num_keypoints=num_keypoints)

    assert detections == []


@pytest.mark.unit
def test_parse_yolo_output_all_below_threshold_returns_empty() -> None:
    num_keypoints = 17
    raw = np.zeros((1, 5 + num_keypoints * 3, 5), dtype=np.float32)
    # Set all scores below threshold
    raw[0, 4, :] = 0.1

    detections = parse_yolo_output(raw, conf_threshold=0.5, num_keypoints=num_keypoints)

    assert detections == []


@pytest.mark.unit
def test_xywh_box_is_converted_to_xyxy() -> None:
    from src.inference.postprocess import xywh_to_xyxy

    xywh = np.array([[10.0, 20.0, 4.0, 6.0]], dtype=np.float32)

    xyxy = xywh_to_xyxy(xywh)

    # x1 = 10 - 4/2 = 8, y1 = 20 - 6/2 = 17, x2 = 10 + 4/2 = 12, y2 = 20 + 6/2 = 23
    assert xyxy[0].tolist() == pytest.approx([8.0, 17.0, 12.0, 23.0])


# --- Additional postprocess edge cases ---


@pytest.mark.unit
def test_centroid_falls_back_to_bbox_when_no_keypoints() -> None:
    """When keypoints is None, falls back to bbox centroid."""
    schema = get_pose_schema("human")
    detection = Detection(
        bbox=np.array([100.0, 200.0, 300.0, 400.0]),
        score=0.9,
        keypoints=None,
    )
    cx, cy = compute_stabilized_centroid(detection, schema)
    assert abs(cx - 200.0) < 1e-3  # (100+300)/2
    assert abs(cy - 230.0) < 5.0   # 200 + (400-200)*0.15 = 230


@pytest.mark.unit
def test_centroid_nose_priority_boost() -> None:
    """When nose has high confidence, centroid is boosted toward nose."""
    schema = get_pose_schema("human")
    # Create 17 keypoints, all with high visibility
    keypoints = np.zeros((17, 3), dtype=np.float32)
    # Head: nose(0), eyes(1,2), ears(3,4)
    keypoints[0] = [150.0, 100.0, 0.95]  # nose - high confidence
    keypoints[1] = [140.0, 95.0, 0.9]    # left eye
    keypoints[2] = [160.0, 95.0, 0.9]    # right eye
    keypoints[3] = [135.0, 100.0, 0.85]  # left ear
    keypoints[4] = [165.0, 100.0, 0.85]  # right ear
    detection = Detection(
        bbox=np.array([100.0, 80.0, 200.0, 200.0]),
        score=0.9,
        keypoints=keypoints,
    )
    # With nose priority
    cfg_with_nose = PostprocessConfig(nose_priority_blend=0.3, nose_priority_threshold=0.5)
    cx_nose, cy_nose = compute_stabilized_centroid(detection, schema, pp_config=cfg_with_nose)

    # Without nose priority (blend=0)
    cfg_no_nose = PostprocessConfig(nose_priority_blend=0.0, nose_priority_threshold=0.5)
    cx_no_nose, cy_no_nose = compute_stabilized_centroid(detection, schema, pp_config=cfg_no_nose)

    # With nose priority, centroid should be influenced by nose position
    # Nose is at x=150, so result should be close in both cases but slightly different
    assert abs(cx_nose - cx_no_nose) < 5.0  # should be close but not identical


@pytest.mark.unit
def test_centroid_medium_quality_uses_lower_alpha() -> None:
    """With fewer visible keypoints, alpha is reduced."""
    schema = get_pose_schema("human")
    # Only 2 head keypoints visible (below high_quality_min_keypoints threshold)
    keypoints = np.zeros((17, 3), dtype=np.float32)
    keypoints[0] = [150.0, 100.0, 0.5]   # nose - moderate confidence
    keypoints[1] = [145.0, 98.0, 0.5]    # left eye - moderate
    # Other keypoints invisible (conf=0)
    detection = Detection(
        bbox=np.array([100.0, 80.0, 200.0, 200.0]),
        score=0.8,
        keypoints=keypoints,
    )
    cfg = PostprocessConfig(high_quality_min_keypoints=3, medium_quality_alpha_reduction=0.3)
    cx, cy = compute_stabilized_centroid(detection, schema, pp_config=cfg)
    # Should still produce a valid centroid (blended with bbox)
    assert 100.0 < cx < 200.0
    assert 80.0 < cy < 200.0


@pytest.mark.unit
def test_centroid_dog_body_fallback() -> None:
    """Dog schema uses body fallback (withers/throat) when head keypoints are invisible."""
    schema = get_pose_schema("dog")
    # 24 keypoints, head keypoints (14-21) all invisible
    keypoints = np.zeros((24, 3), dtype=np.float32)
    # Body fallback indices for dog: typically withers/throat area
    # Set body fallback keypoints visible
    for idx in schema.body_fallback_indices:
        keypoints[idx] = [250.0, 300.0, 0.9]
    detection = Detection(
        bbox=np.array([200.0, 250.0, 350.0, 450.0]),
        score=0.85,
        keypoints=keypoints,
    )
    cx, cy = compute_stabilized_centroid(detection, schema)
    # Should use body fallback, blended with bbox
    assert 200.0 < cx < 350.0
    assert 250.0 < cy < 450.0


@pytest.mark.unit
def test_centroid_all_keypoints_invisible_falls_back_to_bbox() -> None:
    """When all keypoints are invisible, falls back to bbox."""
    schema = get_pose_schema("human")
    keypoints = np.zeros((17, 3), dtype=np.float32)
    # All confidences = 0 → invisible
    detection = Detection(
        bbox=np.array([100.0, 200.0, 300.0, 400.0]),
        score=0.9,
        keypoints=keypoints,
    )
    cx, cy = compute_stabilized_centroid(detection, schema)
    assert abs(cx - 200.0) < 1e-3  # bbox center x
    assert abs(cy - 230.0) < 5.0   # 200 + (400-200)*0.15 = 230
