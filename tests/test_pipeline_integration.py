from __future__ import annotations

import numpy as np
import pytest

from src.config import adapt_tracking_for_fps, TrackingConfig
from src.inference.postprocess import (
    compute_stabilized_centroid,
    KeypointStabilizer,
    parse_yolo_output,
)
from src.shared.pose_schema import get_pose_schema
from src.shared.types import Detection
from src.tracking.bytetrack import ByteTracker
from src.tracking.kalman import KalmanFilter


def _make_raw_yolo(num_detections: int, num_keypoints: int, score: float = 0.9) -> np.ndarray:
    """Build a minimal YOLO output tensor with `num_detections` above-threshold boxes."""
    raw = np.zeros((1, 5 + num_keypoints * 3, num_detections), dtype=np.float32)
    for i in range(num_detections):
        raw[0, 0:4, i] = np.array([float(50 + i * 20), 50.0, 20.0, 30.0])
        raw[0, 4, i] = score
    return raw


@pytest.mark.integration
def test_ring_buffer_write_then_read_round_trip(ring_buffer_pair, sample_frame) -> None:
    buffer, _ = ring_buffer_pair
    ts = 1_234_567_890

    frame_id = buffer.write(sample_frame, timestamp_ns=ts)
    record = buffer.read_frame(frame_id, copy=True)

    assert record is not None
    assert record.frame_id == frame_id
    assert record.timestamp_ns == ts
    np.testing.assert_array_equal(record.frame, sample_frame)


@pytest.mark.integration
def test_postprocess_to_tracker_pipeline(sample_detections) -> None:
    tracker = ByteTracker(track_thresh=0.5, match_thresh=0.3, max_lost=5, birth_min_hits=1)

    tracks_1 = tracker.update(sample_detections, timestamp_ns=1_000_000)
    # Shift detections slightly for second frame
    shifted = [
        Detection(bbox=d.bbox + np.array([1.0, 0.0, 1.0, 0.0]), score=d.score)
        for d in sample_detections
    ]
    tracks_2 = tracker.update(shifted, timestamp_ns=2_000_000)

    assert len(tracks_1) == len(sample_detections)
    assert len(tracks_2) == len(sample_detections)
    # Track IDs should be preserved for well-overlapping boxes
    ids_1 = {t.track_id for t in tracks_1}
    ids_2 = {t.track_id for t in tracks_2}
    assert ids_1 == ids_2


@pytest.mark.integration
def test_tracker_to_kalman_pipeline() -> None:
    tracker = ByteTracker(track_thresh=0.5, match_thresh=0.3, max_lost=5, birth_min_hits=1)
    kalman = KalmanFilter(process_noise=1.0, measurement_noise=2.0)
    schema = get_pose_schema("human")

    stabilizer = KeypointStabilizer()
    positions = [(100.0, 200.0), (110.0, 200.0), (120.0, 200.0)]
    for i, (cx, cy) in enumerate(positions):
        x1, y1, x2, y2 = cx - 10, cy - 20, cx + 10, cy + 20
        det = Detection(bbox=np.array([x1, y1, x2, y2]), score=0.9)
        tracks = tracker.update([det], timestamp_ns=i * 16_000_000)
        if tracks:
            centroid = compute_stabilized_centroid(
                Detection(bbox=tracks[0].bbox, score=tracks[0].score), schema, stabilizer
            )
            state = kalman.update(centroid, dt=0.016)

    assert state is not None
    # The filter has only 3 measurements — just verify it's tracking in the right direction
    assert state.x > 100.0
    assert state.vx > 0.0


@pytest.mark.integration
def test_parse_yolo_full_pipeline_with_nms() -> None:
    num_keypoints = 17
    # Two overlapping boxes — NMS should keep only the higher-score one
    raw = np.zeros((1, 5 + num_keypoints * 3, 2), dtype=np.float32)
    raw[0, 0:4, 0] = np.array([100.0, 100.0, 40.0, 60.0])
    raw[0, 4, 0] = 0.95
    raw[0, 0:4, 1] = np.array([101.0, 101.0, 40.0, 60.0])  # Heavy overlap
    raw[0, 4, 1] = 0.80

    detections = parse_yolo_output(raw, conf_threshold=0.5, num_keypoints=num_keypoints)

    assert len(detections) == 1
    assert detections[0].score == pytest.approx(0.95, abs=1e-5)


# --- Phase 4+5 integration tests ---


@pytest.mark.integration
def test_stabilized_centroid_into_adaptive_kalman() -> None:
    """Phase 4 (stabilized centroid) feeds Phase 5 (adaptive Kalman)."""
    schema = get_pose_schema("human")
    stabilizer = KeypointStabilizer(alpha=0.3)
    kalman = KalmanFilter(process_noise=3.0, measurement_noise=8.0)

    # Simulate 10 frames of a person moving right
    for frame in range(10):
        cx, cy = 100.0 + frame * 5.0, 200.0
        # Build a minimal detection with head keypoints
        kps = np.zeros((17, 3), dtype=np.float32)
        kps[0] = [cx, cy, 0.9]       # nose
        kps[1] = [cx - 2, cy, 0.8]   # l_eye
        kps[2] = [cx + 2, cy, 0.85]  # r_eye
        kps[3] = [cx - 4, cy + 1, 0.7]  # l_ear
        kps[4] = [cx + 4, cy + 1, 0.7]  # r_ear
        det = Detection(bbox=np.array([cx - 20, cy - 30, cx + 20, cy + 30], dtype=np.float32), score=0.9, keypoints=kps)

        centroid = compute_stabilized_centroid(det, schema, stabilizer)
        state = kalman.update(centroid, dt=1.0 / 60.0)

    assert state is not None
    assert state.x > 100.0  # moved right
    assert state.vx > 0.0  # positive velocity
    assert not kalman.last_innovation_gated  # no outliers


@pytest.mark.integration
def test_occlusion_recovery_with_oru() -> None:
    """Phase 5: After occlusion, ORU re-update smooths recovery."""
    kalman = KalmanFilter(process_noise=3.0, measurement_noise=8.0)
    # Track for 10 frames
    for i in range(10):
        kalman.update((100.0 + i * 5.0, 200.0), dt=1.0 / 60.0)

    # Lose target for 5 frames
    for _ in range(5):
        result = kalman.predict(1.0 / 60.0)

    assert result is not None
    assert kalman.consecutive_predictions == 5

    # Re-acquire: ORU re-update + new measurement
    kalman.oru_re_update()
    state = kalman.update((150.0, 200.0), dt=1.0 / 60.0)

    assert state is not None
    assert kalman.consecutive_predictions == 0  # reset after valid update


@pytest.mark.integration
def test_prediction_cap_integration() -> None:
    """Phase 5: Prediction cap resets filter, next measurement re-initializes."""
    from src.tracking.kalman import MAX_CONSECUTIVE_PREDICTIONS

    kalman = KalmanFilter(process_noise=3.0, measurement_noise=8.0)
    kalman.update((100.0, 200.0), dt=0.016)

    # Exhaust predictions
    for _ in range(MAX_CONSECUTIVE_PREDICTIONS + 1):
        kalman.predict(0.016)

    assert not kalman.initialized

    # Re-initialize with new measurement
    state = kalman.update((300.0, 400.0), dt=0.016)
    assert state.x == 300.0
    assert state.y == 400.0


@pytest.mark.integration
def test_adaptive_config_with_fps_adaptation_pipeline() -> None:
    """Unified adaptive config + FPS adaptation work together in tracking."""
    cfg = TrackingConfig()
    adapted = adapt_tracking_for_fps(cfg, 30.0)

    tracker = ByteTracker(
        track_thresh=adapted.tracker_track_threshold,
        match_thresh=adapted.tracker_match_threshold,
        max_lost=adapted.max_lost_frames,
        birth_min_hits=1,
    )
    kalman = KalmanFilter(
        process_noise=adapted.process_noise,
        measurement_noise=adapted.measurement_noise,
    )

    # Simulate tracking
    det = Detection(bbox=np.array([100, 150, 140, 210], dtype=np.float32), score=0.8)
    tracks = tracker.update([det], timestamp_ns=0)
    assert len(tracks) == 1

    state = kalman.update((120.0, 180.0), dt=1.0 / 30.0)
    assert state is not None
    assert adapted.process_noise > cfg.process_noise  # lower FPS → higher Q


@pytest.mark.integration
def test_full_pipeline_detection_to_protocol() -> None:
    """End-to-end: YOLO output → stabilized centroid → adaptive Kalman → protocol v2."""
    from src.calibration.camera_model import CameraModel
    from src.shared.protocol import build_state_flags, encode_packet_v2, decode_packet_v2
    from math import degrees

    num_keypoints = 17
    schema = get_pose_schema("human")
    stabilizer = KeypointStabilizer()
    kalman = KalmanFilter(process_noise=3.0, measurement_noise=8.0)
    camera = CameraModel.identity(640, 640)

    # Build YOLO-like output
    raw = np.zeros((1, 5 + num_keypoints * 3, 1), dtype=np.float32)
    raw[0, 0:4, 0] = [320.0, 320.0, 50.0, 50.0]  # center of frame
    raw[0, 4, 0] = 0.95  # confidence
    raw[0, 5:8, 0] = [320.0, 315.0, 0.9]  # nose
    raw[0, 8:11, 0] = [318.0, 315.0, 0.85]  # l_eye
    raw[0, 11:14, 0] = [322.0, 315.0, 0.85]  # r_eye

    dets = parse_yolo_output(raw, conf_threshold=0.5, num_keypoints=num_keypoints)
    assert len(dets) == 1

    centroid = compute_stabilized_centroid(dets[0], schema, stabilizer)
    state = kalman.update(centroid, dt=1.0 / 60.0)

    angles = camera.pixel_to_angle(*state.position)
    ang_vel = camera.pixel_velocity_to_angular(state.vx, state.vy)

    pan_cd = int(round(degrees(angles[0]) * 100.0))
    tilt_cd = int(round(degrees(angles[1]) * 100.0))
    pan_vel_cd = int(round(degrees(ang_vel[0]) * 100.0))
    tilt_vel_cd = int(round(degrees(ang_vel[1]) * 100.0))

    flags = build_state_flags(
        state_source="measurement", target_acquired=True,
        confidence=0.95, velocity_magnitude_dps=0.0, is_occlusion_recovery=False,
    )

    packet = encode_packet_v2(
        sequence=1, timestamp_ms=16, pan=pan_cd, tilt=tilt_cd,
        pan_vel=pan_vel_cd, tilt_vel=tilt_vel_cd,
        confidence=int(0.95 * 255), state=flags, quality=int(0.95 * 255), latency=10,
    )

    decoded = decode_packet_v2(packet)
    assert decoded is not None
    assert decoded.sequence == 1
    # Near center → angles near 0
    assert abs(decoded.pan) < 500  # < 5 degrees in centidegrees
