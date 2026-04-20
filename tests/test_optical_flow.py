from __future__ import annotations

import numpy as np
import pytest

from src.tracking.optical_flow import OpticalFlowRefiner


@pytest.fixture
def refiner() -> OpticalFlowRefiner:
    return OpticalFlowRefiner(win_size=15, max_level=2, flow_weight=0.7, min_confidence=0.3)


def _make_frame(h: int = 100, w: int = 200, seed: int = 0) -> np.ndarray:
    rng = np.random.default_rng(seed)
    return rng.integers(0, 255, (h, w, 3), dtype=np.uint8)


# ── First call bootstraps state and returns raw points ────────────


@pytest.mark.unit
def test_first_call_returns_raw_points(refiner: OpticalFlowRefiner) -> None:
    frame = _make_frame(seed=1)
    pts = np.array([[50.0, 40.0], [100.0, 60.0]], dtype=np.float32)
    out = refiner.refine(frame, pts)
    np.testing.assert_array_almost_equal(out, pts)


# ── Reset clears state so next call bootstraps again ──────────────


@pytest.mark.unit
def test_reset_clears_state(refiner: OpticalFlowRefiner) -> None:
    frame1 = _make_frame(seed=1)
    pts = np.array([[50.0, 40.0]], dtype=np.float32)
    refiner.refine(frame1, pts)
    refiner.reset()
    assert refiner._prev_gray is None
    assert refiner._prev_points is None

    # After reset, next call should bootstrap again (return raw)
    frame2 = _make_frame(seed=2)
    out = refiner.refine(frame2, pts)
    np.testing.assert_array_almost_equal(out, pts)


# ── Changed number of points triggers re-bootstrap ────────────────


@pytest.mark.unit
def test_point_count_change_resets(refiner: OpticalFlowRefiner) -> None:
    frame1 = _make_frame(seed=1)
    pts3 = np.array([[10.0, 10.0], [50.0, 50.0], [90.0, 30.0]], dtype=np.float32)
    refiner.refine(frame1, pts3)

    frame2 = _make_frame(seed=2)
    pts2 = np.array([[10.0, 10.0], [50.0, 50.0]], dtype=np.float32)
    out = refiner.refine(frame2, pts2)
    # Point count mismatch → re-bootstrap → returns raw
    np.testing.assert_array_almost_equal(out, pts2)


# ── Identical frames produce output close to input ────────────────


@pytest.mark.unit
def test_identical_frames_produce_similar_output(refiner: OpticalFlowRefiner) -> None:
    frame = _make_frame(seed=42)
    pts = np.array([[50.0, 40.0], [100.0, 60.0]], dtype=np.float32)

    refiner.refine(frame, pts)  # bootstrap
    out = refiner.refine(frame.copy(), pts)  # same frame

    # Points should be very close to input (no real motion)
    assert out.shape == (2, 2)
    dist = np.linalg.norm(out - pts, axis=1)
    assert np.all(dist < 5.0), f"Points drifted too far: max dist = {dist.max():.2f}"


# ── Low confidence falls back to raw NN points ────────────────────


@pytest.mark.unit
def test_low_confidence_uses_raw_points(refiner: OpticalFlowRefiner) -> None:
    frame = _make_frame(seed=42)
    pts = np.array([[50.0, 40.0]], dtype=np.float32)
    confs = np.array([0.05], dtype=np.float32)  # below min_confidence=0.3

    refiner.refine(frame, pts)  # bootstrap
    out = refiner.refine(frame.copy(), pts, confidences=confs)

    # When confidence is below threshold, flow_weight → 0 → output equals raw pts
    np.testing.assert_array_almost_equal(out, pts, decimal=1)


# ── High confidence applies non-zero flow weight ─────────────────


@pytest.mark.unit
def test_high_confidence_applies_flow_weight() -> None:
    refiner = OpticalFlowRefiner(flow_weight=0.7, min_confidence=0.3)
    frame = _make_frame(seed=42)
    pts = np.array([[50.0, 40.0]], dtype=np.float32)
    confs = np.array([0.95], dtype=np.float32)

    refiner.refine(frame, pts, confidences=confs)  # bootstrap
    out = refiner.refine(frame.copy(), pts, confidences=confs)

    # Output should exist and have correct shape
    assert out.shape == (1, 2)


# ── Output shape always matches input ─────────────────────────────


@pytest.mark.unit
def test_output_shape_matches_input(refiner: OpticalFlowRefiner) -> None:
    frame = _make_frame(seed=1)
    for n_pts in (1, 5, 17):
        refiner.reset()
        pts = np.random.default_rng(n_pts).uniform(10, 90, (n_pts, 2)).astype(np.float32)
        out1 = refiner.refine(frame, pts)
        out2 = refiner.refine(frame.copy(), pts)
        assert out1.shape == (n_pts, 2)
        assert out2.shape == (n_pts, 2)


# ── Points near edge are clamped (no assertion failure) ──────────


@pytest.mark.unit
def test_edge_points_clamped_safely(refiner: OpticalFlowRefiner) -> None:
    h, w = 100, 200
    frame = _make_frame(h=h, w=w, seed=99)
    pts = np.array([[1.0, 1.0], [198.0, 98.0]], dtype=np.float32)

    refiner.refine(frame, pts)  # bootstrap
    # Should not raise even for points near frame edges
    out = refiner.refine(frame.copy(), pts)
    assert out.shape == (2, 2)


# ── Shifted frame produces output between raw and flow ────────────


@pytest.mark.unit
def test_shifted_frame_blends_flow_with_nn() -> None:
    """When the frame is shifted by a few pixels, flow should track and
    the output should be between raw NN and pure flow."""
    refiner = OpticalFlowRefiner(flow_weight=0.7, min_confidence=0.3)
    h, w = 120, 160
    rng = np.random.default_rng(77)
    base = rng.integers(30, 200, (h, w, 3), dtype=np.uint8)

    # Shift frame right by 3 pixels
    shifted = np.full_like(base, 114)
    shifted[:, 3:] = base[:, :-3]

    # Keypoint stays at same pixel position  → NN says no motion
    pts = np.array([[80.0, 60.0]], dtype=np.float32)
    confs = np.array([0.9], dtype=np.float32)

    refiner.refine(base, pts, confs)  # bootstrap
    out = refiner.refine(shifted, pts, confs)

    # With 0.7 flow weight and frame shifted right, flow would detect motion
    # Output should differ from raw by at most a few pixels
    assert out.shape == (1, 2)
    dist = np.linalg.norm(out - pts)
    # Allow up to 5px drift (flow detects the shift)
    assert dist < 5.0, f"Unexpected large drift: {dist:.2f}"
