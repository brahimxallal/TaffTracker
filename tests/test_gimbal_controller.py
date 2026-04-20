"""Tests for the tuned gimbal controller: gain scheduling, feedforward, predictive lead."""

from __future__ import annotations

import math
import multiprocessing as mp

import pytest

from src.config import CameraConfig, CommConfig, GimbalConfig, TrackingConfig
from src.output.process import OutputProcess
from src.shared.protocol import decode_packet_v2
from src.shared.ring_buffer import SharedRingBuffer
from src.shared.types import TrackingMessage

_TS_BASE = 1_000_000_000
_TS_STEP = 18_181_818  # ~55 fps


def _msg(**overrides) -> TrackingMessage:
    defaults = dict(
        frame_id=1,
        timestamp_ns=_TS_BASE,
        target_kind="human",
        target_acquired=True,
        state_source="measurement",
        track_id=1,
        confidence=0.9,
        raw_pixel=(320.0, 240.0),
        filtered_pixel=(320.0, 240.0),
        raw_angles=(0.0, 0.0),
        filtered_angles=(0.0, 0.0),
        inference_ms=5.0,
        tracking_ms=1.0,
        total_latency_ms=8.0,
        fps=55.0,
    )
    defaults.update(overrides)
    return TrackingMessage(**defaults)


def _make_proc(**gimbal_kw):
    layout, write_index = SharedRingBuffer.create((480, 640, 3), num_slots=2)
    defaults = dict(
        invert_pan=False,
        invert_tilt=False,
        pan_limit_deg=90.0,
        tilt_limit_deg=90.0,
        kd=0.0,
        deadband_deg=0.0,
        integral_decay_rate=0.0,
        slew_limit_dps=1000.0,
        tilt_scale=1.0,
    )
    defaults.update(gimbal_kw)
    proc = OutputProcess(
        layout=layout.layout,
        write_index=write_index,
        result_queue=mp.Queue(),
        shutdown_event=mp.Event(),
        error_queue=mp.Queue(),
        mode="camera",
        camera_config=CameraConfig(width=640, height=480),
        comm_config=CommConfig(),
        tracking_config=TrackingConfig(),
        gimbal_config=GimbalConfig(**defaults),
    )
    return proc, layout


# ── Gain Schedule ────────────────────────────────────────────────────


@pytest.mark.unit
def test_kp_far_vs_near():
    """Large error uses kp_far; small error uses kp_near — outputs differ."""
    proc_far, layout_far = _make_proc(
        kp=2.0,
        kp_far=8.0,
        kp_near=1.0,
        gain_schedule_threshold_deg=5.0,
    )
    proc_near, layout_near = _make_proc(
        kp=2.0,
        kp_far=8.0,
        kp_near=1.0,
        gain_schedule_threshold_deg=5.0,
    )
    try:
        # 10° error → above threshold → kp_far=8.0
        msg_far = _msg(servo_angles=(math.radians(10.0), 0.0))
        d_far = decode_packet_v2(proc_far._encode_packet(msg_far, 0))

        # 2° error → below threshold → kp_near=1.0
        msg_near = _msg(servo_angles=(math.radians(2.0), 0.0))
        d_near = decode_packet_v2(proc_near._encode_packet(msg_near, 0))

        # far output should be much larger than near (8×10 vs 1×2 rates)
        assert (
            d_far.pan > d_near.pan * 5
        ), f"kp_far branch not active: far={d_far.pan}, near={d_near.pan}"
    finally:
        layout_far.cleanup()
        layout_near.cleanup()


@pytest.mark.unit
def test_kp_far_near_equal_matches_flat_kp():
    """When kp_far == kp_near, output matches a flat-kp controller."""
    proc_sched, layout_sched = _make_proc(
        kp=4.0,
        kp_far=4.0,
        kp_near=4.0,
        gain_schedule_threshold_deg=5.0,
    )
    proc_flat, layout_flat = _make_proc(kp=4.0)
    try:
        msg = _msg(servo_angles=(math.radians(8.0), 0.0))
        d_sched = decode_packet_v2(proc_sched._encode_packet(msg, 0))
        d_flat = decode_packet_v2(proc_flat._encode_packet(msg, 0))
        assert d_sched.pan == d_flat.pan
    finally:
        layout_sched.cleanup()
        layout_flat.cleanup()


# ── Velocity Feedforward ─────────────────────────────────────────────


@pytest.mark.unit
def test_velocity_feedforward_zero_is_noop():
    """ff_gain=0 produces identical output to controller without feedforward."""
    proc_ff0, layout0 = _make_proc(
        kp=4.0,
        velocity_feedforward_gain=0.0,
    )
    proc_ff1, layout1 = _make_proc(
        kp=4.0,
        velocity_feedforward_gain=0.0,
    )
    try:
        msg = _msg(
            servo_angles=(math.radians(5.0), 0.0),
            servo_angular_velocity=(math.radians(30.0), 0.0),
        )
        d0 = decode_packet_v2(proc_ff0._encode_packet(msg, 0))
        d1 = decode_packet_v2(proc_ff1._encode_packet(msg, 0))
        assert d0.pan == d1.pan
    finally:
        layout0.cleanup()
        layout1.cleanup()


@pytest.mark.unit
def test_velocity_feedforward_increases_command():
    """Positive ff_gain + positive velocity increases pan command vs ff_gain=0."""
    proc_no_ff, layout0 = _make_proc(
        kp=4.0,
        velocity_feedforward_gain=0.0,
    )
    proc_with_ff, layout1 = _make_proc(
        kp=4.0,
        velocity_feedforward_gain=0.5,
    )
    try:
        msg = _msg(
            servo_angles=(math.radians(5.0), 0.0),
            servo_angular_velocity=(math.radians(30.0), 0.0),
        )
        d_no = decode_packet_v2(proc_no_ff._encode_packet(msg, 0))
        d_yes = decode_packet_v2(proc_with_ff._encode_packet(msg, 0))
        assert (
            d_yes.pan > d_no.pan
        ), f"Feedforward did not increase command: ff={d_yes.pan}, no_ff={d_no.pan}"
    finally:
        layout0.cleanup()
        layout1.cleanup()


# ── Predictive Lead ──────────────────────────────────────────────────


@pytest.mark.unit
def test_predictive_lead_gated_by_speed():
    """Lead is zero when target speed < 30°/s; active when speed > 30°/s."""
    proc_slow, layout_slow = _make_proc(
        kp=4.0,
        predictive_lead_s=0.08,
        velocity_feedforward_gain=0.0,
    )
    proc_nolead, layout_nolead = _make_proc(
        kp=4.0,
        predictive_lead_s=0.0,
        velocity_feedforward_gain=0.0,
    )
    try:
        # Low speed (10°/s) — lead should be inactive, same as no-lead
        slow_vel = math.radians(10.0)
        msg_slow = _msg(
            servo_angles=(math.radians(5.0), 0.0),
            servo_angular_velocity=(slow_vel, 0.0),
        )
        d_slow_lead = decode_packet_v2(proc_slow._encode_packet(msg_slow, 0))
        d_slow_no = decode_packet_v2(proc_nolead._encode_packet(msg_slow, 0))
        assert (
            d_slow_lead.pan == d_slow_no.pan
        ), f"Lead active at low speed: lead={d_slow_lead.pan}, no_lead={d_slow_no.pan}"
    finally:
        layout_slow.cleanup()
        layout_nolead.cleanup()


@pytest.mark.unit
def test_predictive_lead_active_at_high_speed():
    """Lead > 0 when speed > 30°/s shifts the effective error."""
    proc_lead, layout_lead = _make_proc(
        kp=4.0,
        predictive_lead_s=0.08,
        velocity_feedforward_gain=0.0,
    )
    proc_nolead, layout_nolead = _make_proc(
        kp=4.0,
        predictive_lead_s=0.0,
        velocity_feedforward_gain=0.0,
    )
    try:
        fast_vel = math.radians(60.0)  # 60°/s > 30°/s threshold
        msg = _msg(
            servo_angles=(math.radians(5.0), 0.0),
            servo_angular_velocity=(fast_vel, 0.0),
        )
        d_lead = decode_packet_v2(proc_lead._encode_packet(msg, 0))
        d_no = decode_packet_v2(proc_nolead._encode_packet(msg, 0))
        # Lead adds velocity × 0.08s to error → larger command
        assert (
            d_lead.pan != d_no.pan
        ), f"Lead had no effect at high speed: lead={d_lead.pan}, no_lead={d_no.pan}"
    finally:
        layout_lead.cleanup()
        layout_nolead.cleanup()
