from __future__ import annotations

import math
import multiprocessing as mp

import pytest

from src.config import CameraConfig, CommConfig, GimbalConfig, TrackingConfig
from src.output.process import OutputProcess
from src.shared.protocol import (
    FLAG_LASER_ON,
    FLAG_RELAY_ON,
    FLAG_TARGET_ACQUIRED,
    FLAG_HIGH_CONFIDENCE,
    HEADER_V2,
    PACKET_V2_SIZE,
    decode_packet_v2,
)
from src.shared.ring_buffer import SharedRingBuffer
from src.shared.types import TrackingMessage


@pytest.fixture
def output_process():
    """Create an OutputProcess instance without starting it as a process."""
    layout, write_index = SharedRingBuffer.create((480, 640, 3), num_slots=2)
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
        gimbal_config=GimbalConfig(
            invert_pan=False,
            invert_tilt=False,
            pan_limit_deg=90.0,
            tilt_limit_deg=90.0,
            kp=1.0,
            ki=0.0,
            kd=0.0,
        ),
    )
    yield proc
    layout.cleanup()


def _msg(**overrides) -> TrackingMessage:
    defaults = dict(
        frame_id=1,
        timestamp_ns=1_000_000_000,
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
    )
    defaults.update(overrides)
    return TrackingMessage(**defaults)


# ── Basic encode produces valid v2 packet ─────────────────────────


@pytest.mark.unit
def test_encode_produces_valid_v2_packet(output_process: OutputProcess) -> None:
    msg = _msg()
    packet = output_process._encode_packet(msg, sequence=42)
    assert len(packet) == PACKET_V2_SIZE
    assert packet[0] == HEADER_V2

    decoded = decode_packet_v2(packet)
    assert decoded is not None
    assert decoded.sequence == 42


# ── Zero angles produce zero pan/tilt ────────────────────────────


@pytest.mark.unit
def test_zero_angles_produce_zero_pan_tilt(output_process: OutputProcess) -> None:
    msg = _msg(filtered_angles=(0.0, 0.0), servo_angles=None)
    packet = output_process._encode_packet(msg, sequence=0)
    decoded = decode_packet_v2(packet)
    assert decoded.pan == 0
    assert decoded.tilt == 0


# ── servo_angles takes priority over filtered_angles ─────────────


@pytest.mark.unit
def test_servo_angles_priority(output_process: OutputProcess) -> None:
    """When servo_angles is set, the controller uses it over filtered_angles."""
    msg = _msg(
        filtered_angles=(0.1, 0.2),
        servo_angles=(0.05, 0.03),
    )
    packet = output_process._encode_packet(msg, sequence=0)
    decoded = decode_packet_v2(packet)
    # Incremental controller: first-frame output ∝ servo_angles error × kp × dt.
    # servo_angles pan (0.05 rad ≈ 2.86°) > tilt (0.03 rad ≈ 1.72°), both positive.
    assert decoded.pan > 0
    assert decoded.tilt > 0
    assert decoded.pan > decoded.tilt  # pan error larger than tilt error


# ── Invert pan flips sign ────────────────────────────────────────


@pytest.mark.unit
def test_invert_pan():
    layout, write_index = SharedRingBuffer.create((480, 640, 3), num_slots=2)
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
        gimbal_config=GimbalConfig(invert_pan=True, invert_tilt=False),
    )
    msg = _msg(servo_angles=(0.1, 0.0))
    decoded_inv = decode_packet_v2(proc._encode_packet(msg, 0))

    proc2 = OutputProcess(
        layout=layout.layout,
        write_index=write_index,
        result_queue=mp.Queue(),
        shutdown_event=mp.Event(),
        error_queue=mp.Queue(),
        mode="camera",
        camera_config=CameraConfig(width=640, height=480),
        comm_config=CommConfig(),
        tracking_config=TrackingConfig(),
        gimbal_config=GimbalConfig(invert_pan=False, invert_tilt=False),
    )
    decoded_normal = decode_packet_v2(proc2._encode_packet(msg, 0))

    assert decoded_inv.pan == -decoded_normal.pan
    layout.cleanup()


# ── Offset fields removed (firmware owns offsets) ─────────────


@pytest.mark.unit
def test_no_offset_fields_on_gimbal_config():
    """GimbalConfig no longer has offset fields; firmware owns offsets."""
    gc = GimbalConfig()
    assert not hasattr(gc, 'pan_offset_deg')
    assert not hasattr(gc, 'tilt_offset_deg')


# ── Pan/tilt clamped to limits ───────────────────────────────────


@pytest.mark.unit
def test_clamped_to_limits():
    layout, write_index = SharedRingBuffer.create((480, 640, 3), num_slots=2)
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
        gimbal_config=GimbalConfig(
            pan_limit_deg=10.0,
            tilt_limit_deg=10.0,
            kp=100000.0,
            kd=0.0,
            slew_limit_dps=1e6,
            deadband_deg=0.0,
        ),
    )
    # Large angle → incremental controller accumulates → clamped to ±10°
    # Distance-scaled slew limits per-frame increment, so feed enough frames.
    msg = _msg(servo_angles=(1.0, -1.0))  # ~57.3°
    for i in range(5):
        decoded = decode_packet_v2(proc._encode_packet(msg, i))
    assert decoded.pan == 1000   # 10° × 100
    assert decoded.tilt == -1000  # -10° × 100
    layout.cleanup()


# ── Laser flag: ON when target_acquired, OFF when lost ───────────


@pytest.mark.unit
def test_laser_flag_set_when_target_acquired(output_process: OutputProcess) -> None:
    """Laser stays armed by default whenever target_acquired=True."""
    msg = _msg(target_acquired=True, laser_pixel=None)
    decoded = decode_packet_v2(output_process._encode_packet(msg, 0))
    assert decoded.state & FLAG_LASER_ON


@pytest.mark.unit
def test_laser_flag_not_set_when_target_lost(output_process: OutputProcess) -> None:
    """Laser OFF when target_acquired=False."""
    msg = _msg(target_acquired=False, laser_pixel=None)
    decoded = decode_packet_v2(output_process._encode_packet(msg, 0))
    assert not (decoded.state & FLAG_LASER_ON)


@pytest.mark.unit
def test_laser_flag_not_set_when_user_disabled() -> None:
    layout, write_index = SharedRingBuffer.create((480, 640, 3), num_slots=2)
    laser_enabled = mp.Value("b", 0)
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
        laser_enabled=laser_enabled,
        gimbal_config=GimbalConfig(),
    )
    msg = _msg(target_acquired=True, laser_pixel=None)
    decoded = decode_packet_v2(proc._encode_packet(msg, 0))
    assert not (decoded.state & FLAG_LASER_ON)
    layout.cleanup()


# ── Relay flag reflects relay_flag value ─────────────────────────


@pytest.mark.unit
def test_relay_flag_on():
    layout, write_index = SharedRingBuffer.create((480, 640, 3), num_slots=2)
    relay = mp.Value("b", 1)
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
        relay_flag=relay,
    )
    msg = _msg()
    decoded = decode_packet_v2(proc._encode_packet(msg, 0))
    assert decoded.state & FLAG_RELAY_ON
    layout.cleanup()


@pytest.mark.unit
def test_relay_flag_off(output_process: OutputProcess) -> None:
    msg = _msg()
    decoded = decode_packet_v2(output_process._encode_packet(msg, 0))
    assert not (decoded.state & FLAG_RELAY_ON)


# ── Confidence mapped to 0-255 ───────────────────────────────────


@pytest.mark.unit
def test_confidence_mapping(output_process: OutputProcess) -> None:
    msg = _msg(confidence=1.0)
    decoded = decode_packet_v2(output_process._encode_packet(msg, 0))
    assert decoded.confidence == 255

    msg0 = _msg(confidence=0.0)
    decoded0 = decode_packet_v2(output_process._encode_packet(msg0, 0))
    assert decoded0.confidence == 0


# ── High confidence flag set when confidence > 0.7 ───────────────


@pytest.mark.unit
def test_high_confidence_flag(output_process: OutputProcess) -> None:
    msg_high = _msg(confidence=0.9)
    decoded = decode_packet_v2(output_process._encode_packet(msg_high, 0))
    assert decoded.state & FLAG_HIGH_CONFIDENCE

    msg_low = _msg(confidence=0.3)
    decoded_low = decode_packet_v2(output_process._encode_packet(msg_low, 0))
    assert not (decoded_low.state & FLAG_HIGH_CONFIDENCE)


# ── Target acquired flag ─────────────────────────────────────────


@pytest.mark.unit
def test_target_acquired_flag(output_process: OutputProcess) -> None:
    msg_acq = _msg(target_acquired=True)
    decoded = decode_packet_v2(output_process._encode_packet(msg_acq, 0))
    assert decoded.state & FLAG_TARGET_ACQUIRED

    msg_lost = _msg(target_acquired=False, state_source="lost")
    decoded_lost = decode_packet_v2(output_process._encode_packet(msg_lost, 0))
    assert not (decoded_lost.state & FLAG_TARGET_ACQUIRED)


# ── Latency clamped to 0-255 ────────────────────────────────────


@pytest.mark.unit
def test_latency_clamped(output_process: OutputProcess) -> None:
    msg = _msg(total_latency_ms=999.0)
    decoded = decode_packet_v2(output_process._encode_packet(msg, 0))
    assert decoded.latency == 255

    msg0 = _msg(total_latency_ms=0.0)
    decoded0 = decode_packet_v2(output_process._encode_packet(msg0, 0))
    assert decoded0.latency == 0


# ── Velocity encoded correctly ───────────────────────────────────


@pytest.mark.unit
def test_velocity_encoding(output_process: OutputProcess) -> None:
    msg = _msg(
        servo_angular_velocity=(math.radians(10.0), math.radians(-5.0)),
        angular_velocity=None,
    )
    decoded = decode_packet_v2(output_process._encode_packet(msg, 0))
    # 10 deg/s → 1000 centideg/s
    assert abs(decoded.pan_vel - 1000) <= 1
    assert abs(decoded.tilt_vel - (-500)) <= 1


# ── Camera-on-gimbal PI controller ───────────────────────────────

_TS_BASE = 1_000_000_000  # 1 second in ns
_TS_STEP = 18_181_818     # ~55 fps in ns


def _make_gimbal_proc(**kwargs):
    """Create OutputProcess for incremental controller tests.

    Default gains: kp=4.0 (approach), kd=0.0 (no derivative), deadband=0.0.
    Override via kwargs to test specific behavior.
    """
    layout, write_index = SharedRingBuffer.create((480, 640, 3), num_slots=2)
    defaults = dict(
        invert_pan=False, invert_tilt=False,
        pan_limit_deg=90.0, tilt_limit_deg=90.0,
        kp=4.0, kd=0.0, deadband_deg=0.0,
        integral_decay_rate=0.0, slew_limit_dps=1000.0,
        tilt_scale=1.0,
    )
    defaults.update(kwargs)
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


@pytest.mark.unit
def test_pi_first_frame_uses_kp_plus_ki():
    """First-frame command equals kp × error × dt (incremental accumulation)."""
    proc, layout = _make_gimbal_proc(kp=4.0, kd=0.0)
    try:
        err_deg = 10.0
        msg = _msg(
            servo_angles=(math.radians(err_deg), math.radians(0.0)),
            timestamp_ns=_TS_BASE, fps=55.0,
        )
        d = decode_packet_v2(proc._encode_packet(msg, 0))
        dt = 1.0 / 55.0
        expected = 4.0 * err_deg * dt  # approach_gain × error × dt
        assert abs(d.pan - round(expected * 100)) <= 2
    finally:
        layout.cleanup()


@pytest.mark.unit
def test_pi_integral_accumulates_over_frames():
    """Accumulated command grows with each frame of sustained error."""
    proc, layout = _make_gimbal_proc(kp=4.0, kd=0.0)
    try:
        err_deg = 5.0
        msg1 = _msg(
            servo_angles=(math.radians(err_deg), math.radians(0.0)),
            timestamp_ns=_TS_BASE, fps=55.0,
        )
        proc._encode_packet(msg1, 0)

        msg2 = _msg(
            servo_angles=(math.radians(err_deg), math.radians(0.0)),
            timestamp_ns=_TS_BASE + _TS_STEP, fps=55.0,
        )
        d2 = decode_packet_v2(proc._encode_packet(msg2, 1))
        dt = _TS_STEP / 1e9
        # Two frames: integral = kp * err * dt_frame1 + kp * err * dt_frame2
        expected = 4.0 * err_deg * (1.0 / 55.0) + 4.0 * err_deg * dt
        assert abs(d2.pan - round(expected * 100)) <= 3
    finally:
        layout.cleanup()


@pytest.mark.unit
def test_pi_zero_error_holds_integral():
    """When error drops to 0 (inside deadband=0), accumulated command holds."""
    proc, layout = _make_gimbal_proc(kp=4.0, kd=0.0, integral_decay_rate=0.0)
    try:
        msg1 = _msg(
            servo_angles=(math.radians(10.0), math.radians(0.0)),
            timestamp_ns=_TS_BASE, fps=55.0,
        )
        proc._encode_packet(msg1, 0)
        integral_now = proc._pi_integral_pan

        msg2 = _msg(
            servo_angles=(math.radians(0.0), math.radians(0.0)),
            timestamp_ns=_TS_BASE + _TS_STEP, fps=55.0,
        )
        d2 = decode_packet_v2(proc._encode_packet(msg2, 1))
        # Zero error → rate=0 → integral unchanged (no decay since rate=0)
        assert abs(d2.pan - round(integral_now * 100)) <= 2
    finally:
        layout.cleanup()


@pytest.mark.unit
def test_pi_resets_on_center():
    """Center state resets integral and sends zero."""
    proc, layout = _make_gimbal_proc()
    try:
        msg1 = _msg(
            servo_angles=(math.radians(10.0), math.radians(5.0)),
            timestamp_ns=_TS_BASE, fps=55.0,
        )
        proc._encode_packet(msg1, 0)

        msg_center = _msg(
            target_acquired=False, state_source="center",
            servo_angles=(0.0, 0.0), confidence=0.0,
            timestamp_ns=_TS_BASE + _TS_STEP, fps=55.0,
        )
        d = decode_packet_v2(proc._encode_packet(msg_center, 1))
        assert d.pan == 0
        assert d.tilt == 0

        msg3 = _msg(
            servo_angles=(math.radians(5.0), math.radians(0.0)),
            timestamp_ns=_TS_BASE + 2 * _TS_STEP, fps=55.0,
        )
        d3 = decode_packet_v2(proc._encode_packet(msg3, 2))
        assert d3.pan < 400
    finally:
        layout.cleanup()


@pytest.mark.unit
def test_pi_holds_integral_during_prediction():
    """Prediction holds accumulated command (does not add to it)."""
    proc, layout = _make_gimbal_proc(kp=4.0, kd=0.0)
    try:
        msg1 = _msg(
            servo_angles=(math.radians(8.0), math.radians(0.0)),
            timestamp_ns=_TS_BASE, fps=55.0,
        )
        proc._encode_packet(msg1, 0)
        integral_held = proc._pi_integral_pan

        msg_pred = _msg(
            target_acquired=False, state_source="prediction",
            servo_angles=(math.radians(2.0), math.radians(0.0)),
            confidence=0.0,
            timestamp_ns=_TS_BASE + _TS_STEP, fps=55.0,
        )
        d = decode_packet_v2(proc._encode_packet(msg_pred, 1))
        # Prediction: holds integral, doesn't accumulate further
        assert abs(d.pan - round(integral_held * 100)) <= 2

        # After re-acquisition, integral resumes from held value
        msg3 = _msg(
            servo_angles=(math.radians(1.0), math.radians(0.0)),
            timestamp_ns=_TS_BASE + 2 * _TS_STEP, fps=55.0,
        )
        d3 = decode_packet_v2(proc._encode_packet(msg3, 2))
        dt = _TS_STEP / 1e9
        expected3 = integral_held + 4.0 * 1.0 * dt
        assert abs(d3.pan - round(expected3 * 100)) <= 2
    finally:
        layout.cleanup()


@pytest.mark.unit
def test_pi_antiwindup_clamps_integral():
    """Accumulated command is clamped to pan/tilt limits (anti-windup)."""
    proc, layout = _make_gimbal_proc(
        pan_limit_deg=15.0, tilt_limit_deg=15.0,
        kp=100000.0, kd=0.0, slew_limit_dps=1e6,
    )
    try:
        msg = _msg(
            servo_angles=(math.radians(20.0), math.radians(-20.0)),
            timestamp_ns=_TS_BASE, fps=55.0,
        )
        d = decode_packet_v2(proc._encode_packet(msg, 0))
        assert d.pan == 1500
        assert d.tilt == -1500
    finally:
        layout.cleanup()


@pytest.mark.unit
def test_pi_with_inversion():
    """Sign inversion applies correctly in PI mode."""
    proc, layout = _make_gimbal_proc(invert_pan=True, invert_tilt=True)
    try:
        msg = _msg(
            servo_angles=(math.radians(5.0), math.radians(3.0)),
            timestamp_ns=_TS_BASE, fps=55.0,
        )
        d = decode_packet_v2(proc._encode_packet(msg, 0))
        assert d.pan < 0
        assert d.tilt < 0
    finally:
        layout.cleanup()


@pytest.mark.unit
def test_pi_convergence_simulation():
    """Simulate closed-loop: incremental controller + firmware EMA converges.

    With kp=4.0, kd=0, firmware alpha=0.5, the critically-damped loop
    should converge to <5% error within ~90 frames (~1.6s at 55fps).
    """
    proc, layout = _make_gimbal_proc(kp=4.0, kd=0.0)
    try:
        target_world_deg = 15.0
        servo_pos_deg = 0.0
        alpha = 0.5  # firmware EMA alpha (new tuning)

        for frame in range(90):
            ts = _TS_BASE + frame * _TS_STEP
            camera_err_deg = target_world_deg - servo_pos_deg
            msg = _msg(
                servo_angles=(math.radians(camera_err_deg), math.radians(0.0)),
                timestamp_ns=ts, fps=55.0,
            )
            d = decode_packet_v2(proc._encode_packet(msg, frame))
            command_deg = d.pan / 100.0
            servo_pos_deg = alpha * command_deg + (1 - alpha) * servo_pos_deg

        tracking_error = abs(target_world_deg - servo_pos_deg)
        assert tracking_error < target_world_deg * 0.05, (
            f"Incremental controller failed to converge: servo={servo_pos_deg:.2f}° "
            f"target={target_world_deg}° error={tracking_error:.2f}°"
        )
    finally:
        layout.cleanup()


@pytest.mark.unit
def test_pi_integral_survives_track_change():
    """Accumulated command is preserved when track_id changes (no harmful reset)."""
    proc, layout = _make_gimbal_proc(kp=4.0, kd=0.0)
    try:
        msg1 = _msg(
            servo_angles=(math.radians(10.0), math.radians(5.0)),
            timestamp_ns=_TS_BASE, fps=55.0, track_id=1,
        )
        proc._encode_packet(msg1, 0)
        integral_before = proc._pi_integral_pan
        assert integral_before != 0.0

        # Track changes to 5 — integral should NOT reset
        msg2 = _msg(
            servo_angles=(math.radians(3.0), math.radians(1.0)),
            timestamp_ns=_TS_BASE + _TS_STEP, fps=55.0, track_id=5,
        )
        proc._encode_packet(msg2, 1)
        # Integral grew (added kp*3*dt), not reset
        assert proc._pi_integral_pan > integral_before
    finally:
        layout.cleanup()
