from __future__ import annotations

import logging
import multiprocessing as mp
from math import degrees
from unittest.mock import MagicMock

import pytest

from src.config import CameraConfig, CommConfig, GimbalConfig, RuntimeFlags, TrackingConfig
from src.output.auto_comm import AutoCommTransport
from src.output.process import OutputProcess, _get_transport_status
from src.output.serial_comm import SerialComm
from src.output.udp_comm import UDPComm
from src.shared.protocol import FLAG_FAST_MOTION, FLAG_LASER_ON, FLAG_TARGET_ACQUIRED, decode_packet_v2
from src.shared.ring_buffer import SharedRingBuffer
from src.shared.types import TrackingMessage


def _make_message(**overrides) -> TrackingMessage:
    defaults: dict = dict(
        frame_id=1,
        timestamp_ns=1_000_000_000,
        target_kind="human",
        target_acquired=True,
        state_source="measurement",
        track_id=1,
        confidence=0.9,
        raw_pixel=(100.0, 200.0),
        filtered_pixel=(101.0, 201.0),
        raw_angles=(0.1, 0.2),
        filtered_angles=(0.11, 0.21),
        inference_ms=10.0,
        tracking_ms=1.0,
        total_latency_ms=15.0,
    )
    defaults.update(overrides)
    return TrackingMessage(**defaults)


def _make_proc(width: int = 640, height: int = 480, hold_time_s: float = 0.5) -> OutputProcess:
    return OutputProcess(
        layout=None,  # type: ignore[arg-type]
        write_index=None,  # type: ignore[arg-type]
        result_queue=None,  # type: ignore[arg-type]
        shutdown_event=None,  # type: ignore[arg-type]
        error_queue=None,  # type: ignore[arg-type]
        mode="camera",
        camera_config=CameraConfig(width=width, height=height, fps=30),
        comm_config=CommConfig(channel="serial", serial_port="COM3"),
        tracking_config=TrackingConfig(hold_time_s=hold_time_s),
        gimbal_config=GimbalConfig(kp=1.0, ki=0.0, kd=0.0),
    )


@pytest.mark.unit
def test_apply_fail_safe_passes_through_when_acquired() -> None:
    proc = _make_proc()
    msg = _make_message(target_acquired=True)

    result = proc._apply_fail_safe(msg, None)

    assert result is msg


@pytest.mark.unit
def test_apply_fail_safe_centers_when_no_last_valid() -> None:
    proc = _make_proc(width=640, height=480)
    msg = _make_message(target_acquired=False, filtered_angles=None)

    result = proc._apply_fail_safe(msg, None)

    assert result.filtered_angles == (0.0, 0.0)
    assert result.state_source == "center"
    assert result.track_id is None
    assert result.confidence == 0.0


@pytest.mark.unit
def test_apply_fail_safe_holds_within_hold_time() -> None:
    proc = _make_proc(hold_time_s=1.0)
    last_valid = _make_message(timestamp_ns=0, target_acquired=True)
    msg = _make_message(
        timestamp_ns=500_000_000,  # 0.5s later, within 1.0s hold
        target_acquired=False,
        filtered_angles=(0.5, 0.6),
    )

    result = proc._apply_fail_safe(msg, last_valid)

    assert result.filtered_angles == (0.5, 0.6)
    assert result.state_source != "center"


@pytest.mark.unit
def test_apply_fail_safe_centers_after_hold_expired() -> None:
    proc = _make_proc(hold_time_s=0.5)
    last_valid = _make_message(timestamp_ns=0, target_acquired=True)
    msg = _make_message(
        timestamp_ns=2_000_000_000,  # 2s later, past 0.5s hold
        target_acquired=False,
        filtered_angles=(0.5, 0.6),
    )

    result = proc._apply_fail_safe(msg, last_valid)

    assert result.filtered_angles == (0.0, 0.0)
    assert result.state_source == "center"


@pytest.mark.unit
def test_center_message_sets_center_pixel() -> None:
    proc = _make_proc(width=640, height=480)
    msg = _make_message()

    result = proc._center_message(msg)

    assert result.filtered_pixel == (320.0, 240.0)
    assert result.raw_pixel is None
    assert result.raw_angles is None
    assert result.filtered_angles == (0.0, 0.0)
    assert result.state_source == "center"
    assert result.track_id is None
    assert result.confidence == 0.0


@pytest.mark.unit
def test_encode_packet_produces_decodable_packet() -> None:
    proc = _make_proc()
    msg = _make_message(filtered_angles=(0.0, 0.0), timestamp_ns=1_234_000_000, confidence=0.8)

    packet = proc._encode_packet(msg, sequence=42)
    decoded = decode_packet_v2(packet)

    assert decoded is not None
    assert decoded.sequence == 42


@pytest.mark.unit
def test_encode_packet_converts_angles_to_centidegrees() -> None:
    # Use very high gain + slew limit so first-frame output saturates to error
    proc = OutputProcess(
        layout=None,  # type: ignore[arg-type]
        write_index=None,  # type: ignore[arg-type]
        result_queue=None,  # type: ignore[arg-type]
        shutdown_event=None,  # type: ignore[arg-type]
        error_queue=None,  # type: ignore[arg-type]
        mode="camera",
        camera_config=CameraConfig(width=640, height=480, fps=30),
        comm_config=CommConfig(channel="serial", serial_port="COM3"),
        tracking_config=TrackingConfig(hold_time_s=0.5),
        gimbal_config=GimbalConfig(
            kp=60.0, kd=0.0, deadband_deg=0.0, slew_limit_dps=1e6,
            tilt_scale=1.0,
        ),
    )
    pan_rad = 0.1
    tilt_rad = -0.2
    msg = _make_message(filtered_angles=(pan_rad, tilt_rad))

    packet = proc._encode_packet(msg, sequence=0)
    decoded = decode_packet_v2(packet)

    assert decoded is not None
    assert decoded.pan == int(round(degrees(pan_rad) * 100.0))
    assert decoded.tilt == int(round(degrees(tilt_rad) * 100.0))


@pytest.mark.unit
def test_encode_packet_clamps_confidence() -> None:
    proc = _make_proc()
    msg_full = _make_message(filtered_angles=(0.0, 0.0), confidence=1.0)
    msg_zero = _make_message(filtered_angles=(0.0, 0.0), confidence=0.0)

    pkt_full = proc._encode_packet(msg_full, sequence=0)
    pkt_zero = proc._encode_packet(msg_zero, sequence=0)

    assert decode_packet_v2(pkt_full).confidence == 255  # type: ignore[union-attr]
    assert decode_packet_v2(pkt_zero).confidence == 0  # type: ignore[union-attr]


@pytest.mark.unit
def test_encode_packet_encodes_velocity() -> None:
    proc = _make_proc()
    pan_vel_rad = 0.15   # rad/s
    tilt_vel_rad = -0.25
    msg = _make_message(
        filtered_angles=(0.0, 0.0),
        angular_velocity=(pan_vel_rad, tilt_vel_rad),
    )

    packet = proc._encode_packet(msg, sequence=0)
    decoded = decode_packet_v2(packet)

    assert decoded is not None
    assert decoded.pan_vel == int(round(degrees(pan_vel_rad) * 100.0))
    assert decoded.tilt_vel == int(round(degrees(tilt_vel_rad) * 100.0))


@pytest.mark.unit
def test_center_message_clears_velocity() -> None:
    proc = _make_proc(width=640, height=480)
    msg = _make_message(
        angular_velocity=(0.1, 0.2),
        filtered_velocity=(10.0, 20.0),
    )

    result = proc._center_message(msg)

    assert result.filtered_velocity is None
    assert result.angular_velocity is None


# --- Sign inversion tests ---


def _make_proc_with_gimbal(
    invert_pan: bool = False, invert_tilt: bool = False
) -> OutputProcess:
    return OutputProcess(
        layout=None,  # type: ignore[arg-type]
        write_index=None,  # type: ignore[arg-type]
        result_queue=None,  # type: ignore[arg-type]
        shutdown_event=None,  # type: ignore[arg-type]
        error_queue=None,  # type: ignore[arg-type]
        mode="camera",
        camera_config=CameraConfig(width=640, height=640, fps=30),
        comm_config=CommConfig(channel="serial", serial_port="COM3"),
        tracking_config=TrackingConfig(),
        gimbal_config=GimbalConfig(
            invert_pan=invert_pan,
            invert_tilt=invert_tilt,
            kp=1.0, ki=0.0, kd=0.0,
        ),
    )


def _make_proc_with_manual_state(
    *,
    manual_mode: bool,
    manual_pan: float = 0.0,
    manual_tilt: float = 0.0,
    laser_enabled: bool = True,
    gimbal_config: GimbalConfig | None = None,
) -> tuple[OutputProcess, object, object, object, object]:
    manual_mode_value = mp.Value("b", int(manual_mode))
    manual_pan_value = mp.Value("d", manual_pan)
    manual_tilt_value = mp.Value("d", manual_tilt)
    laser_enabled_value = mp.Value("b", int(laser_enabled))
    proc = OutputProcess(
        layout=None,  # type: ignore[arg-type]
        write_index=None,  # type: ignore[arg-type]
        result_queue=None,  # type: ignore[arg-type]
        shutdown_event=None,  # type: ignore[arg-type]
        error_queue=None,  # type: ignore[arg-type]
        mode="camera",
        camera_config=CameraConfig(width=640, height=640, fps=30),
        comm_config=CommConfig(channel="serial", serial_port="COM3"),
        tracking_config=TrackingConfig(),
        gimbal_config=gimbal_config or GimbalConfig(kp=1.0, ki=0.0, kd=0.0),
        laser_enabled=laser_enabled_value,
        manual_mode=manual_mode_value,
        manual_pan=manual_pan_value,
        manual_tilt=manual_tilt_value,
    )
    return proc, manual_mode_value, manual_pan_value, manual_tilt_value, laser_enabled_value


@pytest.mark.unit
def test_sign_inversion_tilt_negates_tilt_angle() -> None:
    proc_normal = _make_proc()
    proc_invert = _make_proc_with_gimbal(invert_tilt=True)
    msg = _make_message(filtered_angles=(0.1, 0.2), angular_velocity=(0.05, 0.1))

    pkt_normal = proc_normal._encode_packet(msg, sequence=0)
    pkt_invert = proc_invert._encode_packet(msg, sequence=0)

    dec_normal = decode_packet_v2(pkt_normal)
    dec_invert = decode_packet_v2(pkt_invert)

    assert dec_normal.pan == dec_invert.pan  # pan unchanged
    assert dec_normal.tilt == -dec_invert.tilt  # tilt inverted
    assert dec_normal.pan_vel == dec_invert.pan_vel
    assert dec_normal.tilt_vel == -dec_invert.tilt_vel


@pytest.mark.unit
def test_sign_inversion_pan_negates_pan_angle() -> None:
    proc_normal = _make_proc()
    proc_invert = _make_proc_with_gimbal(invert_pan=True)
    msg = _make_message(filtered_angles=(0.1, 0.2), angular_velocity=(0.05, 0.1))

    pkt_normal = proc_normal._encode_packet(msg, sequence=0)
    pkt_invert = proc_invert._encode_packet(msg, sequence=0)

    dec_normal = decode_packet_v2(pkt_normal)
    dec_invert = decode_packet_v2(pkt_invert)

    assert dec_normal.pan == -dec_invert.pan  # pan inverted
    assert dec_normal.tilt == dec_invert.tilt  # tilt unchanged
    assert dec_normal.pan_vel == -dec_invert.pan_vel
    assert dec_normal.tilt_vel == dec_invert.tilt_vel


@pytest.mark.unit
def test_sign_inversion_both_axes() -> None:
    proc_normal = _make_proc_with_gimbal(invert_pan=False, invert_tilt=False)
    proc_invert = _make_proc_with_gimbal(invert_pan=True, invert_tilt=True)
    msg = _make_message(filtered_angles=(0.1, 0.2), angular_velocity=(0.05, 0.1))

    dec_normal = decode_packet_v2(proc_normal._encode_packet(msg, sequence=0))
    dec_invert = decode_packet_v2(proc_invert._encode_packet(msg, sequence=0))

    # Both axes negated
    assert dec_normal.pan == -dec_invert.pan
    assert dec_normal.tilt == -dec_invert.tilt


@pytest.mark.unit
def test_encode_packet_manual_mode_respects_inversion_and_clears_tracking_flags() -> None:
    proc, _, _, _, _ = _make_proc_with_manual_state(
        manual_mode=True,
        manual_pan=12.5,
        manual_tilt=-7.25,
        gimbal_config=GimbalConfig(invert_pan=True, invert_tilt=True),
    )
    msg = _make_message(
        filtered_angles=(0.1, 0.2),
        angular_velocity=(0.5, -0.4),
        target_acquired=True,
        confidence=0.95,
    )

    packet = proc._encode_packet(msg, sequence=0)
    decoded = decode_packet_v2(packet)

    assert decoded is not None
    assert decoded.pan == -1250
    assert decoded.tilt == 725
    assert decoded.pan_vel == 0
    assert decoded.tilt_vel == 0
    assert decoded.confidence == 0
    assert not (decoded.state & FLAG_TARGET_ACQUIRED)
    assert decoded.state & FLAG_LASER_ON


@pytest.mark.unit
def test_encode_packet_manual_mode_laser_respects_user_toggle() -> None:
    proc, _, _, _, laser_enabled = _make_proc_with_manual_state(
        manual_mode=True,
        laser_enabled=False,
    )
    msg = _make_message(target_acquired=False, laser_pixel=None)

    packet = proc._encode_packet(msg, sequence=0)
    decoded = decode_packet_v2(packet)

    assert decoded is not None
    assert laser_enabled.value == 0
    assert not (decoded.state & FLAG_LASER_ON)


@pytest.mark.unit
def test_encode_packet_syncs_manual_handoff_with_live_auto_command() -> None:
    proc, _, manual_pan, manual_tilt, _ = _make_proc_with_manual_state(
        manual_mode=False,
        gimbal_config=GimbalConfig(invert_pan=True, invert_tilt=True),
    )
    msg = _make_message(filtered_angles=(0.1, -0.2))

    packet = proc._encode_packet(msg, sequence=0)
    decoded = decode_packet_v2(packet)

    assert decoded is not None
    assert manual_pan.value == pytest.approx(-decoded.pan / 100.0, abs=0.01)
    assert manual_tilt.value == pytest.approx(-decoded.tilt / 100.0, abs=0.01)


@pytest.mark.unit
def test_encode_packet_manual_motion_sets_fast_response_flag() -> None:
    proc, _, manual_pan, _, _ = _make_proc_with_manual_state(manual_mode=True)
    first_msg = _make_message(timestamp_ns=1_000_000_000)
    second_msg = _make_message(timestamp_ns=1_050_000_000)

    first_packet = proc._encode_packet(first_msg, sequence=0)
    first_decoded = decode_packet_v2(first_packet)

    manual_pan.value = 1.0
    second_packet = proc._encode_packet(second_msg, sequence=1)
    second_decoded = decode_packet_v2(second_packet)

    assert first_decoded is not None
    assert second_decoded is not None
    assert first_decoded.pan_vel == 0
    assert not (first_decoded.state & FLAG_FAST_MOTION)
    assert second_decoded.pan_vel == 8000  # boosted to 80°/s floor × 100
    assert second_decoded.state & FLAG_FAST_MOTION


# --- _center_message servo field tests ---


@pytest.mark.unit
def test_center_message_zeros_servo_angles() -> None:
    proc = _make_proc(width=640, height=480)
    msg = _make_message(servo_angles=(0.3, 0.4))

    result = proc._center_message(msg)

    assert result.servo_angles == (0.0, 0.0)


@pytest.mark.unit
def test_center_message_zeros_servo_velocity() -> None:
    proc = _make_proc(width=640, height=480)
    msg = _make_message(servo_angular_velocity=(0.1, 0.2))

    result = proc._center_message(msg)

    assert result.servo_angular_velocity == (0.0, 0.0)


@pytest.mark.unit
def test_center_message_sets_servo_mode_center() -> None:
    proc = _make_proc(width=640, height=480)
    msg = _make_message(servo_mode="acquisition")

    result = proc._center_message(msg)

    assert result.servo_mode == "center"


# --- FLAG_LASER_ON conditional tests ---


@pytest.mark.unit
def test_encode_packet_laser_flag_when_target_acquired() -> None:
    """Laser stays armed by default whenever tracking has a target."""
    proc = _make_proc()
    msg = _make_message(filtered_angles=(0.0, 0.0), target_acquired=True, laser_pixel=None)

    packet = proc._encode_packet(msg, sequence=0)
    decoded = decode_packet_v2(packet)

    assert decoded is not None
    assert decoded.state & (1 << 6), "FLAG_LASER_ON should be set when target_acquired"


@pytest.mark.unit
def test_encode_packet_no_laser_flag_when_target_lost() -> None:
    """Laser OFF when target_acquired=False."""
    proc = _make_proc()
    msg = _make_message(filtered_angles=(0.0, 0.0), target_acquired=False, laser_pixel=None)

    packet = proc._encode_packet(msg, sequence=0)
    decoded = decode_packet_v2(packet)

    assert decoded is not None
    assert not (decoded.state & (1 << 6)), "FLAG_LASER_ON should not be set when target lost"


@pytest.mark.unit
def test_encode_packet_no_laser_flag_when_user_disabled() -> None:
    proc, _, _, _, laser_enabled = _make_proc_with_manual_state(
        manual_mode=False,
        laser_enabled=False,
    )
    msg = _make_message(filtered_angles=(0.0, 0.0), target_acquired=True, laser_pixel=None)

    packet = proc._encode_packet(msg, sequence=0)
    decoded = decode_packet_v2(packet)

    assert decoded is not None
    assert laser_enabled.value == 0
    assert not (decoded.state & FLAG_LASER_ON)


@pytest.mark.unit
def test_no_inversion_by_default() -> None:
    proc = _make_proc()
    msg = _make_message(filtered_angles=(0.1, 0.2))

    pkt = proc._encode_packet(msg, sequence=0)
    dec = decode_packet_v2(pkt)

    # Default: no inversion → positive angles produce positive output
    assert dec.pan > 0
    assert dec.tilt > 0


# --- _create_sender tests ---


def _make_full_proc(
    comm_config: CommConfig | None = None,
    mode: str = "camera",
    headless: bool = True,
) -> OutputProcess:
    layout, write_index = SharedRingBuffer.create((4, 4, 3), num_slots=2)
    layout.cleanup()
    return OutputProcess(
        layout=layout.layout,
        write_index=write_index,
        result_queue=mp.Queue(),
        shutdown_event=mp.Event(),
        error_queue=mp.Queue(),
        mode=mode,
        camera_config=CameraConfig(width=640, height=480, fps=30),
        comm_config=comm_config or CommConfig(channel="serial", serial_port="COM3"),
        tracking_config=TrackingConfig(),
        flags=RuntimeFlags(headless=headless),
    )


@pytest.mark.unit
def test_get_transport_status_reports_offline_when_no_sender() -> None:
    label, color = _get_transport_status(None)

    assert label == "LINK OFFLINE"
    assert color == (50, 50, 255)


@pytest.mark.unit
def test_get_transport_status_reports_serial_sender() -> None:
    sender = SerialComm.__new__(SerialComm)
    sender._connected = True

    label, color = _get_transport_status(sender)

    assert label == "LINK SERIAL"
    assert color == (50, 255, 50)


@pytest.mark.unit
def test_get_transport_status_reports_wifi_for_auto_udp_sender() -> None:
    class AutoSenderProbe:
        def __init__(self) -> None:
            self.active_channel = "udp"
            self.is_connected = True

    label, color = _get_transport_status(AutoSenderProbe())

    assert label == "LINK WIFI"
    assert color == (50, 255, 50)


@pytest.mark.unit
def test_get_transport_status_reports_offline_for_disconnected_udp_sender() -> None:
    class AutoSenderProbe:
        def __init__(self) -> None:
            self.active_channel = "udp"
            self.is_connected = False

    label, color = _get_transport_status(AutoSenderProbe())

    assert label == "LINK OFFLINE"
    assert color == (50, 50, 255)


@pytest.mark.unit
def test_run_loop_logs_display_backpressure_warning_once_per_window(monkeypatch, caplog) -> None:
    layout, write_index = SharedRingBuffer.create((4, 4, 3), num_slots=2)
    result_queue = mp.Queue()
    display_queue = MagicMock()
    import numpy as np

    display_queue.put_nowait.side_effect = [Exception("full"), None]
    proc = OutputProcess(
        layout=layout.layout,
        write_index=write_index,
        result_queue=result_queue,
        shutdown_event=mp.Event(),
        error_queue=mp.Queue(),
        mode="camera",
        camera_config=CameraConfig(width=4, height=4, fps=30),
        comm_config=CommConfig(enabled=False),
        tracking_config=TrackingConfig(),
        flags=RuntimeFlags(headless=False),
        display_queue=display_queue,
    )

    msg = _make_message(filtered_angles=(0.0, 0.0), frame_id=0)
    result_queue.put(msg)
    result_queue.put(None)

    mock_record = MagicMock()
    mock_record.frame = np.zeros((4, 4, 3), dtype=np.uint8)
    mock_rb = MagicMock()
    mock_rb.read_frame.return_value = mock_record
    fake_times = iter([0.0, 0.0, 1.2, 1.2, 1.2])
    monkeypatch.setattr("src.output.process.SharedRingBuffer.attach", lambda layout, wi: mock_rb)
    monkeypatch.setattr("src.output.process.draw_overlay", lambda frame, msg, **kw: frame)
    monkeypatch.setattr("src.output.process.time.perf_counter", lambda: next(fake_times))

    with caplog.at_level(logging.WARNING):
        proc.run()

    assert "Display backpressure" in caplog.text
    layout.cleanup()


@pytest.mark.unit
def test_create_sender_serial(monkeypatch) -> None:
    proc = _make_full_proc(comm_config=CommConfig(channel="serial", serial_port="COM99"))
    mock_serial = MagicMock()
    monkeypatch.setattr("src.output.process.SerialComm", lambda port, baud: mock_serial)

    result = proc._create_sender()

    assert result is mock_serial


@pytest.mark.unit
def test_create_sender_udp(monkeypatch) -> None:
    proc = _make_full_proc(comm_config=CommConfig(channel="udp", udp_host="1.2.3.4", udp_port=9999))
    mock_udp = MagicMock()
    monkeypatch.setattr("src.output.process.UDPComm", lambda host, port, **kw: mock_udp)

    result = proc._create_sender()

    assert result is mock_udp


@pytest.mark.unit
def test_create_sender_returns_none_on_failure(monkeypatch) -> None:
    proc = _make_full_proc(comm_config=CommConfig(channel="serial", serial_port="COM99"))
    monkeypatch.setattr("src.output.process.SerialComm", MagicMock(side_effect=OSError("no port")))

    result = proc._create_sender()

    assert result is None


@pytest.mark.unit
def test_create_sender_auto_returns_auto_transport(monkeypatch) -> None:
    proc = _make_full_proc(
        comm_config=CommConfig(channel="auto", serial_port="COM99", udp_host="1.2.3.4", udp_port=9999)
    )
    sentinel = MagicMock(spec=AutoCommTransport)
    monkeypatch.setattr("src.output.process.AutoCommTransport", lambda config: sentinel)

    result = proc._create_sender()

    assert result is sentinel


@pytest.mark.unit
def test_create_sender_auto_returns_none_when_transport_init_fails(monkeypatch) -> None:
    proc = _make_full_proc(
        comm_config=CommConfig(channel="auto", serial_port="COM99", udp_host="1.2.3.4", udp_port=9999)
    )
    monkeypatch.setattr(
        "src.output.process.AutoCommTransport",
        MagicMock(side_effect=RuntimeError("boom")),
    )

    result = proc._create_sender()

    assert result is None


# --- _report_error tests ---


@pytest.mark.unit
def test_report_error_puts_to_queue() -> None:
    layout, write_index = SharedRingBuffer.create((4, 4, 3), num_slots=2)
    error_queue = mp.Queue()
    proc = OutputProcess(
        layout=layout.layout,
        write_index=write_index,
        result_queue=mp.Queue(),
        shutdown_event=mp.Event(),
        error_queue=error_queue,
        mode="camera",
        camera_config=CameraConfig(width=640, height=480, fps=30),
        comm_config=CommConfig(),
        tracking_config=TrackingConfig(),
    )

    proc._report_error(RuntimeError("test failure"))

    report = error_queue.get(timeout=1.0)
    assert report.process_name == "OutputProcess"
    assert "test failure" in report.summary
    layout.cleanup()


# --- run() loop tests ---


@pytest.mark.unit
def test_run_loop_processes_messages_and_stops(monkeypatch) -> None:
    """Test run() dequeues messages, applies fail-safe, encodes, and sends."""
    layout, write_index = SharedRingBuffer.create((4, 4, 3), num_slots=2)
    result_queue = mp.Queue()
    shutdown_event = mp.Event()
    mock_sender = MagicMock()
    mock_sender.is_connected = True

    proc = OutputProcess(
        layout=layout.layout,
        write_index=write_index,
        result_queue=result_queue,
        shutdown_event=shutdown_event,
        error_queue=mp.Queue(),
        mode="camera",
        camera_config=CameraConfig(width=640, height=480, fps=30),
        comm_config=CommConfig(channel="serial", serial_port="COM99"),
        tracking_config=TrackingConfig(),
        flags=RuntimeFlags(headless=True),
    )

    # Feed two messages, then None sentinel to exit
    msg = _make_message(filtered_angles=(0.05, 0.1))
    result_queue.put(msg)
    result_queue.put(None)  # sentinel to break

    monkeypatch.setattr(proc, "_create_sender", lambda: mock_sender)
    # Stub SharedRingBuffer.attach to return a MagicMock ring buffer
    mock_rb = MagicMock()
    monkeypatch.setattr("src.output.process.SharedRingBuffer.attach", lambda layout, wi: mock_rb)

    proc.run()

    assert mock_sender.send.call_count == 1
    mock_sender.close.assert_called_once()
    mock_rb.close.assert_called_once()
    layout.cleanup()


@pytest.mark.unit
def test_run_loop_reconnects_sender(monkeypatch) -> None:
    """Test run() calls reconnect when sender is not connected."""
    layout, write_index = SharedRingBuffer.create((4, 4, 3), num_slots=2)
    result_queue = mp.Queue()
    mock_sender = MagicMock()
    mock_sender.is_connected = False  # Force reconnect path

    proc = OutputProcess(
        layout=layout.layout,
        write_index=write_index,
        result_queue=result_queue,
        shutdown_event=mp.Event(),
        error_queue=mp.Queue(),
        mode="camera",
        camera_config=CameraConfig(width=640, height=480, fps=30),
        comm_config=CommConfig(channel="serial", serial_port="COM99"),
        tracking_config=TrackingConfig(),
        flags=RuntimeFlags(headless=True),
    )

    msg = _make_message(filtered_angles=(0.05, 0.1))
    result_queue.put(msg)
    result_queue.put(None)

    monkeypatch.setattr(proc, "_create_sender", lambda: mock_sender)
    mock_rb = MagicMock()
    monkeypatch.setattr("src.output.process.SharedRingBuffer.attach", lambda layout, wi: mock_rb)

    proc.run()

    mock_sender.reconnect.assert_called_once()
    mock_sender.send.assert_called_once()
    layout.cleanup()


@pytest.mark.unit
def test_run_loop_no_sender_when_comm_disabled(monkeypatch) -> None:
    """When comm is disabled, run() should not create a sender."""
    layout, write_index = SharedRingBuffer.create((4, 4, 3), num_slots=2)
    result_queue = mp.Queue()
    result_queue.put(None)

    proc = OutputProcess(
        layout=layout.layout,
        write_index=write_index,
        result_queue=result_queue,
        shutdown_event=mp.Event(),
        error_queue=mp.Queue(),
        mode="camera",
        camera_config=CameraConfig(width=640, height=480, fps=30),
        comm_config=CommConfig(enabled=False),
        tracking_config=TrackingConfig(),
        flags=RuntimeFlags(headless=True),
    )

    mock_rb = MagicMock()
    monkeypatch.setattr("src.output.process.SharedRingBuffer.attach", lambda layout, wi: mock_rb)

    proc.run()

    mock_rb.close.assert_called_once()
    layout.cleanup()


@pytest.mark.unit
def test_run_loop_shutdown_event_exits(monkeypatch) -> None:
    """run() should exit when shutdown_event is set even with no messages."""
    layout, write_index = SharedRingBuffer.create((4, 4, 3), num_slots=2)
    shutdown_event = mp.Event()
    shutdown_event.set()  # Pre-set so loop exits immediately

    proc = OutputProcess(
        layout=layout.layout,
        write_index=write_index,
        result_queue=mp.Queue(),
        shutdown_event=shutdown_event,
        error_queue=mp.Queue(),
        mode="camera",
        camera_config=CameraConfig(width=640, height=480, fps=30),
        comm_config=CommConfig(enabled=False),
        tracking_config=TrackingConfig(),
        flags=RuntimeFlags(headless=True),
    )

    mock_rb = MagicMock()
    monkeypatch.setattr("src.output.process.SharedRingBuffer.attach", lambda layout, wi: mock_rb)

    proc.run()  # Should return quickly

    mock_rb.close.assert_called_once()
    layout.cleanup()


@pytest.mark.unit
def test_run_loop_display_queue_gets_frames(monkeypatch) -> None:
    """When headless=False and display_queue exists, frames get sent to display."""
    layout, write_index = SharedRingBuffer.create((4, 4, 3), num_slots=2)
    result_queue = mp.Queue()
    display_queue = MagicMock()
    import numpy as np

    proc = OutputProcess(
        layout=layout.layout,
        write_index=write_index,
        result_queue=result_queue,
        shutdown_event=mp.Event(),
        error_queue=mp.Queue(),
        mode="camera",
        camera_config=CameraConfig(width=4, height=4, fps=30),
        comm_config=CommConfig(enabled=False),
        tracking_config=TrackingConfig(),
        flags=RuntimeFlags(headless=False),
        display_queue=display_queue,
    )

    msg = _make_message(filtered_angles=(0.0, 0.0), frame_id=0)
    result_queue.put(msg)
    result_queue.put(None)

    # Mock ring buffer to return a fake record
    mock_record = MagicMock()
    mock_record.frame = np.zeros((4, 4, 3), dtype=np.uint8)
    mock_rb = MagicMock()
    mock_rb.read_frame.return_value = mock_record
    monkeypatch.setattr("src.output.process.SharedRingBuffer.attach", lambda layout, wi: mock_rb)
    monkeypatch.setattr("src.output.process.draw_overlay", lambda frame, msg, **kw: frame)

    proc.run()

    display_queue.put_nowait.assert_called_once()
    layout.cleanup()


@pytest.mark.unit
def test_run_loop_display_fallback_to_latest(monkeypatch) -> None:
    """When read_frame returns None, falls back to read_latest."""
    layout, write_index = SharedRingBuffer.create((4, 4, 3), num_slots=2)
    result_queue = mp.Queue()
    display_queue = MagicMock()
    import numpy as np

    proc = OutputProcess(
        layout=layout.layout,
        write_index=write_index,
        result_queue=result_queue,
        shutdown_event=mp.Event(),
        error_queue=mp.Queue(),
        mode="camera",
        camera_config=CameraConfig(width=4, height=4, fps=30),
        comm_config=CommConfig(enabled=False),
        tracking_config=TrackingConfig(),
        flags=RuntimeFlags(headless=False),
        display_queue=display_queue,
    )

    msg = _make_message(filtered_angles=(0.0, 0.0), frame_id=999)
    result_queue.put(msg)
    result_queue.put(None)

    mock_record = MagicMock()
    mock_record.frame = np.zeros((4, 4, 3), dtype=np.uint8)
    mock_rb = MagicMock()
    mock_rb.read_frame.return_value = None
    mock_rb.read_latest.return_value = mock_record
    monkeypatch.setattr("src.output.process.SharedRingBuffer.attach", lambda layout, wi: mock_rb)
    monkeypatch.setattr("src.output.process.draw_overlay", lambda frame, msg, **kw: frame)

    proc.run()

    mock_rb.read_latest.assert_called_once()
    display_queue.put_nowait.assert_called_once()
    layout.cleanup()


@pytest.mark.unit
def test_run_loop_exception_reports_error(monkeypatch) -> None:
    """When an exception occurs in run(), it gets reported to error_queue."""
    layout, write_index = SharedRingBuffer.create((4, 4, 3), num_slots=2)
    error_queue = mp.Queue()
    shutdown_event = mp.Event()

    proc = OutputProcess(
        layout=layout.layout,
        write_index=write_index,
        result_queue=mp.Queue(),
        shutdown_event=shutdown_event,
        error_queue=error_queue,
        mode="camera",
        camera_config=CameraConfig(width=4, height=4, fps=30),
        comm_config=CommConfig(enabled=False),
        tracking_config=TrackingConfig(),
        flags=RuntimeFlags(headless=True),
    )

    mock_rb = MagicMock()
    mock_rb.close.return_value = None
    monkeypatch.setattr("src.output.process.SharedRingBuffer.attach", lambda layout, wi: mock_rb)
    # Make result_queue.get raise to trigger exception path
    proc._result_queue = MagicMock()
    proc._result_queue.get.side_effect = RuntimeError("boom")

    proc.run()

    assert shutdown_event.is_set()
    report = error_queue.get(timeout=1.0)
    assert "boom" in report.summary
    layout.cleanup()


@pytest.mark.unit
def test_run_loop_fps_logging_triggers_at_300(monkeypatch) -> None:
    """FPS logging triggers every 300 frames and resets counter."""
    layout, write_index = SharedRingBuffer.create((4, 4, 3), num_slots=2)
    result_queue = mp.Queue()

    proc = OutputProcess(
        layout=layout.layout,
        write_index=write_index,
        result_queue=result_queue,
        shutdown_event=mp.Event(),
        error_queue=mp.Queue(),
        mode="camera",
        camera_config=CameraConfig(width=4, height=4, fps=30),
        comm_config=CommConfig(enabled=False),
        tracking_config=TrackingConfig(),
        flags=RuntimeFlags(headless=True),
    )

    # Feed 301 messages then sentinel
    for i in range(301):
        result_queue.put(_make_message(filtered_angles=(0.0, 0.0), frame_id=i))
    result_queue.put(None)

    mock_rb = MagicMock()
    monkeypatch.setattr("src.output.process.SharedRingBuffer.attach", lambda layout, wi: mock_rb)

    proc.run()

    # After 300 frames, counter resets to 0, then +1 more = 1
    assert proc._frames_processed == 1
    layout.cleanup()
