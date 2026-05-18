from __future__ import annotations

from src.config_loader import build_config_from_yaml
from src.gui import runtime_session
from src.gui.runtime_session import RuntimeSession
from src.gui.view_model import TuningSettings


def _config():
    return build_config_from_yaml(
        {
            "mode": "video",
            "source": "videos/sample.mp4",
            "camera": {"width": 320, "height": 240, "fps": 30},
            "comms": {"enabled": False},
            "laser": {"enabled": False},
        }
    )


def _fake_process_class(name: str):
    class FakeProcess:
        def __init__(self, *args, **kwargs) -> None:
            self.name = name
            self.pid: int | None = None
            self.exitcode: int | None = None
            self._alive = False

        def start(self) -> None:
            self.pid = 1000 + len(name)
            self.exitcode = None
            self._alive = True

        def is_alive(self) -> bool:
            return self._alive

        def join(self, timeout: float | None = None) -> None:
            self._alive = False
            self.exitcode = 0

        def terminate(self) -> None:
            self._alive = False
            self.exitcode = -15

    return FakeProcess


def test_runtime_session_starts_existing_process_graph_with_fake_processes(monkeypatch):
    monkeypatch.setattr(runtime_session, "CaptureProcess", _fake_process_class("CaptureProcess"))
    monkeypatch.setattr(
        runtime_session, "InferenceProcess", _fake_process_class("InferenceProcess")
    )
    monkeypatch.setattr(runtime_session, "OutputProcess", _fake_process_class("OutputProcess"))

    session = RuntimeSession(_config(), validate_on_start=False)
    try:
        snapshot = session.start()

        assert snapshot.running is True
        assert [process.name for process in snapshot.process_states] == [
            "CaptureProcess",
            "InferenceProcess",
            "OutputProcess",
        ]
        assert all(process.alive for process in snapshot.process_states)
        assert snapshot.display_available is True
        assert snapshot.laser_enabled is False
    finally:
        session.stop(join_timeout=0.01, terminate_timeout=0.01)


def test_runtime_session_emergency_stop_clears_safety_flags(monkeypatch):
    monkeypatch.setattr(runtime_session, "CaptureProcess", _fake_process_class("CaptureProcess"))
    monkeypatch.setattr(
        runtime_session, "InferenceProcess", _fake_process_class("InferenceProcess")
    )
    monkeypatch.setattr(runtime_session, "OutputProcess", _fake_process_class("OutputProcess"))

    session = RuntimeSession(_config(), validate_on_start=False)
    try:
        session.start()
        session.set_laser_enabled(True)
        session.pulse_relay()

        snapshot = session.emergency_stop()

        assert snapshot.stopping is True
        assert snapshot.laser_enabled is False
        assert snapshot.relay_active is False
        assert snapshot.manual_mode is False
    finally:
        session.stop(join_timeout=0.01, terminate_timeout=0.01)


def test_runtime_session_queues_live_output_tuning(monkeypatch):
    monkeypatch.setattr(runtime_session, "CaptureProcess", _fake_process_class("CaptureProcess"))
    monkeypatch.setattr(
        runtime_session, "InferenceProcess", _fake_process_class("InferenceProcess")
    )
    monkeypatch.setattr(runtime_session, "OutputProcess", _fake_process_class("OutputProcess"))

    session = RuntimeSession(_config(), validate_on_start=False)
    try:
        session.start()
        version = session.apply_runtime_tuning(
            TuningSettings(
                tracking_confidence_threshold=0.5,
                tracking_hold_time_s=0.7,
                gimbal_kp=1.4,
                gimbal_ki=0.0,
                gimbal_kd=0.5,
                gimbal_deadband_deg=1.0,
                gimbal_slew_limit_dps=30.0,
                gimbal_kp_near=None,
                gimbal_kp_far=1.8,
                gimbal_predictive_lead_s=0.01,
                laser_startup_enabled=True,
                relay_pulse_ms=250,
                boresight_pan_offset_deg=1.0,
                boresight_tilt_offset_deg=-0.5,
            )
        )
        snapshot = session.snapshot()

        assert version == 1
        assert snapshot.runtime_control_version == 1
        assert snapshot.runtime_control_ack_version == 0
        assert snapshot.laser_enabled is True
        assert session._control_queue is not None  # noqa: SLF001
        assert session._control_queue.get(timeout=1.0).gimbal_kp == 1.4  # noqa: SLF001
    finally:
        session.stop(join_timeout=0.01, terminate_timeout=0.01)
