from __future__ import annotations

import multiprocessing as mp
from pathlib import Path
from queue import Empty, Queue
import signal
import sys

import pytest

from src.main import (
    _all_processes_stopped,
    _check_runtime_failures,
    _cleanup_runtime_resources,
    _drain_error_reports,
    _is_quit_hotkey,
    _setup_signal_handlers,
    _stop_processes,
    parse_args,
)
from src.shared.types import ProcessErrorReport


class _FakeProcess:
    def __init__(
        self,
        name: str,
        alive_states: list[bool],
        exitcode: int | None = None,
        pid: int | None = 1234,
    ) -> None:
        self.name = name
        self._alive_states = list(alive_states)
        self.exitcode = exitcode
        self.pid = pid
        self.join_calls: list[float] = []
        self.terminate_calls = 0

    def is_alive(self) -> bool:
        return self._alive_states[0] if self._alive_states else False

    def join(self, timeout: float | None = None) -> None:
        self.join_calls.append(timeout if timeout is not None else -1.0)
        if len(self._alive_states) > 1:
            self._alive_states.pop(0)

    def terminate(self) -> None:
        self.terminate_calls += 1
        self._alive_states = [False]


class _ScriptedQueue:
    def __init__(self, items: list[ProcessErrorReport | type[Empty]]) -> None:
        self._items = list(items)

    def get_nowait(self) -> ProcessErrorReport:
        if not self._items:
            raise Empty
        item = self._items.pop(0)
        if item is Empty:
            raise Empty
        return item


class _DummyRingBuffer:
    def __init__(self, should_fail: bool = False) -> None:
        self.should_fail = should_fail
        self.cleanup_calls = 0

    def cleanup(self) -> None:
        self.cleanup_calls += 1
        if self.should_fail:
            raise RuntimeError("cleanup failed")


@pytest.mark.unit
def test_signal_handler_function_sets_shutdown_event() -> None:
    shutdown_event = mp.Event()
    # Capture the installed handler without actually raising a signal
    _setup_signal_handlers(shutdown_event)
    installed_handler = signal.getsignal(signal.SIGINT)

    # Invoke the handler directly (signum, frame) as the OS would
    installed_handler(signal.SIGINT, None)  # type: ignore[call-arg]

    assert shutdown_event.is_set()


@pytest.mark.unit
def test_signal_handler_idempotent_when_called_twice() -> None:
    shutdown_event = mp.Event()
    _setup_signal_handlers(shutdown_event)
    handler = signal.getsignal(signal.SIGINT)

    handler(signal.SIGINT, None)  # type: ignore[call-arg]
    handler(signal.SIGINT, None)  # type: ignore[call-arg]

    assert shutdown_event.is_set()


@pytest.mark.unit
def test_error_report_has_expected_fields() -> None:
    report = ProcessErrorReport(
        process_name="TestProcess",
        summary="something broke",
        traceback_text="Traceback...",
        timestamp_ns=1_000_000,
        severity="error",
    )

    assert report.process_name == "TestProcess"
    assert report.summary == "something broke"
    assert report.severity == "error"


@pytest.mark.unit
def test_is_quit_hotkey_accepts_escape_when_enabled() -> None:
    assert _is_quit_hotkey(27, no_quit=False)


@pytest.mark.unit
def test_is_quit_hotkey_ignores_quit_letter_to_preserve_manual_azerty_controls() -> None:
    assert not _is_quit_hotkey(ord("q"), no_quit=False)


@pytest.mark.unit
def test_is_quit_hotkey_honors_no_quit_flag() -> None:
    assert not _is_quit_hotkey(27, no_quit=True)


@pytest.mark.unit
def test_parse_args_rejects_removed_camera_mount_flag(monkeypatch: pytest.MonkeyPatch) -> None:
    monkeypatch.setattr(sys, "argv", ["tracker", "--camera-mount", "on_gimbal"])

    with pytest.raises(SystemExit):
        parse_args()


@pytest.mark.unit
def test_parse_args_leaves_yaml_owned_overrides_unset(monkeypatch: pytest.MonkeyPatch) -> None:
    monkeypatch.setattr(sys, "argv", ["tracker"])

    args = parse_args()

    assert args.width is None
    assert args.height is None
    assert args.fps is None
    assert args.backend is None
    assert args.log_level is None


@pytest.mark.unit
def test_drain_error_reports_returns_all_queued_reports() -> None:
    error_queue: Queue[ProcessErrorReport] = Queue()
    first = ProcessErrorReport(
        process_name="CaptureProcess",
        summary="camera failed",
        traceback_text="trace-1",
        timestamp_ns=1,
        severity="error",
    )
    second = ProcessErrorReport(
        process_name="InferenceProcess",
        summary="engine failed",
        traceback_text="trace-2",
        timestamp_ns=2,
        severity="error",
    )
    error_queue.put(first)
    error_queue.put(second)

    reports = _drain_error_reports(error_queue)

    assert reports == [first, second]
    assert error_queue.empty()


@pytest.mark.unit
def test_check_runtime_failures_collects_late_report_after_dead_process() -> None:
    late_report = ProcessErrorReport(
        process_name="OutputProcess",
        summary="socket failed",
        traceback_text="trace-late",
        timestamp_ns=3,
        severity="error",
    )
    error_queue = _ScriptedQueue([Empty, late_report, Empty])
    processes = [_FakeProcess("OutputProcess", [False], exitcode=1)]

    reports, dead_processes = _check_runtime_failures(processes, error_queue)  # type: ignore[arg-type]

    assert reports == [late_report]
    assert [process.name for process in dead_processes] == ["OutputProcess"]


@pytest.mark.unit
def test_all_processes_stopped_requires_every_process_to_exit() -> None:
    processes = [
        _FakeProcess("CaptureProcess", [False], exitcode=0),
        _FakeProcess("InferenceProcess", [True], exitcode=None),
    ]

    assert not _all_processes_stopped(processes)  # type: ignore[arg-type]


@pytest.mark.unit
def test_stop_processes_terminates_only_hung_processes() -> None:
    graceful = _FakeProcess("CaptureProcess", [True, False], exitcode=0)
    hung = _FakeProcess("InferenceProcess", [True, True], exitcode=None)

    forced = _stop_processes([graceful, hung], join_timeout=5.0, terminate_timeout=2.0)  # type: ignore[arg-type]

    assert forced == ["InferenceProcess"]
    assert graceful.join_calls == [5.0]
    assert graceful.terminate_calls == 0
    assert hung.join_calls == [5.0, 2.0]
    assert hung.terminate_calls == 1


@pytest.mark.unit
def test_stop_processes_skips_never_started_processes() -> None:
    not_started = _FakeProcess("OutputProcess", [False], exitcode=None, pid=None)

    forced = _stop_processes([not_started], join_timeout=5.0, terminate_timeout=2.0)  # type: ignore[arg-type]

    assert forced == []
    assert not_started.join_calls == []
    assert not_started.terminate_calls == 0


@pytest.mark.unit
def test_cleanup_runtime_resources_removes_pid_lock_even_on_cleanup_failure(tmp_path: Path) -> None:
    pid_lock = tmp_path / ".tracker.pid"
    pid_lock.write_text("1234", encoding="utf-8")
    ring_buffer = _DummyRingBuffer(should_fail=True)

    _cleanup_runtime_resources(ring_buffer, pid_lock)

    assert ring_buffer.cleanup_calls == 1
    assert not pid_lock.exists()
