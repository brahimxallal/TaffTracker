from __future__ import annotations

import logging
import multiprocessing as mp
import time
from collections.abc import Callable
from dataclasses import dataclass
from pathlib import Path
from queue import Empty
from typing import Any

import numpy as np

from src.capture.process import CaptureProcess
from src.cli import validate_environment
from src.config import PipelineConfig
from src.gui.view_model import TuningSettings, apply_tuning_settings
from src.inference.process import InferenceProcess
from src.output.process import OutputProcess
from src.process_supervisor import (
    all_processes_stopped,
    check_runtime_failures,
    drain_error_reports,
    log_error_reports,
    stop_processes,
)
from src.shared.display_buffer import SharedDisplayBuffer
from src.shared.ring_buffer import SharedRingBuffer
from src.shared.runtime_control import OutputRuntimeTuning
from src.shared.telemetry import RuntimeTelemetry
from src.shared.types import ProcessErrorReport

LOGGER = logging.getLogger("gui.runtime")

_MAX_PROCESS_RESTARTS = 3


@dataclass(frozen=True, slots=True)
class ProcessState:
    name: str
    pid: int | None
    alive: bool
    exitcode: int | None
    restarts: int


@dataclass(frozen=True, slots=True)
class RuntimeSnapshot:
    running: bool
    stopping: bool
    display_available: bool
    process_states: tuple[ProcessState, ...]
    error_summaries: tuple[str, ...]
    telemetry: RuntimeTelemetry | None
    laser_enabled: bool
    manual_mode: bool
    relay_active: bool
    runtime_control_version: int
    runtime_control_ack_version: int


class RuntimeSession:
    """GUI-owned wrapper around the existing tracker process graph.

    The session keeps the current process boundaries intact. It only owns
    lifecycle, display-buffer polling, and shared control flags that already
    exist in the script runtime.
    """

    def __init__(self, config: PipelineConfig, *, validate_on_start: bool = True) -> None:
        self._config = config
        self._validate_on_start = validate_on_start
        self._ctx = mp.get_context()
        self._ring_buffer: SharedRingBuffer | None = None
        self._write_index: Any | None = None
        self._result_queue: mp.Queue | None = None
        self._error_queue: mp.Queue | None = None
        self._telemetry_queue: mp.Queue | None = None
        self._control_queue: mp.Queue | None = None
        self._control_ack_version: Any | None = None
        self._capture_done_event: Any | None = None
        self._shutdown_event: Any | None = None
        self._display_buffer: SharedDisplayBuffer | None = None
        self._relay_flag: Any | None = None
        self._laser_enabled: Any | None = None
        self._manual_mode: Any | None = None
        self._manual_pan: Any | None = None
        self._manual_tilt: Any | None = None
        self._laser_boresight_pan: Any | None = None
        self._laser_boresight_tilt: Any | None = None
        self._relock_event: Any | None = None
        self._cycle_target_event: Any | None = None
        self._processes: list[mp.Process] = []
        self._process_factories: dict[str, Callable[[], mp.Process]] = {}
        self._restart_counts: dict[str, int] = {}
        self._last_errors: list[str] = []
        self._latest_telemetry: RuntimeTelemetry | None = None
        self._relay_off_time: float = 0.0
        self._runtime_control_version = 0
        self._running = False
        self._stopping = False

    @property
    def is_running(self) -> bool:
        return self._running

    def start(self) -> RuntimeSnapshot:
        if self._running:
            return self.snapshot()

        if self._validate_on_start:
            validate_environment(self._config)
        self._config.paths.resolved_log_dir().mkdir(parents=True, exist_ok=True)
        self._allocate_runtime_resources()
        self._build_process_factories()
        self._processes = [
            self._process_factories["CaptureProcess"](),
            self._process_factories["InferenceProcess"](),
            self._process_factories["OutputProcess"](),
        ]

        try:
            for process in self._processes:
                process.start()
        except BaseException:
            self._shutdown_event_set()
            stop_processes(self._processes, join_timeout=1.0, terminate_timeout=0.5)
            self._cleanup_resources()
            raise

        self._running = True
        self._stopping = False
        return self.snapshot()

    def stop(self, *, join_timeout: float = 2.0, terminate_timeout: float = 1.0) -> RuntimeSnapshot:
        if not self._running and self._ring_buffer is None:
            return self.snapshot()
        self._stopping = True
        self._shutdown_event_set()
        reports = self._drain_errors()
        if reports:
            log_error_reports(reports)
        stop_processes(
            self._processes,
            join_timeout=join_timeout,
            terminate_timeout=terminate_timeout,
        )
        self._cleanup_resources()
        self._running = False
        self._stopping = False
        return self.snapshot()

    def emergency_stop(self) -> RuntimeSnapshot:
        self.set_laser_enabled(False)
        if self._relay_flag is not None:
            self._relay_flag.value = 0
        if self._manual_mode is not None:
            self._manual_mode.value = 0
        if self._manual_pan is not None:
            self._manual_pan.value = 0.0
        if self._manual_tilt is not None:
            self._manual_tilt.value = 0.0
        self._shutdown_event_set()
        self._stopping = self._running
        return self.snapshot()

    def poll(self) -> RuntimeSnapshot:
        self._expire_relay_pulse()
        self._drain_telemetry()
        if not self._running:
            return self.snapshot()

        reports, dead_processes = check_runtime_failures(self._processes, self._error_queue)
        self._record_errors(reports)

        if dead_processes:
            restarted_any = False
            for dead in dead_processes:
                factory = self._process_factories.get(dead.name)
                if factory is not None and self._restart_counts[dead.name] < _MAX_PROCESS_RESTARTS:
                    self._restart_counts[dead.name] += 1
                    LOGGER.warning(
                        "Restarting %s (attempt %d/%d)",
                        dead.name,
                        self._restart_counts[dead.name],
                        _MAX_PROCESS_RESTARTS,
                    )
                    new_proc = factory()
                    index = self._processes.index(dead)
                    self._processes[index] = new_proc
                    new_proc.start()
                    restarted_any = True
                else:
                    self._last_errors.append(
                        f"{dead.name} exited with {dead.exitcode}; restart limit reached"
                    )
                    self._shutdown_event_set()
                    self._stopping = True
            if restarted_any:
                return self.snapshot()

        if reports and not dead_processes:
            self._shutdown_event_set()
            self._stopping = True

        if all_processes_stopped(self._processes):
            self._record_errors(self._drain_errors())
            self._cleanup_resources()
            self._running = False
            self._stopping = False

        return self.snapshot()

    def snapshot(self) -> RuntimeSnapshot:
        return RuntimeSnapshot(
            running=self._running,
            stopping=self._stopping,
            display_available=self._display_buffer is not None,
            process_states=tuple(self._process_state(process) for process in self._processes),
            error_summaries=tuple(self._last_errors[-10:]),
            telemetry=self._latest_telemetry,
            laser_enabled=self._shared_bool(self._laser_enabled),
            manual_mode=self._shared_bool(self._manual_mode),
            relay_active=self._shared_bool(self._relay_flag),
            runtime_control_version=self._runtime_control_version,
            runtime_control_ack_version=self._shared_int(self._control_ack_version),
        )

    def read_display_frame(self) -> np.ndarray | None:
        if self._display_buffer is None:
            return None
        return self._display_buffer.read()

    def request_relock(self) -> None:
        if self._relock_event is not None:
            self._relock_event.set()

    def request_cycle_target(self) -> None:
        if self._cycle_target_event is not None:
            self._cycle_target_event.set()

    def set_laser_enabled(self, enabled: bool) -> None:
        if self._laser_enabled is not None:
            self._laser_enabled.value = int(enabled)

    def toggle_laser(self) -> bool:
        enabled = not self._shared_bool(self._laser_enabled)
        self.set_laser_enabled(enabled)
        return enabled

    def set_manual_mode(self, enabled: bool) -> None:
        if self._manual_mode is not None:
            self._manual_mode.value = int(enabled)
        if not enabled:
            self.request_relock()

    def toggle_manual_mode(self) -> bool:
        enabled = not self._shared_bool(self._manual_mode)
        self.set_manual_mode(enabled)
        return enabled

    def pulse_relay(self) -> None:
        if self._relay_flag is None:
            return
        self._relay_flag.value = 1
        self._relay_off_time = time.perf_counter() + (self._config.relay.pulse_ms / 1000.0)

    def apply_runtime_tuning(self, settings: TuningSettings) -> int:
        self._config = apply_tuning_settings(self._config, settings)
        self.set_laser_enabled(settings.laser_startup_enabled)
        if self._laser_boresight_pan is not None:
            self._laser_boresight_pan.value = float(settings.boresight_pan_offset_deg)
        if self._laser_boresight_tilt is not None:
            self._laser_boresight_tilt.value = float(settings.boresight_tilt_offset_deg)
        if not self._running or self._control_queue is None:
            return self._runtime_control_version

        self._runtime_control_version += 1
        command = OutputRuntimeTuning(
            version=self._runtime_control_version,
            hold_time_s=settings.tracking_hold_time_s,
            gimbal_kp=settings.gimbal_kp,
            gimbal_ki=settings.gimbal_ki,
            gimbal_kd=settings.gimbal_kd,
            gimbal_deadband_deg=settings.gimbal_deadband_deg,
            gimbal_slew_limit_dps=settings.gimbal_slew_limit_dps,
            gimbal_kp_near=settings.gimbal_kp_near,
            gimbal_kp_far=settings.gimbal_kp_far,
            gimbal_predictive_lead_s=settings.gimbal_predictive_lead_s,
        )
        try:
            self._control_queue.put_nowait(command)
        except Exception:
            try:
                self._control_queue.get_nowait()
            except Exception:
                pass
            self._control_queue.put_nowait(command)
        return self._runtime_control_version

    def close(self) -> None:
        self.stop()

    def _allocate_runtime_resources(self) -> None:
        self._ring_buffer, self._write_index = SharedRingBuffer.create(
            (self._config.camera.height, self._config.camera.width, 3),
            num_slots=3,
            context=self._ctx,
        )
        self._result_queue = self._ctx.Queue(maxsize=16)
        self._error_queue = self._ctx.Queue(maxsize=8)
        self._telemetry_queue = self._ctx.Queue(maxsize=1)
        self._control_queue = self._ctx.Queue(maxsize=1)
        self._control_ack_version = self._ctx.Value("i", self._runtime_control_version)
        self._capture_done_event = self._ctx.Event()
        self._shutdown_event = self._ctx.Event()
        self._display_buffer = None
        if not self._config.flags.headless:
            self._display_buffer = SharedDisplayBuffer.create(
                (self._config.camera.height, self._config.camera.width, 3),
            )
        self._relay_flag = self._ctx.Value("b", 0)
        self._laser_enabled = self._ctx.Value("b", int(self._config.laser.enabled))
        self._manual_mode = self._ctx.Value("b", 0)
        self._manual_pan = self._ctx.Value("d", 0.0)
        self._manual_tilt = self._ctx.Value("d", 0.0)
        self._laser_boresight_pan = self._ctx.Value(
            "d", float(self._config.laser_boresight.pan_offset_deg)
        )
        self._laser_boresight_tilt = self._ctx.Value(
            "d", float(self._config.laser_boresight.tilt_offset_deg)
        )
        self._relock_event = self._ctx.Event()
        self._cycle_target_event = self._ctx.Event()
        self._restart_counts = {
            "CaptureProcess": 0,
            "InferenceProcess": 0,
            "OutputProcess": 0,
        }
        self._last_errors = []
        self._latest_telemetry = None

    def _build_process_factories(self) -> None:
        assert self._ring_buffer is not None
        assert self._write_index is not None
        assert self._result_queue is not None
        assert self._error_queue is not None
        assert self._telemetry_queue is not None
        assert self._control_queue is not None
        assert self._control_ack_version is not None
        assert self._capture_done_event is not None
        assert self._shutdown_event is not None

        def make_capture() -> mp.Process:
            return CaptureProcess(
                self._ring_buffer.layout,
                self._write_index,
                self._config.source,
                self._config.camera,
                self._capture_done_event,
                self._shutdown_event,
                self._error_queue,
                gpu_preprocess=self._config.flags.gpu_preprocess,
                phone_camera_config=self._config.phone_camera,
                droidcam_config=self._config.droidcam,
            )

        def make_inference() -> mp.Process:
            return InferenceProcess(
                self._ring_buffer.layout,
                self._write_index,
                self._result_queue,
                self._capture_done_event,
                self._shutdown_event,
                self._error_queue,
                self._config.mode,
                self._config.target,
                self._config.camera,
                self._config.tracking,
                self._config.models,
                self._config.paths,
                self._config.laser,
                preflight_config=self._config.preflight,
                profile=self._config.flags.profile,
                relock_event=self._relock_event,
                cycle_target_event=self._cycle_target_event,
                command_pan=self._manual_pan,
                command_tilt=self._manual_tilt,
            )

        def make_output() -> mp.Process:
            return OutputProcess(
                self._ring_buffer.layout,
                self._write_index,
                self._result_queue,
                self._shutdown_event,
                self._error_queue,
                self._config.mode,
                self._config.camera,
                self._config.comms,
                self._config.tracking,
                self._config.flags,
                self._config.gimbal,
                servo_control_config=self._config.servo_control,
                search_config=self._config.search,
                display_buffer_layout=(
                    self._display_buffer.layout if self._display_buffer is not None else None
                ),
                telemetry_queue=self._telemetry_queue,
                control_queue=self._control_queue,
                control_ack_version=self._control_ack_version,
                relay_flag=self._relay_flag,
                laser_enabled=self._laser_enabled,
                manual_mode=self._manual_mode,
                manual_pan=self._manual_pan,
                manual_tilt=self._manual_tilt,
                laser_boresight_pan=self._laser_boresight_pan,
                laser_boresight_tilt=self._laser_boresight_tilt,
            )

        self._process_factories = {
            "CaptureProcess": make_capture,
            "InferenceProcess": make_inference,
            "OutputProcess": make_output,
        }

    def _cleanup_resources(self) -> None:
        if self._display_buffer is not None:
            try:
                self._display_buffer.cleanup()
            except Exception as exc:
                LOGGER.warning("Error during display buffer cleanup: %s", exc)
        if self._ring_buffer is not None:
            try:
                self._ring_buffer.cleanup()
            except Exception as exc:
                LOGGER.warning("Error during ring buffer cleanup: %s", exc)
        self._display_buffer = None
        self._ring_buffer = None
        self._write_index = None
        self._result_queue = None
        self._error_queue = None
        self._telemetry_queue = None
        self._control_queue = None
        self._control_ack_version = None
        self._capture_done_event = None
        self._shutdown_event = None
        self._relay_flag = None
        self._laser_enabled = None
        self._manual_mode = None
        self._manual_pan = None
        self._manual_tilt = None
        self._laser_boresight_pan = None
        self._laser_boresight_tilt = None
        self._relock_event = None
        self._cycle_target_event = None
        self._process_factories = {}

    def _drain_errors(self) -> list[ProcessErrorReport]:
        if self._error_queue is None:
            return []
        return drain_error_reports(self._error_queue)

    def _drain_telemetry(self) -> None:
        if self._telemetry_queue is None:
            return
        while True:
            try:
                self._latest_telemetry = self._telemetry_queue.get_nowait()
            except Empty:
                return

    def _record_errors(self, reports: list[ProcessErrorReport]) -> None:
        for report in reports:
            self._last_errors.append(f"{report.process_name}: {report.summary}")

    def _shutdown_event_set(self) -> None:
        if self._shutdown_event is not None:
            self._shutdown_event.set()

    def _expire_relay_pulse(self) -> None:
        if (
            self._relay_flag is not None
            and self._relay_flag.value
            and self._relay_off_time > 0.0
            and time.perf_counter() >= self._relay_off_time
        ):
            self._relay_flag.value = 0
            self._relay_off_time = 0.0

    def _process_state(self, process: mp.Process) -> ProcessState:
        return ProcessState(
            name=process.name,
            pid=getattr(process, "pid", None),
            alive=process.is_alive(),
            exitcode=getattr(process, "exitcode", None),
            restarts=self._restart_counts.get(process.name, 0),
        )

    @staticmethod
    def _shared_bool(value: Any | None) -> bool:
        return bool(value is not None and value.value)

    @staticmethod
    def _shared_int(value: Any | None) -> int:
        return int(value.value) if value is not None else 0


def load_runtime_config(config_path: Path) -> PipelineConfig:
    from src.config_loader import build_config_from_yaml, load_yaml_config

    return build_config_from_yaml(load_yaml_config(config_path))
