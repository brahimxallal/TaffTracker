from __future__ import annotations

import logging
import multiprocessing as mp
import os
import signal
import time
from pathlib import Path

from src.capture.process import CaptureProcess
from src.cli import build_config, parse_args, validate_environment
from src.inference.process import InferenceProcess
from src.output.process import OutputProcess
from src.process_supervisor import (
    all_processes_stopped,
    check_runtime_failures,
    cleanup_runtime_resources,
    drain_error_reports,
    log_error_reports,
    stop_processes,
)
from src.shared.display_buffer import SharedDisplayBuffer
from src.shared.ring_buffer import SharedRingBuffer
from src.shared.types import ProcessErrorReport
from src.ui.hotkeys import is_quit_hotkey
from src.ui.overlays import draw_help_overlay, draw_laser_cal_hud

# Re-export for tests/backwards compatibility: ``from src.main import validate_environment``
__all__ = [
    "build_config",
    "main",
    "parse_args",
    "validate_environment",
]

LOGGER = logging.getLogger("main")

MANUAL_FINE_SPEED_DPS = 120.0
MANUAL_COARSE_SPEED_DPS = 300.0
MANUAL_ACCEL_TIME = 0.05  # seconds to reach full speed
_MAX_PROCESS_RESTARTS = 3


def _setup_signal_handlers(shutdown_event: mp.synchronize.Event) -> None:
    """Setup signal handlers for graceful shutdown."""

    def signal_handler(signum, frame):
        LOGGER.info("Received signal %s, initiating shutdown...", signum)
        shutdown_event.set()

    signal.signal(signal.SIGINT, signal_handler)
    signal.signal(signal.SIGTERM, signal_handler)


def _log_system_info() -> None:
    """Log system information for debugging and monitoring."""
    try:
        import platform

        import torch

        LOGGER.info("System: %s %s (%s)", platform.system(), platform.release(), platform.machine())
        LOGGER.info("Python: %s", platform.python_version())
        LOGGER.info("PyTorch: %s", torch.__version__)
        LOGGER.info("CUDA Available: %s", torch.cuda.is_available())
        if torch.cuda.is_available():
            LOGGER.info("CUDA Device: %s", torch.cuda.get_device_name(0))
            LOGGER.info(
                "CUDA Memory: %.1f GB", torch.cuda.get_device_properties(0).total_memory / 1e9
            )
    except ImportError:
        LOGGER.debug("Optional dependencies not available for system info logging")
    except Exception as e:
        LOGGER.warning("Failed to log system info: %s", e)


def _draw_help_overlay(frame) -> None:
    """Thin alias preserved for in-file callers (see src.ui.overlays)."""
    draw_help_overlay(frame)


def _draw_laser_cal_hud(frame, text: str) -> None:
    """Thin alias preserved for in-file callers (see src.ui.overlays)."""
    draw_laser_cal_hud(frame, text)


def _is_quit_hotkey(key_code: int, no_quit: bool) -> bool:
    """Thin alias preserved for in-file callers (see src.ui.hotkeys)."""
    return is_quit_hotkey(key_code, no_quit)


# --- Back-compat underscore aliases (see src.process_supervisor) -----------
# Existing tests import these by private name from ``src.main``; keep the
# thin wrappers so refactoring does not cascade into the test suite.
def _drain_error_reports(error_queue: mp.Queue) -> list[ProcessErrorReport]:
    return drain_error_reports(error_queue)


def _log_error_reports(reports: list[ProcessErrorReport]) -> None:
    log_error_reports(reports)


def _all_processes_stopped(processes: list[mp.Process]) -> bool:
    return all_processes_stopped(processes)


def _check_runtime_failures(
    processes: list[mp.Process],
    error_queue: mp.Queue,
) -> tuple[list[ProcessErrorReport], list[mp.Process]]:
    return check_runtime_failures(processes, error_queue)


def _stop_processes(
    processes: list[mp.Process],
    join_timeout: float = 5.0,
    terminate_timeout: float = 2.0,
) -> list[str]:
    return stop_processes(processes, join_timeout=join_timeout, terminate_timeout=terminate_timeout)


def _cleanup_runtime_resources(
    ring_buffer: SharedRingBuffer,
    pid_lock_path: Path = Path("logs/.tracker.pid"),
    *,
    display_buffer: SharedDisplayBuffer | None = None,
) -> None:
    cleanup_runtime_resources(ring_buffer, pid_lock_path, display_buffer=display_buffer)


def _kill_previous_instance() -> None:
    """Kill any previous tracker instance using a PID lock file."""
    lock_path = Path("logs/.tracker.pid")
    lock_path.parent.mkdir(parents=True, exist_ok=True)
    if lock_path.exists():
        try:
            old_pid = int(lock_path.read_text().strip())
            if old_pid != os.getpid():
                os.kill(old_pid, signal.SIGTERM)
                import time

                time.sleep(0.5)
                try:
                    os.kill(old_pid, signal.SIGTERM)
                except OSError:
                    pass  # already dead
                LOGGER.info("Killed previous tracker instance (PID %d)", old_pid)
        except (ValueError, OSError):
            pass  # stale or invalid pid file
    lock_path.write_text(str(os.getpid()))


def main() -> None:
    mp.freeze_support()
    args = parse_args()
    config = build_config(args)
    log_level = getattr(logging, config.flags.log_level, logging.INFO)
    log_handlers: list[logging.Handler] = [logging.StreamHandler()]
    if config.flags.log_file is not None:
        config.flags.log_file.parent.mkdir(parents=True, exist_ok=True)
        log_handlers.append(logging.FileHandler(str(config.flags.log_file)))
    logging.basicConfig(
        level=log_level,
        format="%(asctime)s %(processName)s %(levelname)s %(message)s",
        handlers=log_handlers,
    )
    validate_environment(config)
    _kill_previous_instance()
    config.paths.resolved_log_dir().mkdir(parents=True, exist_ok=True)

    # Log system information for debugging and monitoring
    _log_system_info()

    # Multiprocessing isolates CPU-bound tracking from capture/output
    # so the GIL does not serialize the pipeline.
    ring_buffer, write_index = SharedRingBuffer.create(
        (config.camera.height, config.camera.width, 3),
        num_slots=3,
    )
    result_queue: mp.Queue = mp.Queue(maxsize=16)
    error_queue: mp.Queue = mp.Queue(maxsize=8)
    capture_done_event = mp.Event()
    shutdown_event = mp.Event()

    # Setup signal handlers for graceful shutdown
    _setup_signal_handlers(shutdown_event)

    # Display via shared memory (zero-copy, no pickle overhead)
    # (cv2.imshow only works reliably in main process on Windows)
    display_buffer: SharedDisplayBuffer | None = None
    if not config.flags.headless:
        display_buffer = SharedDisplayBuffer.create(
            (config.camera.height, config.camera.width, 3),
        )

    relay_flag = mp.Value("b", 0)
    laser_enabled = mp.Value("b", 1)
    relay_pulse_s = config.relay.pulse_ms / 1000.0
    relay_off_time: float = 0.0
    show_help: bool = False
    relock_event = mp.Event()
    cycle_target_event = mp.Event()

    # Manual gimbal control shared state
    manual_mode = mp.Value("b", 0)  # 0=auto, 1=manual
    manual_pan = mp.Value("d", 0.0)  # absolute pan angle (degrees)
    manual_tilt = mp.Value("d", 0.0)  # absolute tilt angle (degrees)
    manual_last_time: float = time.perf_counter()
    _manual_vel_pan: float = 0.0
    _manual_vel_tilt: float = 0.0

    # Laser boresight offset (live-updated by interactive calibrator, read by OutputProcess)
    laser_boresight_pan = mp.Value("d", float(config.laser_boresight.pan_offset_deg))
    laser_boresight_tilt = mp.Value("d", float(config.laser_boresight.tilt_offset_deg))

    from src.calibration.laser_boresight import LaserBoresight
    from src.calibration.laser_calibrator import LaserCalibrator

    laser_cal_save_path = (
        config.paths.resolve_path(config.paths.calibration_dir) / "servo_limits.json"
    )
    laser_calibrator = LaserCalibrator(
        save_path=laser_cal_save_path,
        initial=LaserBoresight(
            pan_offset_deg=float(config.laser_boresight.pan_offset_deg),
            tilt_offset_deg=float(config.laser_boresight.tilt_offset_deg),
        ),
    )

    def _make_capture():
        return CaptureProcess(
            ring_buffer.layout,
            write_index,
            config.source,
            config.camera,
            capture_done_event,
            shutdown_event,
            error_queue,
        )

    def _make_inference():
        return InferenceProcess(
            ring_buffer.layout,
            write_index,
            result_queue,
            capture_done_event,
            shutdown_event,
            error_queue,
            config.mode,
            config.target,
            config.camera,
            config.tracking,
            config.models,
            config.paths,
            config.laser,
            preflight_config=config.preflight,
            profile=config.flags.profile,
            relock_event=relock_event,
            cycle_target_event=cycle_target_event,
            command_pan=manual_pan,
            command_tilt=manual_tilt,
        )

    def _make_output():
        return OutputProcess(
            ring_buffer.layout,
            write_index,
            result_queue,
            shutdown_event,
            error_queue,
            config.mode,
            config.camera,
            config.comms,
            config.tracking,
            config.flags,
            config.gimbal,
            servo_control_config=config.servo_control,
            display_buffer_layout=display_buffer.layout if display_buffer is not None else None,
            relay_flag=relay_flag,
            laser_enabled=laser_enabled,
            manual_mode=manual_mode,
            manual_pan=manual_pan,
            manual_tilt=manual_tilt,
            laser_boresight_pan=laser_boresight_pan,
            laser_boresight_tilt=laser_boresight_tilt,
        )

    capture = _make_capture()
    inference = _make_inference()
    output = _make_output()

    _process_factories: dict[str, callable] = {
        "CaptureProcess": _make_capture,
        "InferenceProcess": _make_inference,
        "OutputProcess": _make_output,
    }
    _restart_counts: dict[str, int] = {name: 0 for name in _process_factories}

    processes = [capture, inference, output]
    try:
        for process in processes:
            process.start()

        # Main monitoring loop (includes display when not headless)
        while not shutdown_event.is_set():
            reports, dead_processes = _check_runtime_failures(processes, error_queue)
            if dead_processes:
                restarted_any = False
                for dead in dead_processes:
                    name = dead.name
                    LOGGER.error("%s exited unexpectedly with code %s", name, dead.exitcode)
                    factory = _process_factories.get(name)
                    if factory and _restart_counts[name] < _MAX_PROCESS_RESTARTS:
                        _restart_counts[name] += 1
                        LOGGER.warning(
                            "Restarting %s (attempt %d/%d)",
                            name,
                            _restart_counts[name],
                            _MAX_PROCESS_RESTARTS,
                        )
                        new_proc = factory()
                        idx = processes.index(dead)
                        processes[idx] = new_proc
                        new_proc.start()
                        restarted_any = True
                    else:
                        LOGGER.error(
                            "%s exhausted restart attempts (%d) — shutting down",
                            name,
                            _MAX_PROCESS_RESTARTS,
                        )
                        _log_error_reports(reports)
                        shutdown_event.set()
                        break
                if restarted_any and not shutdown_event.is_set():
                    _log_error_reports(reports)
                    continue
            if reports and not dead_processes:
                _log_error_reports(reports)
                shutdown_event.set()
                break

            # Check if all processes have completed normally
            if _all_processes_stopped(processes):
                break

            # Display frames in main process (cv2.imshow requires main process on Windows)
            if display_buffer is not None:
                frame = display_buffer.read()

                import cv2

                if frame is not None:
                    if show_help:
                        _draw_help_overlay(frame)
                    if laser_calibrator.active:
                        _draw_laser_cal_hud(frame, laser_calibrator.hud_line())
                    cv2.imshow("Vision Gimbal Tracker", frame)

                key = cv2.waitKeyEx(1)
                key_low = key & 0xFF  # ASCII portion for letter keys

                # Laser boresight calibration: when active, it consumes arrow
                # keys + Enter + Esc so they do not leak into manual-jog.
                if laser_calibrator.active:
                    if laser_calibrator.handle_key(key):
                        off = laser_calibrator.current_offset()
                        laser_boresight_pan.value = off.pan_offset_deg
                        laser_boresight_tilt.value = off.tilt_offset_deg
                        key = -1
                        key_low = 0

                if _is_quit_hotkey(key_low, args.no_quit) and not laser_calibrator.active:
                    LOGGER.info("Quit requested via keyboard")
                    shutdown_event.set()
                    break
                if key_low == ord("k") and not laser_calibrator.active:
                    laser_calibrator.start()
                    # Seed shared offset so the first frame in cal mode reflects the saved value
                    off = laser_calibrator.current_offset()
                    laser_boresight_pan.value = off.pan_offset_deg
                    laser_boresight_tilt.value = off.tilt_offset_deg
                if key_low == ord("o"):
                    relay_flag.value = 1
                    relay_off_time = time.perf_counter() + relay_pulse_s
                    LOGGER.info("Relay pulse %dms", config.relay.pulse_ms)
                if key_low == ord("h"):
                    show_help = not show_help
                    LOGGER.info("Help overlay %s", "ON" if show_help else "OFF")
                if key_low == ord("l"):
                    relock_event.set()
                    LOGGER.info("Relock requested — releasing target lock")
                if key_low == ord("p"):
                    laser_enabled.value = 0 if laser_enabled.value else 1
                    LOGGER.info(
                        "Laser output %s",
                        "ENABLED" if laser_enabled.value else "DISABLED",
                    )
                if key_low == 9:  # Tab key
                    cycle_target_event.set()
                    LOGGER.info("Cycle target requested")
                if key_low == ord("m"):
                    if manual_mode.value:
                        # Exiting manual → back to auto: snap to nearest target
                        manual_mode.value = 0
                        relock_event.set()
                        LOGGER.info("Switched to AUTO tracking (relock)")
                    else:
                        # Entering manual: seamless handoff from current position
                        manual_mode.value = 1
                        LOGGER.info(
                            "Switched to MANUAL (pan=%.1f tilt=%.1f)",
                            manual_pan.value,
                            manual_tilt.value,
                        )
                    manual_last_time = time.perf_counter()

                # Manual gimbal movement — diagonal + acceleration ramping
                if manual_mode.value:
                    now = time.perf_counter()
                    dt = min(now - manual_last_time, 0.1)
                    manual_last_time = now
                    pan_lim = config.gimbal.pan_limit_deg
                    tilt_lim = config.gimbal.tilt_limit_deg
                    speed_fine = MANUAL_FINE_SPEED_DPS
                    speed_coarse = MANUAL_COARSE_SPEED_DPS

                    # Determine requested pan/tilt velocity from keys
                    req_pan = 0.0
                    req_tilt = 0.0
                    if key_low == ord("q"):
                        req_pan = -speed_fine
                    elif key_low == ord("d"):
                        req_pan = speed_fine
                    elif key_low == ord("z"):
                        req_tilt = -speed_fine
                    elif key_low == ord("s"):
                        req_tilt = speed_fine
                    # Arrow keys (can combine with ZQSD for diagonal)
                    if key == 0x250000:
                        req_pan = -speed_coarse
                    elif key == 0x270000:
                        req_pan = speed_coarse
                    if key == 0x260000:
                        req_tilt = -speed_coarse
                    elif key == 0x280000:
                        req_tilt = speed_coarse

                    # Smooth acceleration ramp
                    ramp_rate = dt / MANUAL_ACCEL_TIME if MANUAL_ACCEL_TIME > 0 else 1.0
                    if req_pan != 0.0:
                        _manual_vel_pan += (req_pan - _manual_vel_pan) * min(1.0, ramp_rate)
                    else:
                        _manual_vel_pan *= max(0.0, 1.0 - ramp_rate * 3)  # fast decel
                    if req_tilt != 0.0:
                        _manual_vel_tilt += (req_tilt - _manual_vel_tilt) * min(1.0, ramp_rate)
                    else:
                        _manual_vel_tilt *= max(0.0, 1.0 - ramp_rate * 3)

                    moved = abs(_manual_vel_pan) > 0.05 or abs(_manual_vel_tilt) > 0.05
                    if moved:
                        manual_pan.value = max(
                            -pan_lim, min(pan_lim, manual_pan.value + _manual_vel_pan * dt)
                        )
                        manual_tilt.value = max(
                            -tilt_lim, min(tilt_lim, manual_tilt.value + _manual_vel_tilt * dt)
                        )
                    else:
                        _manual_vel_pan = 0.0
                        _manual_vel_tilt = 0.0
                        manual_last_time = now
                else:
                    manual_last_time = time.perf_counter()

                # Auto-off relay after pulse duration
                if relay_flag.value and time.perf_counter() >= relay_off_time:
                    relay_flag.value = 0
                    LOGGER.info("Relay OFF")
            else:
                shutdown_event.wait(0.05)

    except KeyboardInterrupt:
        LOGGER.info("Shutdown requested by user")
        shutdown_event.set()
    except Exception as e:
        LOGGER.exception("Unexpected error in main process: %s", e)
        shutdown_event.set()
    finally:
        # Ensure shutdown event is set
        shutdown_event.set()

        _log_error_reports(_drain_error_reports(error_queue))
        _stop_processes(processes)
        _cleanup_runtime_resources(ring_buffer, display_buffer=display_buffer)


if __name__ == "__main__":
    main()
