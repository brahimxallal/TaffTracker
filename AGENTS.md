# Project Guidelines

This repository is a low-latency vision-guided gimbal tracker with a Python multi-process pipeline and ESP32-S3 firmware. Prefer code and tests over documentation changes. Do not add new Markdown files unless the user explicitly asks for them.

## Build And Test

- Python dependencies: `pip install -r requirements-dev.txt`
- Run the tracker directly: `python -m src.main --mode camera --target human --source "0" --config config.yaml`
- `launch.bat` is the Windows interactive launcher. It prefers the project `.venv`, then an already-active virtual environment, then `python` on `PATH`. Use `python -m src.main` (or `tracker-gui` for the desktop GUI) on other platforms.
- TensorRT engines are required before first run. Export them with `python scripts/export_engines.py --target all` or a single target.
- Run tests with `python -m pytest tests/`. CI runs `python -m pytest -m "not perf" --cov=src`, plus `ruff check .`, `black --check .` (versions pinned in `requirements-lint.txt`) and a non-blocking `mypy src`.
- Run focused tests first when touching a subsystem, then broaden if needed.
- Firmware tasks already exist in VS Code: `Firmware: Build`, `Firmware: Rebuild`, `Firmware: Upload`, `Firmware: Monitor`.
- PlatformIO must use `PLATFORMIO_CORE_DIR=C:\PIO`; builds are known to fail when the core path contains spaces.

## Architecture

- The runtime pipeline is three isolated processes: `CaptureProcess` -> shared ring buffer -> `InferenceProcess` -> result queue -> `OutputProcess`. An optional fourth process, `PhoneImuProcess` (`phone_imu.enabled`, off by default), streams the phone's motion sensors.
- The main process owns display work; keep `cv2.imshow()` and UI event handling there on Windows.
- Shared memory is performance-critical. The custom ring buffer (`runtime.ring_buffer_slots`, default 8) uses generation counters and zero-copy frame sharing; avoid adding extra copies on the hot path. The result queue holds only the newest `TrackingMessage` (the publisher drops stale ones), so the output never acts on old data.
- Inference flow is: frame read -> TensorRT YOLO -> postprocess -> BoTSORT -> Kalman -> centroid selection -> 1 Euro filtering -> optional laser detection -> `TrackingMessage`. `DetectorScheduler` (`tracking.inference_schedule`) can skip YOLO on planned Kalman-only prediction frames while the lock is healthy.
- Output flow is intentionally fire-and-forget over serial or UDP. Do not add retransmission or blocking acknowledgements in the control path.
- Firmware runs a 200 Hz control loop on ESP32-S3 and applies its own watchdog, smoothing, and home behavior.

## Conventions

- Configuration is modeled with frozen dataclasses in `src/config.py`, loaded from `config.yaml`, then overridden by CLI arguments.
- Camera geometry uses a pinhole model with `atan2`-based pixel-to-angle conversion. Do not replace it with homography-style mapping for pan/tilt aiming.
- The camera is assumed to be effectively on-axis with the gimbal pivot. Parallax compensation and pose-based depth estimation have been removed; angles come straight from the pinhole `pixel_to_angle` mapping.
- The calibration path is: camera intrinsics derived from `camera.fov`, mount offset from `config.yaml`, servo mechanical limits and laser boresight from `calibration_data/servo_limits.json` or firmware config.
- Keep latency-sensitive code simple: avoid blocking I/O, avoid unnecessary allocations, and preserve the existing multiprocessing separation.
- Tests are written to run without CUDA or TensorRT hardware. `tests/conftest.py` stubs hardware-only modules during collection.
- Keep changes narrow. Do not collapse process boundaries or remove fail-safe behavior unless the user explicitly asks for that architectural change.

## Desktop GUI Rules

- TaffTracker uses PySide6. Apply PyQt6 UI-development guidance through the PySide6 equivalents: `Signal`, `Slot`, `QObject`, `QThread`, `QThreadPool`, QSS, and Qt layout managers.
- Keep strict separation between UI widgets, view-model/config transforms, runtime orchestration, and hot tracking logic. UI widgets should emit commands or call narrow controller/session APIs, not own business logic.
- Never block the Qt UI thread with camera, TensorRT, serial/UDP, subprocess, filesystem-heavy, or model-loading work. Use the existing multiprocessing runtime, `QThread`, `QThreadPool`, timers, or bounded polling.
- Apply theme rules centrally at the `QApplication` or main-window level. Avoid scattered per-widget inline styling except object names needed by QSS.
- Do not use absolute widget coordinates. Use `QVBoxLayout`, `QHBoxLayout`, `QGridLayout`, `QFormLayout`, stretch factors, size policies, and scroll areas for responsive desktop layouts.
- For cross-thread UI updates, emit lightweight metadata/signals only. Never emit `QPixmap` from a worker thread, and never send full video frames through Qt signals or queues.
- Preserve high-DPI/cross-platform behavior: avoid hardcoded text clipping, verify fonts and controls on Windows first, and use layouts/size policies instead of fixed pixel placement.
- For custom widgets, keep painting inside `paintEvent`, avoid expensive allocations per paint, and request updates with Qt mechanisms rather than tight loops.
- Add focused tests around view-model/config transforms and runtime-session boundaries. GUI smoke checks should verify that windows launch, controls are visible, and software-only demo mode works without CUDA/camera/ESP32.

## Important Pitfalls

- The checked-in TensorRT engine may exist for one model while another target still needs export. Check the requested target before assuming runtime is ready.
- `launch.bat` must keep CRLF line endings (`.gitattributes` enforces it, `tests/test_launch_bat.py` checks it). Edit it byte-safely.
- Safety-critical firmware paths have host-compiled regression tests (`tests/test_firmware_host_harness.py`, sources in `firmware/esp32s3_gimbal/host_tests/`): the USB frame parser must never overrun its buffer, and the relay must release when the host goes silent. Keep them passing when touching `comm_task.c` or `control_task.c`; they skip themselves when no C compiler is present.
- `relay.auto_fire_enabled` fires the relay on every new lock. Keep it `false` in the committed `config.yaml`.
- `sdkconfig.esp32s3` is the authoritative ESP-IDF config for firmware behavior. Do not assume `sdkconfig.defaults` alone controls the build.
- For ESP-IDF 5.5.3, `CONFIG_ESP_TIMER_SHOW_EXPERIMENTAL=y` must be enabled before CPU affinity settings for the timer take effect.
- Ultralytics TensorRT exports in this repo include a metadata prefix before the serialized plan; custom engine loaders must account for that.

## Key Files

- `src/main.py`: process orchestration and the main display/hotkey loop only
- `src/cli.py`: argparse, YAML + CLI config assembly, and `validate_environment`
- `src/process_supervisor.py`: error-queue drain, dead-process detection, graceful stop, cleanup
- `src/ui/overlays.py`: pure OpenCV renderers for the help panel and laser-cal HUD
- `src/ui/hotkeys.py`: keyboard-event dispatch rules for the main display loop
- `src/capture/process.py`: frame capture and ring-buffer writes
- `src/capture/phone_yuv.py`: TaffCam raw-YUV TCP capture source (background reader thread, ADB reverse)
- `src/capture/taffcam_adb.py`: ADB helpers that start the TaffCam app and its sensor service
- `src/inference/process.py`: TensorRT inference, tracking, smoothing, and pinhole pixel→angle generation
- `src/inference/scheduler.py`: `DetectorScheduler` — decides which frames run YOLO and which are Kalman-only predictions
- `src/inference/engine_validation.py`: early check that the selected TensorRT runtime can deserialize an engine
- `src/inference/telemetry.py`: `format_profiler_summary` + `write_profiler_summary` — per-run profiler log line and shutdown-time CSV/JSON metrics dump
- `src/inference/postprocess.py`: YOLO parsing and centroid selection logic
- `src/output/process.py`: protocol encoding, comms, and output fail-safe behavior
- `src/output/diagnostics.py`: `draw_diagnostics` + `get_transport_status` HUD helpers for the output process
- `src/output/sender_factory.py`: `create_sender(comm_config)` — chooses Serial/UDP/Auto transport
- `src/output/telemetry.py`: `write_metrics_summary` — shutdown-time JSON snapshot of packet + display stats
- `src/output/command_smoother.py`: `CommandSlewLimiter` + `BacklashCompensator` — output-side slew cap and direction-reversal pulse
- `src/output/relock_diagnostics.py`: logs micro-loss/relock events for tuning
- `src/output/manual_control.py`: `ManualVelocityTracker` + `boost_manual_velocity` + `build_manual_packet` + `rewrite_packet_sequence` — manual-mode velocity differentiator, response-floor helper, v2 packet builder, and post-lock sequence/CRC rewriter
- `src/shared/ring_buffer.py`: shared-memory transport implementation
- `src/shared/protocol.py`: binary packet format used by firmware
- `src/sensors/`: `PhoneImuProcess`, the TaffCam sensor TCP protocol and decoder
- `src/shared/logging_config.py`: per-process logging setup shared through `TAFFTRACKER_LOG_*` environment variables
- `src/shared/windows_timer.py`: `WindowsTimerResolution` context manager (1 ms scheduler tick on Windows)
- `src/shared/preflight.py`: `FrameHealthMonitor` AE/AWB/AF-drift detector (consumed by the inference process)
- `src/calibration/camera_model.py`: camera geometry and pixel-to-angle conversion
- `config.yaml`: runtime tuning and hardware alignment source of truth
- `scripts/export_engines.py`: required TensorRT export step
- `scripts/calibrate.py`: gimbal center / laser boresight calibration (saves `calibration_data/servo_limits.json`, pushes it to the ESP32 over USB)
- `tests/conftest.py`: test-time hardware stubs and common fixtures
- `firmware/esp32s3_gimbal/host_tests/`: ESP-IDF stubs and C harnesses that compile the real firmware sources on the host (ASan/UBSan)

## Practical Defaults For Agents

- When debugging Python logic, start with targeted `pytest` runs.
- When debugging firmware behavior, use the existing VS Code tasks instead of ad hoc PlatformIO commands.
- When touching aiming math, inspect both `src/inference/process.py` and `src/calibration/camera_model.py` together.
- When touching tracking behavior, verify interactions across postprocess, ByteTrack, Kalman, and output fail-safe paths.
