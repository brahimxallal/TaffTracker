# Project Guidelines

This repository is a low-latency vision-guided gimbal tracker with a Python multi-process pipeline and ESP32-S3 firmware. Prefer code and tests over documentation changes. Do not add new Markdown files unless the user explicitly asks for them.

## Build And Test

- Python dependencies: `pip install -r requirements-dev.txt`
- Run the tracker directly: `python -m src.main --mode camera --target human --source "0" --config config.yaml`
- Use `launch.bat` only when you specifically want the interactive launcher flow; it hardcodes a user-local venv path.
- TensorRT engines are required before first run. Export them with `python scripts/export_engines.py --target all` or a single target.
- Run tests with `python -m pytest tests/`
- Run focused tests first when touching a subsystem, then broaden if needed.
- Firmware tasks already exist in VS Code: `Firmware: Build`, `Firmware: Rebuild`, `Firmware: Upload`, `Firmware: Monitor`.
- PlatformIO must use `PLATFORMIO_CORE_DIR=C:\PIO`; builds are known to fail when the core path contains spaces.

## Architecture

- The runtime pipeline is three isolated processes: `CaptureProcess` -> shared ring buffer -> `InferenceProcess` -> result queue -> `OutputProcess`.
- The main process owns display work; keep `cv2.imshow()` and UI event handling there on Windows.
- Shared memory is performance-critical. The custom ring buffer uses generation counters and zero-copy frame sharing; avoid adding extra copies on the hot path.
- Inference flow is: frame read -> undistort -> TensorRT YOLO -> postprocess -> ByteTrack -> Kalman -> centroid selection -> 1 Euro filtering -> optional optical flow / laser detection -> `TrackingMessage`.
- Output flow is intentionally fire-and-forget over serial or UDP. Do not add retransmission or blocking acknowledgements in the control path.
- Firmware runs a 200 Hz control loop on ESP32-S3 and applies its own watchdog, smoothing, and home behavior.

## Conventions

- Configuration is modeled with frozen dataclasses in `src/config.py`, loaded from `config.yaml`, then overridden by CLI arguments.
- Camera geometry uses a pinhole model with `atan2`-based pixel-to-angle conversion. Do not replace it with homography-style mapping for pan/tilt aiming.
- If the camera is not co-located with the gimbal pivot, use `mount_offset` in `config.yaml`. Parallax compensation and pose-based depth estimation are part of the pipeline.
- The calibration path is: camera intrinsics from `camera.fov` or `calibration_data/intrinsics.npz`, mount offset from `config.yaml`, servo mechanical limits from calibration data or firmware config.
- Keep latency-sensitive code simple: avoid blocking I/O, avoid unnecessary allocations, and preserve the existing multiprocessing separation.
- Tests are written to run without CUDA or TensorRT hardware. `tests/conftest.py` stubs hardware-only modules during collection.
- Keep changes narrow. Do not collapse process boundaries or remove fail-safe behavior unless the user explicitly asks for that architectural change.

## Important Pitfalls

- The checked-in TensorRT engine may exist for one model while another target still needs export. Check the requested target before assuming runtime is ready.
- `launch.bat` validates a specific local Python path; agents should not treat that path as portable project configuration.
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
- `src/capture/preflight.py`: thin re-export shim; real `FrameHealthMonitor` lives in `src/shared/preflight.py`
- `src/inference/process.py`: TensorRT inference, tracking, smoothing, and parallax-aware angle generation
- `src/inference/postprocess.py`: YOLO parsing and centroid selection logic
- `src/output/process.py`: protocol encoding, comms, and output fail-safe behavior
- `src/output/diagnostics.py`: `draw_diagnostics` + `get_transport_status` HUD helpers for the output process
- `src/output/sender_factory.py`: `create_sender(comm_config)` — chooses Serial/UDP/Auto transport
- `src/output/telemetry.py`: `write_metrics_summary` — shutdown-time JSON snapshot of packet + display stats
- `src/output/manual_control.py`: `ManualVelocityTracker` + `boost_manual_velocity` + `build_manual_packet` + `rewrite_packet_sequence` — manual-mode velocity differentiator, response-floor helper, v2 packet builder, and post-lock sequence/CRC rewriter
- `src/shared/ring_buffer.py`: shared-memory transport implementation
- `src/shared/protocol.py`: binary packet format used by firmware
- `src/shared/preflight.py`: `FrameHealthMonitor` AE/AWB/AF-drift detector (consumed by the inference process)
- `src/calibration/camera_model.py`: camera geometry and pixel-to-angle conversion
- `src/calibration/depth_estimator.py`: pose-span depth estimation used by parallax correction
- `config.yaml`: runtime tuning and hardware alignment source of truth
- `scripts/export_engines.py`: required TensorRT export step
- `scripts/calibrate.py`: unified servo limits + center + camera intrinsics + mount offset calibration
- `tests/conftest.py`: test-time hardware stubs and common fixtures

## Practical Defaults For Agents

- When debugging Python logic, start with targeted `pytest` runs.
- When debugging firmware behavior, use the existing VS Code tasks instead of ad hoc PlatformIO commands.
- When touching aiming math, inspect both `src/inference/process.py` and `src/calibration/camera_model.py` together.
- When touching tracking behavior, verify interactions across postprocess, ByteTrack, Kalman, and output fail-safe paths.