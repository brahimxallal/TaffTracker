# Vision Gimbal Tracker

[![Tests](https://github.com/brahimxallal/TaffTracker/actions/workflows/tests.yml/badge.svg)](https://github.com/brahimxallal/TaffTracker/actions/workflows/tests.yml)
[![Lint](https://github.com/brahimxallal/TaffTracker/actions/workflows/lint.yml/badge.svg)](https://github.com/brahimxallal/TaffTracker/actions/workflows/lint.yml)

Ultra-low-latency YOLO + TensorRT pose-tracking pipeline that drives an
ESP32-S3 pan/tilt gimbal. ~59 fps end-to-end on a GTX 1650 SUPER, ~10 ms
PC-side latency, sub-50 ms system latency including servo travel.

## Features

- 3-process pipeline (Capture → Inference → Output) over lock-free shared memory, plus an optional phone-IMU sidecar
- Ultralytics YOLO26n pose models (human 17-kp, dog 24-kp) exported to TensorRT FP16
- BoT-SORT tracking with an adaptive Kalman filter (innovation gating, OC-SORT observation replay), confidence-hysteresis target lock and a gentle ramp-in after short same-track losses
- Optional detector skipping: Kalman-only prediction frames between YOLO runs to free GPU time
- Camera sources: USB / DroidCam virtual camera, DroidCam direct stream, or the TaffCam Android app streaming over USB
- Slew-limited output commands, optional backlash compensation, and a search sweep when the target is lost
- Optional laser boresight calibration, laser-dot overlay, and relay output (manual key or opt-in auto-fire)
- ESP32-S3 firmware: 200 Hz FreeRTOS control loop, USB + WiFi UDP transports, staged watchdog (coast / hold / home) and a relay fail-safe
- PySide6 desktop GUI (`tracker-gui`) and a Windows launcher (`launch.bat`)

## Hardware

- NVIDIA GPU with TensorRT-compatible drivers (tested on GTX 1650 SUPER, CUDA 12)
- USB camera or phone-as-webcam (1080p60 recommended)
- ESP32-S3 board + two MG996R servos + power supply
- Optional: laser diode for boresight calibration

## Quick start

```bash
git clone https://github.com/brahimxallal/TaffTracker.git
cd TaffTracker
python -m venv .venv && .venv\Scripts\activate   # Windows
pip install -r requirements-dev.txt
python scripts/export_engines.py --target human   # one-time TRT export
python -m src.main --mode camera --target human --source 0
```

On Windows, `launch.bat` wraps the same steps in an interactive menu, and `pip install -e ".[gui]"` followed by `tracker-gui` starts the desktop GUI.

Full setup (TRT export, firmware flash, calibration): [docs/INSTALL.md](docs/INSTALL.md).
System overview and pipeline walkthrough: [docs/ARCHITECTURE.md](docs/ARCHITECTURE.md).
Live tuning (lead, smoother, relock, slew limits, detector schedule): [docs/TUNING.md](docs/TUNING.md).

## Safety and security notes

- The relay and laser outputs can switch real loads. `relay.auto_fire_enabled` is off by default, and the firmware releases the relay when the PC has been silent for more than 0.5 s. Follow the eye-safety rules for whatever laser you fit.
- The gimbal accepts commands without authentication (CRC16 integrity check only) over USB and over WiFi UDP port 6000. Run it on a network you trust. See [SECURITY.md](SECURITY.md) for the trust model and how to report a problem.

## Project layout

```
src/
  main.py              orchestrator, hotkey loop
  config.py            frozen-dataclass config schema
  config_loader.py     YAML → PipelineConfig + CLI override
  capture/             camera read + letterbox (OpenCV, DroidCam, TaffCam)
  inference/           TRT YOLO + tracker + Kalman + detector scheduler
  output/              controller, command smoothing, serial / UDP transport
  gui/                 PySide6 desktop GUI and runtime session
  sensors/             phone motion-sensor (IMU) stream
  ui/                  OpenCV overlays, hotkeys, manual keyboard driver
  tracking/            Kalman, ByteTrack, BoT-SORT, 1€ filter
  calibration/         pinhole model and laser boresight
  laser/               HSV laser-dot detector
  shared/              ring buffer, display buffer, protocol, types
firmware/esp32s3_gimbal/   ESP-IDF firmware (200 Hz control loop) + host_tests/
android/taffcam/       TaffCam Android camera app
tests/                 pytest suite (unit / integration / perf markers)
scripts/               export, calibrate, benchmark utilities
config.yaml            runtime tuning (tracked, no secrets)
```

## Testing

```bash
pytest                   # full suite (CPU-only, no CUDA needed)
pytest -m unit           # fast unit tests
pytest -m integration    # cross-process tests
pytest --cov=src         # coverage report
```

`tests/conftest.py` stubs TensorRT/CUDA at collection time, so the suite
runs on any machine. The firmware host tests (`tests/test_firmware_host_harness.py`)
compile the real firmware sources with gcc or clang under ASan/UBSan and skip
themselves when no C compiler is installed.

## Contributing

See [CONTRIBUTING.md](CONTRIBUTING.md) for setup, commit convention, and PR
checklist. `AGENTS.md` documents the internal architecture and pitfalls in
more depth.

## License

MIT — see [LICENSE](LICENSE). Note: Ultralytics YOLO is AGPL-licensed;
commercial use of the trained models requires a separate Ultralytics
license.
