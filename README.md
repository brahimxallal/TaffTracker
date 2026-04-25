# Vision Gimbal Tracker

Ultra-low-latency YOLO + TensorRT pose-tracking pipeline that drives an
ESP32-S3 pan/tilt gimbal. ~59 fps end-to-end on a GTX 1650 SUPER, ~10 ms
PC-side latency, sub-50 ms system latency including servo travel.

## Features

- 3-process pipeline (Capture → Inference → Output) with lock-free shared memory
- Ultralytics YOLO26n pose models (human 17-kp, dog 24-kp) exported to TensorRT FP16
- Adaptive Kalman filter with innovation gating and OC-SORT observation-replay
- Optional laser-guided visual servoing (closed-loop PID)
- ESP32-S3 firmware: 200 Hz FreeRTOS control loop, USB + WiFi UDP transports

## Hardware

- NVIDIA GPU with TensorRT-compatible drivers (tested on GTX 1650 SUPER, CUDA 12)
- USB camera or phone-as-webcam (1080p60 recommended)
- ESP32-S3 board + two MG996R servos + power supply
- Optional: laser diode for visual-servo mode

## Quick start

```bash
git clone https://github.com/brahimxallal/TaffTracker.git
cd TaffTracker
python -m venv .venv && .venv\Scripts\activate   # Windows
pip install -r requirements-dev.txt
python scripts/export_engines.py --target human   # one-time TRT export
python -m src.main --mode camera --target human --source 0
```

Full setup (TRT export, firmware flash, calibration): [docs/INSTALL.md](docs/INSTALL.md).
System overview and pipeline walkthrough: [docs/ARCHITECTURE.md](docs/ARCHITECTURE.md).
Live tuning (lead, smoother, GPU preprocess): [docs/TUNING.md](docs/TUNING.md).

## Project layout

```
src/
  main.py              orchestrator, hotkey loop
  config.py            frozen-dataclass config schema
  config_loader.py     YAML → PipelineConfig + CLI override
  capture/             camera read + letterbox
  inference/           TRT YOLO + tracker + Kalman
  output/              serial / UDP transport + visualizer
  tracking/            Kalman, ByteTrack, BoT-SORT, 1€ filter, visual servo
  calibration/         pinhole model, depth, laser boresight
  laser/               HSV laser-dot detector
  shared/              ring buffer, display buffer, protocol, types
firmware/esp32s3_gimbal/   ESP-IDF firmware (200 Hz control loop)
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
runs on any machine.

## Contributing

See [CONTRIBUTING.md](CONTRIBUTING.md) for setup, commit convention, and PR
checklist. `AGENTS.md` documents the internal architecture and pitfalls in
more depth.

## License

MIT — see [LICENSE](LICENSE). Note: Ultralytics YOLO is AGPL-licensed;
commercial use of the trained models requires a separate Ultralytics
license.
