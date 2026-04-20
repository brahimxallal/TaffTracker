# Installation Guide

End-to-end setup for **Vision Gimbal Tracker**: Python/TensorRT pipeline + ESP32-S3 firmware + camera calibration.

Expect ~30 minutes on a clean machine with CUDA already installed.

---

## 1. Prerequisites

### PC side
| Component | Version | Notes |
|---|---|---|
| OS | Windows 10/11 or Linux x86_64 | Developed on Windows 10 |
| GPU | NVIDIA, compute 7.5+ | Tested on GTX 1650 SUPER |
| CUDA | 12.x | Match your TensorRT build |
| TensorRT | 10.x | `python -c "import tensorrt; print(tensorrt.__version__)"` |
| Python | 3.12 exactly | See `.python-version` |
| Git | 2.40+ | |

### Hardware
- USB camera or phone-as-webcam (1080p60 recommended)
- ESP32-S3 board (tested: ESP32-S3-N16R8)
- Two MG996R servos (pan + tilt) + external 5–6 V power supply
- Optional: red laser diode for closed-loop visual servoing

### Firmware toolchain
- [PlatformIO Core](https://docs.platformio.org/en/latest/core/installation/index.html) or ESP-IDF 5.x
- USB serial driver for the ESP32-S3 (CP210x or native CDC)

---

## 2. Clone & Python environment

```bash
git clone https://github.com/brahimxallal/TaffTracker.git
cd TaffTracker
python -m venv .venv
# Windows
.venv\Scripts\activate
# Linux/macOS
source .venv/bin/activate

pip install -r requirements-dev.txt
pre-commit install        # optional: auto-format on commit
```

Verify the environment:
```bash
pytest -m unit            # should pass in < 20 s, no CUDA needed
```

---

## 3. Download / place YOLO weights

The repo does **not** ship model weights. Drop them here:

```
models/
├── yolo26n-person-17pose.pt         # 17-keypoint human pose
└── Enhanceddog/
    └── best.pt                       # 24-keypoint dog pose (custom-trained)
```

Use your own Ultralytics-exported weights. Any `.pt` file with a matching keypoint schema works.

---

## 4. Export TensorRT engines (one-time)

The live pipeline loads `.engine` files, not `.pt`. Export them once per GPU:

```bash
python scripts/export_engines.py --target human
python scripts/export_engines.py --target dog
```

Output lands in `engines/` (gitignored). Expect FP16 @ 640×640, end-to-end NMS, ~4 GB workspace. On a GTX 1650 SUPER this takes 60–120 s per model.

Re-export when you change:
- GPU (engines are not portable across GPUs)
- TensorRT version
- Model weights

---

## 5. Camera calibration (optional but recommended)

### 5.1 Intrinsics
Print a checkerboard (do **not** use a phone screen — we tried, it gives bad results) and run:
```bash
python scripts/calibrate.py --camera-id 0 --frames 20
```
Output: `calibration_data/intrinsics.npz` — picked up automatically by `CameraModel`.

### 5.2 Camera FOV fallback
If you skip intrinsic calibration, set your camera HFOV in `config.yaml`:
```yaml
camera:
  hfov_deg: 66.2      # measure once, e.g. with a ruler at known distance
```
`CameraModel.from_fov()` computes focal length as `fx = (width/2) / tan(hfov/2)`.

### 5.3 Laser boresight (visual-servo mode only)
Paint a dot on a wall at ~2 m, engage the laser, point the gimbal at the dot, then:
```bash
python scripts/laser_capture.py
```
Records the pixel offset between optical center and laser impact. Stored in `calibration_data/laser_boresight.json`.

---

## 6. Firmware: flash the ESP32-S3

### 6.1 WiFi credentials
```bash
cp firmware/esp32s3_gimbal/wifi_secrets.env.example firmware/esp32s3_gimbal/wifi_secrets.env
# then edit the two lines
```
`wifi_secrets.env` is gitignored — your SSID and password never leave your disk.

### 6.2 Build + flash
```bash
cd firmware/esp32s3_gimbal
pio run -t upload --upload-port COM4        # Windows
pio run -t upload --upload-port /dev/ttyACM0  # Linux
```
First boot writes servo defaults to NVS. Watch the serial console (`pio device monitor`) to confirm the pan/tilt servos sweep through their calibration range.

**Troubleshooting:**
- `Error: Unable to find platform 'espressif32'` → `pio pkg install -g --platform espressif32`
- Whitespace in your Windows user path breaks PlatformIO → set `PLATFORMIO_CORE_DIR=C:/pio` in your shell before `pio run`.
- ESP32 reboots in a loop → check servo power supply is separate from the USB 5 V rail.

---

## 7. First run

```bash
python -m src.main --mode camera --target human --source 0
```

What you should see:
- OpenCV window with bbox, keypoints, centroid, gimbal cross-hair overlay
- Terminal showing ~59 fps, end-to-end PC latency ~8–10 ms
- If ESP32 is connected over USB, servos track the target within 50 ms total system latency

Useful flags:
| Flag | Purpose |
|---|---|
| `--target dog` | Switch to 24-keypoint dog model |
| `--source path/to/video.mp4` | Offline playback instead of live camera |
| `--headless` | Skip the OpenCV window (for benchmarking / logging runs) |
| `--profile` | Emit per-stage latency histograms on shutdown |
| `--record` | Save H.264 video + CSV metadata to `logs/recordings/` |
| `--config path/to/config.yaml` | Override the default config file |
| `--debug` / `--log-level DEBUG` | Verbose logs |

---

## 8. Verify tests pass

```bash
pytest -m "not perf"      # full CPU-only suite, ~12 s, should show 593 passing
pytest --cov=src          # with coverage (~81 % overall)
```

CUDA-dependent tests are skipped automatically via `tests/conftest.py` when TRT is not importable.

---

## 9. Common next steps

- **Tune for your scene:** edit `config.yaml` — thresholds, Kalman R/Q, smoothing alphas, gimbal limits.
- **Add a new target class:** define a `PoseSchema` in `src/shared/pose_schema.py`, add weights + engine, extend `default_tracking_config()`.
- **Swap transport:** set `comms.transport: auto` / `serial` / `udp` in `config.yaml`.

For architecture details see [AGENTS.md](../AGENTS.md). For contribution guidelines see [CONTRIBUTING.md](../CONTRIBUTING.md).
