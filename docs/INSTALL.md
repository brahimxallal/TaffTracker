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
- Optional: red laser diode for boresight calibration

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

## 5. Camera calibration

### 5.1 Camera FOV
The aiming math uses a pinhole model derived from a single horizontal
field-of-view value. Measure your camera's HFOV once (a ruler at a
known distance is fine — see ["Measuring camera HFOV"](#measuring-camera-hfov))
and set it in `config.yaml`:
```yaml
camera:
  fov: 66.2           # phone camera example; set to your camera's value
```
`CameraModel.from_fov()` computes focal length as `fx = (width/2) / tan(hfov/2)`.

> Earlier versions supported intrinsic checkerboard calibration via
> `intrinsics.npz`. That path was removed — the printed-checkerboard
> workflow was never reliably reproducible on the deployed phone-camera
> mount, and the FOV-only path is accurate enough for sub-degree
> aiming at the working distances we hit in practice.

### 5.2 Gimbal center and laser boresight
Align the laser with the camera's optical center and persist the offsets. The
calibrator needs the **USB serial** connection (calibration packets are not
handled on the UDP path):
```bash
python scripts/calibrate.py --port COM4     # manual jog, then auto-refine
python scripts/calibrate.py --auto          # skip the manual jog
python scripts/calibrate.py --reset         # zero all offsets
```
It saves `calibration_data/servo_limits.json` and pushes the offsets to the
firmware's NVS, so the next boot uses them automatically.
`scripts/laser_capture.py` is only a diagnostic helper that saves
`logs/laser_frame.png` and prints HSV values for threshold tuning.

#### Measuring camera HFOV
Place an object of known width `W` (e.g. a 30 cm ruler) at a known
distance `D` (e.g. 1.0 m) perpendicular to the camera. Note the pixel
width `w` it spans in the frame, then:

```
hfov_deg = 2 * degrees(atan((W / 2) / D))
```

A 30 cm ruler at 1 m on a phone camera typically gives ~28°
horizontally for a tight crop or ~66° for the wide lens — pick the
lens you'll actually run.

---

## 6. Firmware: flash the ESP32-S3

### 6.1 WiFi credentials
```bash
cp firmware/esp32s3_gimbal/wifi_secrets.env.example firmware/esp32s3_gimbal/wifi_secrets.env
# then edit the two lines
```
`wifi_secrets.env` is gitignored — your SSID and password never leave your disk.

Optionally pin the ESP32's WiFi address by also setting `TAFFF_WIFI_STATIC_IP`,
`TAFFF_WIFI_GATEWAY`, `TAFFF_WIFI_NETMASK` (and `TAFFF_WIFI_DNS`) in the same
file, and use that address as `comms.udp_host` in `config.yaml`. A DHCP
reservation on your router is the more robust option; if you use the firmware
static IP, pick an address outside the router's DHCP pool. Missing or invalid
values fall back to DHCP.

### 6.2 Build + flash
```bash
cd firmware/esp32s3_gimbal
pio run -t upload --upload-port COM4        # Windows
pio run -t upload --upload-port /dev/ttyACM0  # Linux
```
On boot the firmware loads saved NVS offsets when present, otherwise it uses compile-time defaults. Watch the serial console (`pio device monitor`) to confirm USB/WiFi startup and control-loop health.

**Troubleshooting:**
- `Error: Unable to find platform 'espressif32'` → `pio pkg install -g --platform espressif32`
- Whitespace in your Windows user path breaks PlatformIO → set `PLATFORMIO_CORE_DIR=C:/pio` in your shell before `pio run`.
- ESP32 reboots in a loop → check servo power supply is separate from the USB 5 V rail.
- Gimbal won't join WiFi → read `pio device monitor` over USB. Every failed join logs `WiFi disconnected: reason=<code> (<NAME>)` plus a one-line hint, for example `NO_AP_FOUND` (SSID not seen: check the name, that the router's 2.4 GHz WiFi is on, and channel 1-11) or `AUTH_FAIL` (most likely a wrong password). The `WiFi config:` line shows only the SSID and password lengths, never the values.

---

## 7. First run

```bash
python -m src.main --mode camera --target human --source 0
```

What you should see:
- OpenCV window with bbox, keypoints, centroid, gimbal cross-hair overlay
- Terminal showing ~59 fps, end-to-end PC latency ~8–10 ms
- If ESP32 is connected over USB, servos track the target within 50 ms total system latency

Other ways to start:
- `launch.bat` (Windows): interactive launcher. It prefers the project `.venv`, checks that the selected TensorRT engine can be loaded and offers to re-export it if not.
- `tracker-gui` after `pip install -e ".[gui]"`: the PySide6 desktop front-end.

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

### Optional: phone as camera (TaffCam)

`android/taffcam/` is a small Android app that streams the phone's camera to
the PC over USB. Open it in Android Studio, install the debug build on a phone
with USB debugging enabled, then pick a phone backend in `config.yaml`:

```yaml
camera:
  source_backend: phone_yuv   # raw I420 frames; phone_mpeg / phone_h264 for encoded streams
```

The stream and camera controls (resolution, exposure, ISO, torch, ...) live in
the `phone_camera:` block. On start the tracker sets up `adb reverse` for the
frame and control ports (27183 / 27184) and launches the app for you
(`phone_camera.start_app_on_open`). Both listeners bind to `127.0.0.1` unless
you set `allow_remote_clients: true`, which you should only do on a network you
trust. The shipped default is the DroidCam virtual camera
(`source_backend: opencv`).

The optional phone motion-sensor stream (`phone_imu.enabled`, off by default)
needs a TaffCam build that includes the sensor service.

---

## 8. Verify tests pass

```bash
pytest -m "not perf"      # full CPU-only suite, typically well under a minute, everything should pass
pytest --cov=src          # with a coverage report
```

CUDA-dependent tests are skipped automatically via `tests/conftest.py` when TRT is not importable. The firmware host tests (`tests/test_firmware_host_harness.py`) need gcc or clang and skip themselves when no C compiler is found.

---

## 9. Common next steps

- **Tune for your scene:** edit `config.yaml` — thresholds, Kalman R/Q, smoothing alphas, gimbal limits.
- **Add a new target class:** define a `PoseSchema` in `src/shared/pose_schema.py`, add weights + engine, extend `default_tracking_config()`.
- **Swap transport:** set `comms.channel: auto` / `serial` / `udp` in `config.yaml`.

For architecture details see [AGENTS.md](../AGENTS.md). For contribution guidelines see [CONTRIBUTING.md](../CONTRIBUTING.md).
