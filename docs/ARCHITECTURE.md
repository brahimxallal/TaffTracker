# Architecture

This document is the public-facing architecture overview of **Vision Gimbal Tracker**. For day-to-day contributor guidance and pitfalls, see [`AGENTS.md`](../AGENTS.md).

The system is a low-latency vision-guided pan/tilt tracker. A PC runs YOLO + TensorRT pose detection and produces angular setpoints, an ESP32-S3 runs a 200 Hz servo control loop, and a small binary protocol bridges the two.

---

## 1. System View

```
┌─────────────────────────── PC ─────────────────────────────┐         ┌───────── ESP32-S3 ─────────┐
│                                                             │ Serial  │                            │
│  ┌──────────────┐  ring   ┌────────────────┐  mp.Queue      │ /UDP    │  comm_task (USB+UDP)       │
│  │ CaptureProc  │ ──buf──▶│ InferenceProc  │ ──results────▶│────────▶│  control_task (200 Hz)     │
│  │ OpenCV grab  │ 3-slot  │ YOLO + tracker │                │ packets │  servo_driver (LEDC PWM)   │
│  │ + letterbox  │ seqlock │ + Kalman + ORU │                │         │  watchdog + soft-home      │
│  └──────────────┘         └────────────────┘                │         └────────────────────────────┘
│         ▲                          │                        │                       │
│         │                          ▼                        │                       │
│         │                    ┌──────────────┐               │              ┌────────▼────────┐
│         │                    │  OutputProc  │               │              │  MG996R x 2     │
│         │                    │ pkt encoder  │               │              │  (pan + tilt)   │
│         │                    │ + comms TX   │               │              └─────────────────┘
│         │                    └──────────────┘               │
│         │                          │                        │
│         │                          ▼                        │
│         │                ┌────────────────────┐             │
│         │                │ Main process       │             │
│         │                │ cv2.imshow + keys  │             │
│         │                └────────────────────┘             │
│                                                             │
└─────────────────────────────────────────────────────────────┘
```

End-to-end PC latency is ~8–10 ms on a GTX 1650 SUPER at 1080p60. Total system latency including servo travel stays under 50 ms for typical motion.

---

## 2. Process Boundaries

The runtime is **three OS-level processes** plus the main process for display. Boundaries are deliberate: a hung GPU never freezes capture, a stuck serial port never starves inference.

| Process | Owns | Hot-path budget |
|---|---|---|
| `CaptureProcess` | OpenCV `VideoCapture`, letterbox, ring-buffer writes | < 2 ms / frame |
| `InferenceProcess` | TensorRT engine, tracker (BoTSORT), Kalman filter, centroid, pixel→angle | < 10 ms / frame |
| `OutputProcess` | Protocol encoding, serial / UDP TX, manual-mode joystick blend, fail-safe state | < 1 ms / frame |
| Main process | `cv2.imshow`, hotkeys, supervisor (drains error queue, restarts dead procs) | runs at display rate |

Inter-process plumbing:

- **Capture → Inference**: a 3-slot **shared-memory ring buffer** (`src/shared/ring_buffer.py`) with a generation-counter seqlock. Zero-copy: the consumer reads the same `np.ndarray` view the producer wrote.
- **Inference → Output**: a standard `multiprocessing.Queue[TrackingMessage]`. Bounded depth; backpressure drops the oldest, never the newest.
- **Any → Main**: an error queue. The supervisor reads it, logs, and decides whether to attempt a restart.

---

## 3. Inference Pipeline

```
frame ──▶ TensorRT YOLO ──▶ postprocess ──▶ BoTSORT ──▶ Kalman + ORU ──▶ centroid ──▶ 1€ filter ──▶ (optional) laser ──▶ TrackingMessage
```

Notable stages:

- **TensorRT YOLO** (`src/inference/trt_engine.py`): FP16, static 640×640, end-to-end NMS, CUDA graph captured after 5 warmup frames. Graceful fallback if graph capture fails.
- **Postprocess** (`src/inference/postprocess.py`): cascading centroid — head keypoints → body keypoints (dogs only) → smart bbox top-portion fallback. MAD outlier rejection, conf-weighted blending, temporal `KeypointStabilizer`.
- **Tracker**: BoTSORT for multi-object association; per-track Kalman state (`src/tracking/kalman.py`) with adaptive R/Q, 8σ Mahalanobis innovation gating, OC-SORT observation re-update, prediction capping (30 max).
- **Lock policy** (`src/inference/process.py`): single-source `_select_primary_track` per frame. Spatial-proximity re-lock when a recent target reappears nearby. Per-track state cache (8 entries, evict oldest) snapshots Kalman + stabilizer + EMA across lock transitions so the gimbal doesn't snap.
- **Adaptive controller** (`src/tracking/adaptive.py`): rolling 120-frame detection-reliability window and 60-frame speed window auto-tune confidence thresholds and hold-time at runtime, in addition to FPS-driven Q / `max_lost` adaptation.
- **Pinhole geometry** (`src/calibration/camera_model.py`): `pixel_to_angle` via `atan2`, `pixel_velocity_to_angular` for lead. Focal length is derived from `camera.fov` in `config.yaml` (FOV-only — checkerboard intrinsics removed).

The output is a `TrackingMessage` (`src/shared/types.py`) carrying servo angles, angular velocity, hold-time, latency totals, and tracker state.

---

## 4. Output Pipeline

`OutputProcess` (`src/output/process.py`) is intentionally fire-and-forget:

- No retransmission, no blocking ack — keeps the loop deterministic.
- Auto-mode controller (`src/output/auto_controller.py`) is a pure function over `(message, state, config)`: gain-scheduled PD, predictive lead, slew cap, integral decay, deadband, mechanical-limit clamp.
- Manual-mode controller (`src/ui/manual_keyboard.py`) uses ZQSD = fine speed, arrows = coarse speed, with deterministic acceleration ramping and a clock injection point for unit tests.
- Transport selection (`src/output/sender_factory.py`): `serial`, `udp`, or `auto` with health probing. Both transports report health so the main process can flip between them transparently.
- Protocol v2 (`src/shared/protocol.py`): 20-byte little-endian frame with sync byte 0xBB, target angles, angular velocity, tracker state flags, quality, latency. CRC checksum over the body.

Sign inversion is applied **only at output encoding**, not in the Kalman state, so the internal pipeline stays sign-agnostic and the same models work on flipped mounts.

---

## 5. Firmware

The ESP32-S3 firmware (`firmware/esp32s3_gimbal/`) is small and disciplined:

- `comm_task.c` — receives packets over USB-CDC and WiFi UDP in parallel, parses, validates, and stores into a spinlock-protected `tracking_command_t` shared structure.
- `control_task.c` — `esp_timer` at 200 Hz, pinned to Core 1. Applies its own EMA chain (stat / slow / mod / fast alphas selected by motion magnitude), velocity + acceleration limits, dead-zone, prediction-decay penalty, and soft-home on watchdog timeout.
- `servo_driver.c` — LEDC 14-bit PWM at 50 Hz; converts target angle → duty cycle with mechanical limits.
- `main.c` — init order: servos → USB → WiFi → control loop. WiFi credentials come from `wifi_secrets.env` (gitignored).

Filtering parameters in firmware are intentionally conservative defaults; live tuning is documented in the project memory under "Phase Q".

---

## 6. Configuration

All tuning is data-driven:

```
config.yaml ──▶ src/config_loader.load_yaml_config()
                      │
                      ▼
                CLI overrides (src/cli.py argparse)
                      │
                      ▼
                build_config_from_yaml() ──▶ frozen PipelineConfig ──▶ each process
```

`src/config.py` defines frozen dataclasses (`CameraConfig`, `GimbalConfig`, `KalmanConfig`, `AdaptiveConfig`, `PreflightConfig`, …). Frozen-by-default keeps process boundaries safe — no process can mutate config that another process is reading.

Calibration artifacts (`calibration_data/`), TensorRT engines (`engines/`), and YOLO weights (`models/`) are all gitignored. The repo ships only the recipe to produce them.

---

## 7. Failure Modes & Fail-Safes

| Fault | Detection | Recovery |
|---|---|---|
| Camera disconnect | OpenCV grab returns `False` for N frames | Capture proc restart; main loop continues |
| GPU / TRT crash | Inference proc dies | Supervisor restarts; ring-buffer producer keeps writing |
| Serial-port unplug | `OSError` on write | Output proc reconnects with exponential backoff (0.5 → 5 s) |
| WiFi loss | UDP probe (ping) fails | Background monitor flips `is_connected`; sender drops to serial if available |
| ESP32 packet timeout | Firmware watchdog | Soft-home (slew toward 0°), keep PWM live, resume on next valid packet |
| Target lost | Kalman `prediction_count` exceeds cap | Hold last commanded angle for `hold_time_s`, then re-acquire |

The shared ring buffer and `mp.Queue` boundaries mean any single failure is local — no other process can be blocked or corrupted.

---

## 8. Testing Strategy

- **Unit** (`pytest -m unit`) — pure functions, dataclasses, controllers, filters. Runs in < 20 s on CPU.
- **Integration** (`pytest -m integration`) — multi-process boundaries with stubbed hardware.
- **Performance** (`pytest -m perf`) — opt-in microbenchmarks, only in dedicated CI lanes.

`tests/conftest.py` stubs `tensorrt`, `pycuda`, and `cv2` hardware paths at collection time so the suite runs on any machine, even without CUDA.

---

## 9. Where to Start Reading

If you want to follow the data end-to-end, read in this order:

1. `src/main.py` — orchestration only; spawn / join / display.
2. `src/cli.py` — config plumbing.
3. `src/capture/process.py` → `src/shared/ring_buffer.py` — frame in.
4. `src/inference/process.py` → `src/inference/postprocess.py` → `src/tracking/kalman.py` — frame out as angles.
5. `src/output/process.py` → `src/output/auto_controller.py` → `src/shared/protocol.py` — angles to wire.
6. `firmware/esp32s3_gimbal/src/control_task.c` — wire to servo PWM.

The contributor-focused [`AGENTS.md`](../AGENTS.md) lists every key file and the current pitfalls you should know about before editing.
