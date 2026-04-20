"""Gimbal center calibration tool.

Aligns the laser with the camera's optical center by finding the pan/tilt
offset that places the laser dot at the center of the camera frame.

Results are saved to calibration_data/servo_limits.json and pushed to ESP32 NVS
so the firmware applies them immediately.

Calibration requires USB serial. Firmware calibration packets are not handled
on the UDP path.

Usage:
    python scripts/calibrate.py [--port COM4] [--baud 921600] [--source 0]
    python scripts/calibrate.py --auto          # skip manual jog, auto-converge
    python scripts/calibrate.py --reset         # zero all offsets

Controls (manual jog):
    LEFT / RIGHT       : nudge pan +/-2 deg
    UP / DOWN          : nudge tilt +/-2 deg
    ZQSD               : nudge +/-2 deg  (AZERTY)
    Shift + ZQSD       : nudge +/-10 deg
    +/-                : nudge +/-0.1 deg
    ENTER              : start auto-refine
    ESC                : accept current position
"""
from __future__ import annotations

import argparse
import json
import os
import signal
import sys
import time
from math import degrees, radians, tan
from pathlib import Path

import cv2
import numpy as np
import serial

sys.path.insert(0, ".")
from src.config import LaserConfig
from src.laser.detector import LaserDetector
from src.shared.protocol import (
    CAL_CMD_GET_OFFSETS,
    CAL_CMD_SET_OFFSETS,
    CAL_PACKET_SIZE,
    STATE_MEASUREMENT,
    FLAG_TARGET_ACQUIRED,
    FLAG_LASER_ON,
    decode_cal_response,
    encode_cal_get_offsets,
    encode_packet_v2,
    encode_cal_set_offsets,
)

# ── Paths ────────────────────────────────────────────────────────────────
LIMITS_PATH = Path("calibration_data/servo_limits.json")
CONFIG_PATH = Path("config.yaml")

# ── Timing ───────────────────────────────────────────────────────────────
SEND_INTERVAL_MS = 16
MANUAL_JOG_STEP_DEG = 2.0
MANUAL_JOG_FAST_STEP_DEG = 10.0

# ── Differential laser detection ────────────────────────────────────────
DIFF_SETTLE_MS = 300
DIFF_FRAMES = 8
DIFF_THRESHOLD = 30
DIFF_MIN_AREA = 4.0
DIFF_MAX_AREA = 2000.0

# ── Windows arrow key codes (cv2.waitKeyEx) ──────────────────────────────
_KEY_LEFT = 0x250000
_KEY_RIGHT = 0x270000
_KEY_UP = 0x260000
_KEY_DOWN = 0x280000

_gpu_available = False
_gpu_checked = False


# ═══════════════════════════════════════════════════════════════════════
#  Transport abstraction (serial only)
# ═══════════════════════════════════════════════════════════════════════

class Transport:
    """Thin wrapper around the serial transport used by calibration."""
    def write(self, data: bytes) -> None:
        raise NotImplementedError

    def read(self, size: int, timeout_s: float = 0.2) -> bytes:
        raise NotImplementedError

    def reset_input_buffer(self) -> None:
        raise NotImplementedError

class SerialTransport(Transport):
    def __init__(self, port: str, baud: int) -> None:
        print(f"  Opening {port} at {baud}...")
        self._ser = serial.Serial(port, baud, timeout=0.01)
        time.sleep(0.3)
    def write(self, data: bytes) -> None:
        self._ser.write(data)

    def read(self, size: int, timeout_s: float = 0.2) -> bytes:
        deadline = time.perf_counter() + timeout_s
        payload = bytearray()
        while len(payload) < size and time.perf_counter() < deadline:
            chunk = self._ser.read(size - len(payload))
            if chunk:
                payload.extend(chunk)
            else:
                time.sleep(0.005)
        return bytes(payload)

    def reset_input_buffer(self) -> None:
        self._ser.reset_input_buffer()

    def close(self) -> None:
        self._ser.close()


# ═══════════════════════════════════════════════════════════════════════
#  Helpers
# ═══════════════════════════════════════════════════════════════════════

def _check_gpu() -> bool:
    """Lazy GPU check — runs once."""
    global _gpu_available, _gpu_checked
    if _gpu_checked:
        return _gpu_available
    _gpu_checked = True
    try:
        if cv2.cuda.getCudaEnabledDeviceCount() > 0:
            _gpu_available = True
    except Exception:
        pass
    return _gpu_available


def send_angles(ser: Transport, pan_deg: float, tilt_deg: float, seq: int, *, laser_on: bool = True) -> None:
    """Send a v2 protocol packet with given pan/tilt degrees."""
    state = STATE_MEASUREMENT | FLAG_TARGET_ACQUIRED
    if laser_on:
        state |= FLAG_LASER_ON
    pkt = encode_packet_v2(
        sequence=seq & 0xFFFF,
        timestamp_ms=int(time.time() * 1000) & 0xFFFFFFFF,
        pan=int(round(pan_deg * 100)),
        tilt=int(round(tilt_deg * 100)),
        pan_vel=0,
        tilt_vel=0,
        confidence=255,
        state=state,
        quality=255,
        latency=1,
    )
    ser.write(pkt)


def hold_position(ser: Transport, pan: float, tilt: float, duration_ms: int, seq: int) -> int:
    """Send keep-alive packets for duration_ms. Returns updated sequence."""
    start = time.perf_counter()
    while (time.perf_counter() - start) * 1000 < duration_ms:
        send_angles(ser, pan, tilt, seq)
        seq += 1
        time.sleep(SEND_INTERVAL_MS / 1000)
    return seq


def make_laser_detector() -> LaserDetector:
    """Create a laser detector with relaxed thresholds for calibration."""
    return LaserDetector(LaserConfig(
        enabled=True,
        hue_low_upper=15,
        hue_high_lower=165,
        sat_min=40,
        val_min=100,
        min_area=1.0,
        max_area=1500.0,
        min_circularity=0.05,
        roi_radius_px=1000.0,
    ))


def parse_jog_key(raw_key: int) -> tuple[float, float]:
    """Parse arrow/ZQSD/+- keys into (d_pan, d_tilt). Returns (0, 0) if unrecognised."""
    key = raw_key & 0xFF
    if raw_key == _KEY_LEFT:   return (-MANUAL_JOG_STEP_DEG, 0.0)
    if raw_key == _KEY_RIGHT:  return (+MANUAL_JOG_STEP_DEG, 0.0)
    if raw_key == _KEY_UP:     return (0.0, +MANUAL_JOG_STEP_DEG)
    if raw_key == _KEY_DOWN:   return (0.0, -MANUAL_JOG_STEP_DEG)

    key_lower = chr(key).lower() if key else ""
    step = MANUAL_JOG_FAST_STEP_DEG if chr(key).isupper() else MANUAL_JOG_STEP_DEG
    if key_lower == "q":  return (-step, 0.0)
    if key_lower == "d":  return (+step, 0.0)
    if key_lower == "z":  return (0.0, +step)
    if key_lower == "s":  return (0.0, -step)
    if key == ord("+") or key == ord("="):  return (0.1, 0.0)
    if key == ord("-"):                     return (-0.1, 0.0)
    return (0.0, 0.0)


def save_limits_json(limits: dict) -> None:
    """Write servo_limits.json (backward compat)."""
    LIMITS_PATH.parent.mkdir(parents=True, exist_ok=True)
    limits["calibrated_at"] = time.strftime("%Y-%m-%dT%H:%M:%S")
    LIMITS_PATH.write_text(json.dumps(limits, indent=2), encoding="utf-8")


def load_config_values() -> tuple[float, bool, bool]:
    """Read camera.fov, gimbal.invert_pan, gimbal.invert_tilt from config.yaml."""
    fov = 90.0
    invert_pan = False
    invert_tilt = False
    if CONFIG_PATH.exists():
        import yaml
        with open(CONFIG_PATH, encoding="utf-8") as f:
            data = yaml.safe_load(f) or {}
        cam = data.get("camera", {})
        if cam.get("fov"):
            fov = float(cam["fov"])
        gimbal = data.get("gimbal", {})
        invert_pan = bool(gimbal.get("invert_pan", False))
        invert_tilt = bool(gimbal.get("invert_tilt", False))
    return fov, invert_pan, invert_tilt


def read_current_offsets(ser: Transport) -> tuple[float, float] | None:
    ser.reset_input_buffer()
    ser.write(encode_cal_get_offsets())
    response = ser.read(CAL_PACKET_SIZE, timeout_s=0.3)
    decoded = decode_cal_response(response)
    if decoded is None or decoded.command != CAL_CMD_GET_OFFSETS:
        return None
    return decoded.pan_offset_deg, decoded.tilt_offset_deg


def write_offsets_and_confirm(ser: Transport, pan: float, tilt: float) -> bool:
    ser.reset_input_buffer()
    ser.write(encode_cal_set_offsets(pan, tilt))
    response = ser.read(CAL_PACKET_SIZE, timeout_s=0.3)
    decoded = decode_cal_response(response)
    if decoded is None or decoded.command != CAL_CMD_SET_OFFSETS:
        return False
    return (
        abs(decoded.pan_offset_deg - pan) <= 0.01
        and abs(decoded.tilt_offset_deg - tilt) <= 0.01
    )


# ═══════════════════════════════════════════════════════════════════════
#  Differential laser detection (ON/OFF subtraction)
# ═══════════════════════════════════════════════════════════════════════

def _capture_median_frame(cap: cv2.VideoCapture, n_frames: int) -> np.ndarray | None:
    """Capture n_frames and return the per-pixel median frame."""
    frames = []
    for _ in range(n_frames):
        ret, frame = cap.read()
        if ret:
            frames.append(frame)
    if len(frames) < 2:
        return frames[0] if frames else None
    return np.median(np.stack(frames), axis=0).astype(np.uint8)


def _diff_find_blob(diff: np.ndarray) -> tuple[float, float] | None:
    """Find the laser blob in a diff image using red-channel dominance."""
    if diff.ndim != 3:
        gray = diff
    else:
        b, g, r = cv2.split(diff)
        r_f = r.astype(np.float32)
        g_f = g.astype(np.float32)
        b_f = b.astype(np.float32)
        red_dominant = (r_f > g_f * 1.3 + 10) & (r_f > b_f * 1.3 + 10) & (r > DIFF_THRESHOLD)
        gray = r.copy()
        gray[~red_dominant] = 0

    _, thresh = cv2.threshold(gray, DIFF_THRESHOLD, 255, cv2.THRESH_BINARY)
    thresh = cv2.morphologyEx(thresh, cv2.MORPH_CLOSE,
                              cv2.getStructuringElement(cv2.MORPH_ELLIPSE, (5, 5)))
    contours, _ = cv2.findContours(thresh, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)
    if not contours:
        return None
    best_cx, best_cy, best_brightness = 0.0, 0.0, 0.0
    for cnt in contours:
        area = cv2.contourArea(cnt)
        if area < DIFF_MIN_AREA or area > DIFF_MAX_AREA:
            continue
        m = cv2.moments(cnt)
        if m["m00"] < 1e-6:
            continue
        cx = m["m10"] / m["m00"]
        cy = m["m01"] / m["m00"]
        bx, by, bw, bh = cv2.boundingRect(cnt)
        peak = float(gray[by:by + bh, bx:bx + bw].max()) if bw > 0 and bh > 0 else 0.0
        if peak > best_brightness:
            best_brightness = peak
            best_cx, best_cy = cx, cy
    return (best_cx, best_cy) if best_brightness > DIFF_THRESHOLD else None


def detect_laser_differential(
    ser: Transport,
    cap: cv2.VideoCapture,
    pan: float,
    tilt: float,
    seq: int,
    settle_ms: int = DIFF_SETTLE_MS,
    n_frames: int = DIFF_FRAMES,
) -> tuple[tuple[float, float] | None, int]:
    """Differential laser detection: capture OFF frame, ON frame, subtract."""
    use_gpu = _check_gpu()

    # Laser OFF: capture background
    start = time.perf_counter()
    while (time.perf_counter() - start) * 1000 < settle_ms:
        send_angles(ser, pan, tilt, seq, laser_on=False)
        seq += 1
        cap.grab()
        time.sleep(SEND_INTERVAL_MS / 1000)
    for _ in range(3):
        cap.grab()
    bg = _capture_median_frame(cap, n_frames)

    # Laser ON: capture foreground
    start = time.perf_counter()
    while (time.perf_counter() - start) * 1000 < settle_ms:
        send_angles(ser, pan, tilt, seq, laser_on=True)
        seq += 1
        cap.grab()
        time.sleep(SEND_INTERVAL_MS / 1000)
    for _ in range(3):
        cap.grab()
    fg = _capture_median_frame(cap, n_frames)

    if bg is None or fg is None:
        return None, seq

    if use_gpu:
        try:
            g_bg = cv2.cuda_GpuMat(); g_fg = cv2.cuda_GpuMat()
            g_bg.upload(bg); g_fg.upload(fg)
            diff = cv2.cuda.absdiff(g_fg, g_bg).download()
        except Exception:
            diff = cv2.absdiff(fg, bg)
    else:
        diff = cv2.absdiff(fg, bg)

    result = _diff_find_blob(diff)
    return result, seq


# ═══════════════════════════════════════════════════════════════════════
#  UI
# ═══════════════════════════════════════════════════════════════════════

def draw_center_ui(frame: np.ndarray, pan: float, tilt: float,
                   laser_px: tuple[float, float] | None,
                   error_px: tuple[float, float] | None,
                   label: str) -> np.ndarray:
    """Overlay calibration UI on camera frame."""
    display = frame.copy()
    h, w = display.shape[:2]
    cx, cy = w // 2, h // 2

    # Crosshair at optical center
    cv2.line(display, (cx - 30, cy), (cx + 30, cy), (0, 255, 255), 1)
    cv2.line(display, (cx, cy - 30), (cx, cy + 30), (0, 255, 255), 1)
    cv2.circle(display, (cx, cy), 6, (0, 255, 255), 1)

    # Laser dot
    if laser_px is not None:
        lx, ly = int(laser_px[0]), int(laser_px[1])
        cv2.circle(display, (lx, ly), 14, (0, 255, 0), 2)
        cv2.circle(display, (lx, ly), 2, (0, 255, 0), -1)
        cv2.line(display, (cx, cy), (lx, ly), (0, 100, 255), 1)

    # Status bar (top)
    cv2.rectangle(display, (0, 0), (w, 55), (0, 0, 0), -1)
    cv2.putText(display, f"CALIBRATION  [{label}]", (10, 18),
                cv2.FONT_HERSHEY_SIMPLEX, 0.5, (0, 255, 255), 1)
    cv2.putText(display, f"Pan:{pan:+.2f}  Tilt:{tilt:+.2f}", (10, 38),
                cv2.FONT_HERSHEY_SIMPLEX, 0.45, (255, 255, 255), 1)
    if error_px is not None:
        err_mag = (error_px[0] ** 2 + error_px[1] ** 2) ** 0.5
        color = (0, 255, 0) if err_mag < 5 else (0, 200, 255) if err_mag < 20 else (0, 100, 255)
        cv2.putText(display, f"Error: ({error_px[0]:+.1f}, {error_px[1]:+.1f}) px  [{err_mag:.0f}]",
                    (280, 38), cv2.FONT_HERSHEY_SIMPLEX, 0.45, color, 1)
    if laser_px is None:
        cv2.putText(display, "NO LASER DETECTED", (w // 2 - 100, h - 15),
                    cv2.FONT_HERSHEY_SIMPLEX, 0.6, (0, 0, 255), 2)

    # Help bar (bottom)
    cv2.putText(display, "ENTER:auto-refine | Arrows/ZQSD:manual jog | ESC:done",
                (10, h - 5), cv2.FONT_HERSHEY_SIMPLEX, 0.35, (150, 150, 150), 1)
    return display


# ═══════════════════════════════════════════════════════════════════════
#  Calibration
# ═══════════════════════════════════════════════════════════════════════

def calibrate(ser: Transport, cap: cv2.VideoCapture, auto: bool = False) -> dict | None:
    """Main calibration routine: manual jog + auto-refine, then save offsets."""
    current_offsets = read_current_offsets(ser)
    if current_offsets is None:
        print("  ERROR: failed to read current firmware offsets over serial.")
        return None

    base_pan_offset, base_tilt_offset = current_offsets

    detector = make_laser_detector()
    fov, invert_pan, invert_tilt = load_config_values()

    pan, tilt, seq = 0.0, 0.0, 0
    img_w = int(cap.get(cv2.CAP_PROP_FRAME_WIDTH))
    img_h = int(cap.get(cv2.CAP_PROP_FRAME_HEIGHT))
    cx_target, cy_target = img_w / 2.0, img_h / 2.0
    fx_approx = (img_w / 2.0) / tan(radians(fov / 2.0))

    # Direction signs from config (skip costly probing)
    pan_sign = -1.0 if invert_pan else 1.0
    tilt_sign = -1.0 if invert_tilt else 1.0

    win = "Calibration"
    cv2.namedWindow(win, cv2.WINDOW_AUTOSIZE)

    if auto:
        print("\n  AUTO CALIBRATION")
        print("  Waiting for laser detection...")
    else:
        print("\n  CENTER CALIBRATION")
        print("  Laser ON. Jog to center, then ENTER for auto-refine.\n")
    print(f"  Current firmware trim: pan={base_pan_offset:+.3f} deg  tilt={base_tilt_offset:+.3f} deg")

    label = "AUTO" if auto else "MANUAL JOG"
    laser_px: tuple[float, float] | None = None
    error_px: tuple[float, float] | None = None

    try:
        if auto:
            # Wait for laser to be visible, then auto-refine immediately
            for _ in range(150):  # ~5s at 30fps
                send_angles(ser, pan, tilt, seq, laser_on=True); seq += 1
                ret, frame = cap.read()
                if not ret:
                    time.sleep(0.01)
                    continue
                result = detector.detect(frame, roi_center=None)
                if result is not None:
                    laser_px = result.center
                    error_px = (laser_px[0] - cx_target, laser_px[1] - cy_target)
                    cv2.imshow(win, draw_center_ui(frame, pan, tilt, laser_px, error_px, label))
                    cv2.waitKey(1)
                    print(f"  Laser found at ({laser_px[0]:.0f}, {laser_px[1]:.0f})")
                    pan, tilt, seq = _auto_refine(
                        ser, cap, detector, pan, tilt, seq,
                        cx_target, cy_target, fx_approx, pan_sign, tilt_sign, win,
                    )
                    label = "DONE"
                    break
                cv2.imshow(win, draw_center_ui(frame, pan, tilt, None, None, "WAITING"))
                cv2.waitKey(16)
            else:
                print("  Timeout: no laser detected in 5 seconds.")
                cv2.destroyWindow(win)
                return None
        else:
            # Manual jog loop
            while True:
                send_angles(ser, pan, tilt, seq, laser_on=True); seq += 1

                ret, frame = cap.read()
                if not ret:
                    time.sleep(0.01)
                    continue

                result = detector.detect(frame, roi_center=None)
                if result is not None:
                    laser_px = result.center
                    error_px = (laser_px[0] - cx_target, laser_px[1] - cy_target)
                else:
                    laser_px = None
                    error_px = None

                cv2.imshow(win, draw_center_ui(frame, pan, tilt, laser_px, error_px, label))
                raw = cv2.waitKeyEx(16)
                if raw == -1:
                    continue
                key = raw & 0xFF

                if key == 27:  # ESC — accept current
                    break
                elif key == 13:  # ENTER — auto-refine
                    if laser_px is None:
                        print("  No laser detected. Make sure the laser is on and visible.")
                        continue
                    pan, tilt, seq = _auto_refine(
                        ser, cap, detector, pan, tilt, seq,
                        cx_target, cy_target, fx_approx, pan_sign, tilt_sign, win,
                    )
                    label = "DONE"
                    print(f"  Auto-refine complete: pan={pan:+.3f}  tilt={tilt:+.3f}")
                elif key == ord("c"):
                    pan, tilt = 0.0, 0.0
                else:
                    dp, dt = parse_jog_key(raw)
                    pan += dp
                    tilt += dt
                    if dp != 0 or dt != 0:
                        label = "MANUAL JOG"
    finally:
        cv2.destroyWindow(win)

    return _save_offsets(ser, base_pan_offset + pan, base_tilt_offset + tilt)


def _auto_refine(
    ser: Transport,
    cap: cv2.VideoCapture,
    detector: LaserDetector,
    pan: float,
    tilt: float,
    seq: int,
    cx_target: float,
    cy_target: float,
    fx_approx: float,
    pan_sign: float,
    tilt_sign: float,
    win: str,
    max_iter: int = 30,
    threshold_px: float = 3.0,
    gain: float = 0.6,
) -> tuple[float, float, int]:
    """Iterative feedback loop: adjust pan/tilt until laser is at camera center.

    Uses HSV detection first (fast, ~16ms/iter). Falls back to differential
    ON/OFF detection after 3 consecutive HSV misses (~1.2s/iter).
    """
    print("  Auto-refining...")
    hsv_miss_streak = 0
    HSV_MISS_FALLBACK = 3

    for i in range(max_iter):
        detection: tuple[float, float] | None = None
        frame: np.ndarray | None = None

        # Try HSV first (fast)
        if hsv_miss_streak < HSV_MISS_FALLBACK:
            send_angles(ser, pan, tilt, seq, laser_on=True); seq += 1
            ret, frame = cap.read()
            if ret:
                result = detector.detect(frame, roi_center=None)
                if result is not None:
                    detection = result.center
                    hsv_miss_streak = 0
                else:
                    hsv_miss_streak += 1
            else:
                hsv_miss_streak += 1
                frame = None

        # Fall back to differential (slow but reliable)
        if detection is None:
            detection, seq = detect_laser_differential(ser, cap, pan, tilt, seq)
            if detection is not None:
                hsv_miss_streak = 0
                # Grab a frame for display
                send_angles(ser, pan, tilt, seq, laser_on=True); seq += 1
                ret, frame = cap.read()
                if not ret:
                    frame = None

        if detection is None:
            print(f"    iter {i}: lost laser, stopping.")
            break

        ex = detection[0] - cx_target
        ey = detection[1] - cy_target
        err_mag = (ex * ex + ey * ey) ** 0.5
        method = "HSV" if hsv_miss_streak == 0 else "DIFF"
        print(f"    iter {i}: error=({ex:+.1f}, {ey:+.1f}) px  mag={err_mag:.1f}  [{method}]")

        # Show live feedback
        if frame is not None:
            cv2.imshow(win, draw_center_ui(frame, pan, tilt, detection, (ex, ey), f"AUTO {i+1}/{max_iter}"))
            cv2.waitKey(1)

        if err_mag < threshold_px:
            print(f"  Converged in {i + 1} iterations ({err_mag:.1f} px error)")
            break

        # Adjust gimbal to reduce error
        pan -= gain * pan_sign * (ex / fx_approx) * degrees(1.0)
        tilt -= gain * tilt_sign * (ey / fx_approx) * degrees(1.0)

    return pan, tilt, seq


def _save_offsets(ser: Transport, pan: float, tilt: float) -> dict | None:
    """Save calibrated offsets to servo_limits.json and firmware NVS."""
    if not write_offsets_and_confirm(ser, pan, tilt):
        print("  ERROR: firmware did not acknowledge the new offsets; nothing was saved.")
        return None

    # servo_limits.json (backward compat)
    try:
        existing = json.loads(LIMITS_PATH.read_text(encoding="utf-8"))
    except (FileNotFoundError, json.JSONDecodeError):
        existing = {}
    existing["center_pan_offset_deg"] = round(pan, 3)
    existing["center_tilt_offset_deg"] = round(tilt, 3)
    save_limits_json(existing)

    print(f"\n  Offsets applied: pan={pan:+.3f} deg  tilt={tilt:+.3f} deg")
    print("    -> calibration_data/servo_limits.json updated")
    print("    -> firmware NVS updated")
    return {"center_pan": pan, "center_tilt": tilt}


def reset_offsets(ser: Transport) -> bool:
    """Zero all offsets in servo_limits.json and firmware NVS."""
    if not write_offsets_and_confirm(ser, 0.0, 0.0):
        print("  ERROR: firmware did not acknowledge zeroed offsets; reset aborted.")
        return False

    try:
        existing = json.loads(LIMITS_PATH.read_text(encoding="utf-8"))
    except (FileNotFoundError, json.JSONDecodeError):
        existing = {}
    existing["center_pan_offset_deg"] = 0.0
    existing["center_tilt_offset_deg"] = 0.0
    save_limits_json(existing)

    print("  All offsets reset to zero.")
    print("    -> calibration_data/servo_limits.json zeroed")
    print("    -> firmware NVS zeroed")
    return True


# ═══════════════════════════════════════════════════════════════════════
#  Camera / Transport / Main
# ═══════════════════════════════════════════════════════════════════════

def open_transport(args: argparse.Namespace) -> Transport:
    return SerialTransport(args.port, args.baud)


def open_camera(source: str, backend: str) -> cv2.VideoCapture:
    _backends = {
        "dshow": cv2.CAP_DSHOW,
        "msmf": cv2.CAP_MSMF,
        "ffmpeg": cv2.CAP_FFMPEG,
        "auto": cv2.CAP_ANY,
    }
    src = int(source) if source.isdigit() else source
    cap = cv2.VideoCapture(src, _backends.get(backend, cv2.CAP_ANY))
    cap.set(cv2.CAP_PROP_FPS, 60)
    if not cap.isOpened():
        print(f"  ERROR: Cannot open camera '{source}'")
        sys.exit(1)
    w = int(cap.get(cv2.CAP_PROP_FRAME_WIDTH))
    h = int(cap.get(cv2.CAP_PROP_FRAME_HEIGHT))
    print(f"  Camera: {w}x{h} native feed")
    return cap


def _kill_previous_instance() -> None:
    """Kill any previous calibration/tracker instance using PID lock files."""
    for lock in [Path("logs/.calibrate.pid"), Path("logs/.tracker.pid")]:
        if lock.exists():
            try:
                old_pid = int(lock.read_text().strip())
                if old_pid != os.getpid():
                    os.kill(old_pid, signal.SIGTERM)
                    time.sleep(0.5)
                    try:
                        os.kill(old_pid, signal.SIGTERM)
                    except OSError:
                        pass
                    print(f"  Killed previous instance (PID {old_pid})")
            except (ValueError, OSError):
                pass
    Path("logs").mkdir(parents=True, exist_ok=True)
    Path("logs/.calibrate.pid").write_text(str(os.getpid()))


def main() -> None:
    parser = argparse.ArgumentParser(description="Gimbal center calibration tool")
    parser.add_argument("--port", default="COM4", help="Serial port")
    parser.add_argument("--baud", type=int, default=921600, help="Baud rate")
    parser.add_argument("--source", default="0", help="Camera source")
    parser.add_argument("--backend", default="auto", choices=("dshow", "msmf", "ffmpeg", "auto"))
    parser.add_argument("--auto", action="store_true", help="Skip manual jog, auto-converge immediately")
    parser.add_argument("--reset", action="store_true", help="Reset all offsets to zero and exit")
    args = parser.parse_args()

    print()
    print("  " + "=" * 52)
    print("    GIMBAL CENTER CALIBRATION")
    print("  " + "=" * 52)

    gpu = _check_gpu()
    print(f"  GPU acceleration: {'ENABLED' if gpu else 'CPU fallback'}")

    ser = open_transport(args)
    _kill_previous_instance()

    try:
        if args.reset:
            if not reset_offsets(ser):
                sys.exit(1)
        else:
            cap = open_camera(args.source, args.backend)
            try:
                seq = 0
                result = calibrate(ser, cap, auto=args.auto)
                if result is None:
                    sys.exit(1)
                print("\n  Center calibration stage finished.\n")

                print("  Holding gimbal at centered reference ...")
                hold_position(ser, 0.0, 0.0, duration_ms=3000, seq=seq)
            finally:
                cap.release()
                cv2.destroyAllWindows()
    finally:
        ser.close()
        Path("logs/.calibrate.pid").unlink(missing_ok=True)


if __name__ == "__main__":
    main()
