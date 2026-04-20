"""Diagnostic: visualize what the laser detector sees in real-time.

Shows 4 panels:
  Top-left:     Raw camera feed with detected blobs circled
  Top-right:    HSV Value channel (brightness) —_reshold live.

Reads laser thresholds from config.yaml automatically.
"""
import sys
from pathlib import Path

import cv2
import numpy as np
import yaml

# Load thresholds from config.yaml (fall back to sane defaults)
_CONFIG_PATH = Path(__file__).resolve().parent.parent / "config.yaml"
_laser_cfg: dict = {}
if _CONFIG_PATH.exists():
    with open(_CONFIG_PATH, "r", encoding="utf-8") as f:
        _data = yaml.safe_load(f)
        if isinstance(_data, dict):
            _laser_cfg = _data.get("laser", {})

VAL_MIN = int(_laser_cfg.get("val_min", 160))
SAT_MIN = int(_laser_cfg.get("sat_min", 120))
HUE_LOW = int(_laser_cfg.get("hue_low_upper", 5))
HUE_HIGH = int(_laser_cfg.get("hue_high_lower", 175))
MIN_AREA = float(_laser_cfg.get("min_area", 2.0))
MAX_AREA = float(_laser_cfg.get("max_area", 800.0))

cap = cv2.VideoCapture(0)
if not cap.isOpened():
    print("ERROR: cannot open camera")
    sys.exit(1)

cap.set(cv2.CAP_PROP_FRAME_WIDTH, 640)
cap.set(cv2.CAP_PROP_FRAME_HEIGHT, 480)

print("Laser Debug Tool")
print("  +/- : adjust brightness threshold (V)")
print("  s/a : adjust saturation threshold (S)")
print("  q   : quit")
print(f"  Starting: V_min={VAL_MIN}, S_min={SAT_MIN}")

while True:
    ret, frame = cap.read()
    if not ret:
        continue

    hsv = cv2.cvtColor(frame, cv2.COLOR_BGR2HSV)
    h, s, v = cv2.split(hsv)

    # Red mask (dual range)
    lo_mask = cv2.inRange(hsv, np.array([0, SAT_MIN, VAL_MIN]), np.array([HUE_LOW, 255, 255]))
    hi_mask = cv2.inRange(hsv, np.array([HUE_HIGH, SAT_MIN, VAL_MIN]), np.array([180, 255, 255]))
    raw_mask = lo_mask | hi_mask

    # Morphology
    kernel = cv2.getStructuringElement(cv2.MORPH_ELLIPSE, (3, 3))
    clean_mask = cv2.morphologyEx(raw_mask, cv2.MORPH_OPEN, kernel)
    clean_mask = cv2.morphologyEx(clean_mask, cv2.MORPH_CLOSE, kernel)

    # Find contours
    contours, _ = cv2.findContours(clean_mask, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)

    # Annotate raw feed
    display = frame.copy()
    info_mask = cv2.cvtColor(clean_mask, cv2.COLOR_GRAY2BGR)
    blob_count = 0
    for cnt in contours:
        area = cv2.contourArea(cnt)
        perim = cv2.arcLength(cnt, True)
        circ = (4.0 * 3.14159 * area / (perim * perim)) if perim > 0 else 0
        m = cv2.moments(cnt)
        if m["m00"] < 1e-6:
            continue
        cx = int(m["m10"] / m["m00"])
        cy = int(m["m01"] / m["m00"])

        # Draw ALL detected red blobs (even rejected ones)
        color = (0, 255, 0)  # green = passes filter
        label = f"A={area:.0f} C={circ:.2f}"
        if area < MIN_AREA or area > MAX_AREA:
            color = (0, 165, 255)  # orange = rejected by area
            label += " [area]"
        elif circ < 0.3:
            color = (0, 255, 255)  # yellow = rejected by circularity
            label += " [circ]"
        else:
            blob_count += 1

        cv2.circle(display, (cx, cy), 10, color, 2)
        cv2.putText(display, label, (cx + 12, cy - 5), cv2.FONT_HERSHEY_SIMPLEX, 0.4, color, 1)
        cv2.circle(info_mask, (cx, cy), 10, color, 2)

    # Value channel as grayscale BGR
    v_display = cv2.cvtColor(v, cv2.COLOR_GRAY2BGR)
    raw_mask_display = cv2.cvtColor(raw_mask, cv2.COLOR_GRAY2BGR)

    # Add text overlays
    cv2.putText(display, f"V>={VAL_MIN} S>={SAT_MIN} Blobs:{blob_count}", (10, 25),
                cv2.FONT_HERSHEY_SIMPLEX, 0.6, (0, 255, 0), 2)
    cv2.putText(v_display, "Value (brightness)", (10, 25),
                cv2.FONT_HERSHEY_SIMPLEX, 0.6, (0, 255, 0), 2)
    cv2.putText(raw_mask_display, "Red hue mask (raw)", (10, 25),
                cv2.FONT_HERSHEY_SIMPLEX, 0.6, (0, 255, 0), 2)
    cv2.putText(info_mask, "Final mask + contours", (10, 25),
                cv2.FONT_HERSHEY_SIMPLEX, 0.6, (0, 255, 0), 2)

    # 2x2 grid
    top = np.hstack([display, v_display])
    bottom = np.hstack([raw_mask_display, info_mask])
    grid = np.vstack([top, bottom])
    grid = cv2.resize(grid, (1280, 960))

    cv2.imshow("Laser Debug (q=quit, +/-=V, s/a=S)", grid)
    key = cv2.waitKey(1) & 0xFF
    if key == ord('q'):
        break
    elif key == ord('+') or key == ord('='):
        VAL_MIN = min(255, VAL_MIN + 10)
        print(f"  V_min = {VAL_MIN}")
    elif key == ord('-'):
        VAL_MIN = max(0, VAL_MIN - 10)
        print(f"  V_min = {VAL_MIN}")
    elif key == ord('s'):
        SAT_MIN = min(255, SAT_MIN + 10)
        print(f"  S_min = {SAT_MIN}")
    elif key == ord('a'):
        SAT_MIN = max(0, SAT_MIN - 10)
        print(f"  S_min = {SAT_MIN}")

cap.release()
cv2.destroyAllWindows()
print(f"\nFinal thresholds: V_min={VAL_MIN}, S_min={SAT_MIN}")
print("Update config.yaml with these values if different from defaults.")
