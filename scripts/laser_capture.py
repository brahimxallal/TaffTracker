"""Capture one frame, analyze HSV values of bright red regions, save diagnostic."""
import cv2
import numpy as np
import sys

cap = cv2.VideoCapture(0)
if not cap.isOpened():
    print("ERROR: cannot open camera")
    sys.exit(1)

cap.set(cv2.CAP_PROP_FRAME_WIDTH, 640)
cap.set(cv2.CAP_PROP_FRAME_HEIGHT, 480)

# Grab a few frames to let auto-exposure settle
for _ in range(30):
    cap.read()

ret, frame = cap.read()
cap.release()
if not ret:
    print("ERROR: failed to grab frame")
    sys.exit(1)

cv2.imwrite("logs/laser_frame.png", frame)
print(f"Frame saved: logs/laser_frame.png ({frame.shape[1]}x{frame.shape[0]})")

hsv = cv2.cvtColor(frame, cv2.COLOR_BGR2HSV)
h, s, v = cv2.split(hsv)

# Find the brightest pixel
min_v, max_v, min_loc, max_loc = cv2.minMaxLoc(v)
print(f"\nBrightest pixel: value={max_v} at {max_loc}")
bx, by = max_loc
if 0 <= bx < frame.shape[1] and 0 <= by < frame.shape[0]:
    bh, bs, bv = hsv[by, bx]
    bb, bg, br = frame[by, bx]
    print(f"  HSV: H={bh} S={bs} V={bv}")
    print(f"  BGR: B={bb} G={bg} R={br}")

# Find all pixels above various V thresholds
for thresh in [250, 230, 200, 180, 150, 120]:
    bright_mask = v >= thresh
    count = int(np.sum(bright_mask))
    if count > 0:
        bright_h = h[bright_mask]
        bright_s = s[bright_mask]
        # Check how many are red-ish (H<15 or H>165)
        red_count = int(np.sum((bright_h < 15) | (bright_h > 165)))
        print(f"\nV >= {thresh}: {count} pixels, {red_count} red-ish")
        if red_count > 0:
            red_mask = bright_mask & ((h < 15) | (h > 165))
            rh = h[red_mask]
            rs = s[red_mask]
            rv = v[red_mask]
            print(f"  Red pixels: H=[{rh.min()}-{rh.max()}] S=[{rs.min()}-{rs.max()}] V=[{rv.min()}-{rv.max()}]")
            # Find centroid of red bright region
            ys, xs = np.where(red_mask)
            cx, cy = int(xs.mean()), int(ys.mean())
            print(f"  Centroid: ({cx}, {cy})")
    else:
        print(f"\nV >= {thresh}: 0 pixels")

# Also check: what are the top-10 brightest distinct spots?
print("\n--- Top bright spots (V channel) ---")
v_copy = v.copy()
for i in range(5):
    _, mx, _, loc = cv2.minMaxLoc(v_copy)
    if mx < 100:
        break
    x, y = loc
    bh_val, bs_val, bv_val = hsv[y, x]
    bb_val, bg_val, br_val = frame[y, x]
    print(f"  #{i+1}: pos=({x},{y}) V={mx} H={bh_val} S={bs_val} BGR=({bb_val},{bg_val},{br_val})")
    # Suppress this region
    cv2.circle(v_copy, (x, y), 15, 0, -1)

print("\nDone. Check logs/laser_frame.png to see the captured frame.")
