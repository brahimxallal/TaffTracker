# Live Tuning Guide

Procedures for tuning the gimbal control loop on the deployed hardware.
Each section is self-contained — read only the one you need.

---

## 1. Predictive lead with the velocity smoother

### Background

The auto controller multiplies target angular velocity by
`gimbal.predictive_lead_s` to nudge the commanded angle ahead of where
the target *will be* one frame later. The win is real on fast motion,
but raw Kalman velocity carries enough HF noise that `velocity * lead`
can amplify into command jitter on slow targets.

The fix is `src/output/velocity_smoother.py` — an EMA + deadband filter
that runs on the velocity *before* the lead multiplication. It is off in
the code defaults (`ServoControlConfig.velocity_smoother_enabled = False`);
the shipped `config.yaml` turns it on with values tuned on the author's rig.

### Activation procedure

1. **Pick starting parameters** with the analyzer (no hardware needed):
   ```bash
   python scripts/analyze_velocity_smoother.py
   ```
   Read the recommended `alpha` and `deadband_dps`. The shipped
   `config.yaml` values start from the analyzer's recommendation; only
   change them if your scene's typical motion differs from the script's
   defaults (slow target with HF noise + occasional spikes). Example:
   ```bash
   # Mostly fast targets:
   python scripts/analyze_velocity_smoother.py --drift-amp 200 --drift-period 1.0
   # Mostly static / very slow:
   python scripts/analyze_velocity_smoother.py --drift-amp 30 --drift-period 4.0
   ```

2. **Confirm `predictive_lead_s` is non-zero** in `config.yaml`:
   ```yaml
   gimbal:
     predictive_lead_s: 0.035  # shipped value (code default 0.0 = off); raise for slower servos, lower for faster
   ```
   The smoother is harmless when `predictive_lead_s == 0` (lead path is
   skipped entirely), so this step is required for the smoother to have
   any visible effect.

3. **Enable the smoother**:
   ```yaml
   servo_control:
     velocity_smoother_enabled: true
     velocity_smoother_alpha: 0.45        # from analyzer (code default 0.4)
     velocity_smoother_deadband_dps: 5.0  # snap-to-zero floor for static targets
   ```

4. **Run the live pipeline** and watch for:
   - **Better:** Reduced micro-jitter on the gimbal when the target is
     stationary, no visible aiming lag on fast pans.
   - **Worse:** Visible aiming lag (commanded angle chases the target by
     ~1–2 frames). Lower the alpha — that increases responsiveness at
     the cost of less noise reduction.

5. **If the gimbal oscillates** with the smoother on, raise the deadband
   first. If oscillation continues, drop `predictive_lead_s` by 25% and
   try again.

### Reverting

Set `velocity_smoother_enabled: false` and restart. The lead path goes
back to using the raw velocity, as it did before the smoother existed.

---

## 2. GPU letterbox preprocessing

`runtime.gpu_preprocess` swaps the CPU `cv2.resize` letterbox path for
a PyTorch-on-CUDA implementation. Default is `false`.

### Decide whether to enable

```bash
python scripts/benchmark_preprocess.py
```

The benchmark prints CPU and GPU p50/p95/p99 letterbox timings and a
verdict line. **If p50 speedup is below 1.2×, leave the flag off** —
the H2D transfer cost won't pay back the kernel speedup, and the GPU
path also adds a sync point that can interact poorly with the TensorRT
stream.

The GPU path tends to win on:
- Large source frames (4K capture downscaled to 640×640)
- Hosts with weak CPUs but capable GPUs

It tends to lose on:
- Source already close to the target size (1080p → 640)
- Hosts where the CPU letterbox is already <1 ms

### Enable

```yaml
runtime:
  gpu_preprocess: true
```

Restart the pipeline. There is no live-effect change without the restart
because the flag is read once at `CaptureProcess` construction.

### Reverting

Set `gpu_preprocess: false` and restart.

---

## 3. ESP32 firmware lead compensation (separate from PC lead)

The firmware has its own lead-compensation path in
`firmware/esp32s3_gimbal/src/control_task.c`. It is currently
**DISABLED** because earlier live tuning showed it amplified PC-side
velocity noise into servo jitter.

The PC-side velocity smoother (Section 1 above) addresses that root
cause. Re-enabling firmware lead is a future change that requires:

1. PC-side smoother enabled and validated for at least one full test
   session.
2. Firmware change to set the lead time non-zero.
3. Re-flash and re-test, watching for the same jitter symptoms that
   triggered the original disable.

This has not been done yet.

---

## 4. Relock, command slew limits and backlash (output side)

All of these live under `servo_control` in `config.yaml` and only shape the
command the PC sends; the firmware keeps its own filtering.

### Micro-relock ramp

When the detector drops the *same* track for a very short time and then
finds it again, snapping back to full gain makes the gimbal twitch. If a
lost → measurement transition on the same track ID happens within
`micro_relock_min_loss_s … micro_relock_window_s`, the controller resets its
derivative history and ramps the correction back in over
`micro_relock_ramp_frames` frames, starting at `micro_relock_gain` (0–1).
Longer losses are treated as a normal reacquire (or as search, when search is
enabled).

Relocks that are *not* same-track micro-losses are logged at INFO by
`RelockDiagnostics` (logger `output.relock`) as
`Relock event: frame=… track=… prev=… lost=… err=… cmd_delta=… cmd_vel=…`; same-track losses inside the window
are deliberately silent. If you see many events where `track` equals `prev` and
`lost` is just above `micro_relock_window_s`, the window is a little too narrow
for your camera.

### Command slew caps

`auto_command_slew_limit_dps` and `manual_command_slew_limit_dps` cap how
fast the *final* pan/tilt command may move, applied right before packet
encoding (`0` disables the cap). The velocity sent to the firmware is derived
from the limited command, so feed-forward never asks for more than the cap.
Lower the auto cap if the gimbal overshoots after a relock; raise it if it
trails fast targets. Keep the manual cap above the largest manual speed
(`manual_coarse_speed_dps`), or the keys will feel sluggish.

### Backlash compensation

`backlash_compensation_enabled` adds a short extra pulse when the commanded
direction reverses, to take up slack in the gears. Start with a small
`backlash_pan_deg` / `backlash_tilt_deg`, increase by 0.05° at a time and stop
as soon as reversals no longer hesitate; too much shows up as a visible kick.
`backlash_min_step_deg` is the smallest per-update command change that counts
as movement in a direction (smaller wiggles never trigger a pulse) and
`backlash_pulse_decay_dps` is how fast the pulse fades back to zero.

---

## 5. Lock confidence and detector scheduling

### Lock hysteresis (`tracking`)

- `lock_acquire_confidence`: a *new* lock needs at least this score.
- `lock_keep_confidence`: an existing lock survives down to this score.
- `low_confidence_grace_frames`: after this many weak frames in a row the lock
  ID is dropped.

Weak detections never steer the gimbal. If the locked track is weak but another
visible track clears `lock_acquire_confidence`, the lock moves to it (logged at
INFO as "Switching to high-confidence lock candidate"). Raise the acquire
threshold if the tracker locks onto walls or fixtures; lower the keep
threshold if it lets go of a real target during blur.

### Detector skipping (`tracking.inference_schedule`)

With a healthy lock the detector can be skipped on planned frames, and the
Kalman filter predicts instead:

- `detector_locked_stride`: run YOLO every N-th frame while locked.
  **`1` means YOLO on every frame (no skipping)**, which is what the shipped
  `config.yaml` uses; the code default is `3`.
- `detector_min_skip_confidence`: below this detector confidence, YOLO runs on
  every frame.
- `detector_max_prediction_age_ms`: hard cap on how old the last detector
  result may be before YOLO is forced.
- `detector_skip_enabled: false` turns the whole mechanism off.

Raising the stride frees GPU time at the cost of aim accuracy on fast targets;
the profiler summary (`--profile`) shows the `inf`, `wait` and `age`
percentiles so you can see what you gained.

---

## Quick-reference: where each tunable lives

Code defaults are what you get without a `config.yaml` entry; the last column
is what the repository's `config.yaml` ships with (tuned on the author's
phone-camera rig, so treat it as an example, not a recommendation).

| What | Where (config.yaml) | Code default | Shipped |
|---|---|---|---|
| PC-side lead time | `gimbal.predictive_lead_s` | 0.0 s (off) | 0.035 s |
| Velocity smoother | `servo_control.velocity_smoother_enabled` | false | true |
| Smoother alpha / deadband | `servo_control.velocity_smoother_alpha` / `_deadband_dps` | 0.4 / 5.0 dps | 0.45 / 6.0 dps |
| Auto / manual command slew cap | `servo_control.auto_command_slew_limit_dps` / `manual_command_slew_limit_dps` | 0 (off) / 0 (off) | 130 / 620 dps |
| Micro-relock (min loss, window, gain, ramp frames) | `servo_control.micro_relock_*` | 0.025 s, 0.12 s, 0.35, 4 | same |
| Backlash compensation | `servo_control.backlash_compensation_enabled` | false | true (0.20° pan, 0.06° tilt) |
| Lock acquire / keep confidence | `tracking.lock_acquire_confidence` / `lock_keep_confidence` | 0.65 / 0.45 | 0.65 / 0.45 |
| Detector stride while locked | `tracking.inference_schedule.detector_locked_stride` | 3 | 1 (no skipping) |
| Ring buffer slots | `runtime.ring_buffer_slots` | 8 | 8 |
| GPU letterbox | `runtime.gpu_preprocess` | false | false |
| Relay auto-fire on lock | `relay.auto_fire_enabled` | false | false |
| Firmware lead | `LEAD_COMPENSATION_*` in `control_task.c` | disabled | disabled |

`config.yaml` is read when the tracker starts; restart after editing it.
