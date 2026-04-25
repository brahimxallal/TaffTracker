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
that runs on the velocity *before* the lead multiplication. Disabled by
default so existing deployments are bit-identical until you opt in.

### Activation procedure

1. **Pick starting parameters** with the analyzer (no hardware needed):
   ```bash
   python scripts/analyze_velocity_smoother.py
   ```
   Read the recommended `alpha` and `deadband_dps`. The defaults in
   `config.yaml` are already set to the analyzer's recommendation; only
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
     predictive_lead_s: 0.08  # default; raise for slower servos, lower for faster
   ```
   The smoother is harmless when `predictive_lead_s == 0` (lead path is
   skipped entirely), so this step is required for the smoother to have
   any visible effect.

3. **Enable the smoother**:
   ```yaml
   servo_control:
     velocity_smoother_enabled: true
     velocity_smoother_alpha: 0.5         # from analyzer
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
back to using the raw velocity and behavior matches the pre-smoother
build exactly.

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
**DISABLED** because earlier live tuning (Phase Q) showed it amplified
PC-side velocity noise into servo jitter.

The PC-side velocity smoother (Section 1 above) addresses that root
cause. Re-enabling firmware lead is a future change that requires:

1. PC-side smoother enabled and validated for at least one full test
   session.
2. Firmware change to set the lead time non-zero.
3. Re-flash and re-test, watching for the same jitter symptoms that
   triggered the original disable.

This is not done in any current commit. Tracked as a follow-up in the
project memory's "Performance Frontier" list.

---

## Quick-reference: where each tunable lives

| What | Where | Default |
|---|---|---|
| PC-side lead time | `gimbal.predictive_lead_s` (config.yaml) | 0.08 s |
| PC-side velocity smoother enable | `servo_control.velocity_smoother_enabled` | false |
| PC-side smoother alpha | `servo_control.velocity_smoother_alpha` | 0.5 |
| PC-side smoother deadband | `servo_control.velocity_smoother_deadband_dps` | 5.0 dps |
| GPU letterbox | `runtime.gpu_preprocess` | false |
| Firmware lead | `LEAD_COMPENSATION_*` in `control_task.c` | disabled |

All flags are safe-by-default and require a process restart to take
effect.
