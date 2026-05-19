from __future__ import annotations

from dataclasses import dataclass


@dataclass(frozen=True, slots=True)
class OutputRuntimeTuning:
    version: int
    hold_time_s: float
    gimbal_kp: float
    gimbal_ki: float
    gimbal_kd: float
    gimbal_deadband_deg: float
    gimbal_slew_limit_dps: float
    gimbal_kp_near: float | None
    gimbal_kp_far: float | None
    gimbal_predictive_lead_s: float
