#include "control_task.h"
#include "config.h"
#include "comm_task.h"
#include "servo_driver.h"

#include <math.h>
#include <stdbool.h>
#include "driver/gpio.h"
#include "esp_timer.h"
#include "esp_log.h"
#include "sdkconfig.h"

// Include ESP-DSP for SIMD/Vector acceleration
#include "dsps_math.h"
#include "dsps_mulc.h"
#include "dsps_add.h"
#include "dsps_sub.h"

#if !defined(CONFIG_ESP_TIMER_TASK_AFFINITY_CPU1)
#error "esp_timer must be pinned to Core 1 (CONFIG_ESP_TIMER_TASK_AFFINITY_CPU1=y)"
#endif

static const char *TAG = "ctrl";

/* ── Persistent control state ─────────────────────────────────────────── */

static float s_current[2]  = {SERVO_CENTER_DEG, SERVO_CENTER_DEG};
static float s_smoothed[2] = {SERVO_CENTER_DEG, SERVO_CENTER_DEG};
static float s_prev_dp[2]  = {0.0f, 0.0f};
static bool s_laser_on = true;
static bool s_lamp_on = false;
static int64_t s_lamp_last_target_us = 0;  /* last time we saw a target */

#if LEAD_COMP_ENABLED
static float s_prev_vel[2] = {0.0f, 0.0f};
static float s_filtered_accel[2] = {0.0f, 0.0f};
#endif

static esp_timer_handle_t s_timer = NULL;

/* ── Helpers ──────────────────────────────────────────────────────────── */

static inline float clampf(float v, float lo, float hi)
{
    if (v < lo) return lo;
    if (v > hi) return hi;
    return v;
}

static float compute_adaptive_alpha(float speed_dps, uint8_t state, uint8_t quality)
{
    /* Manual mode: near pass-through for responsive control */
    if (quality & QUALITY_FLAG_MANUAL) return 0.95f;

    float alpha;
    if (speed_dps < ALPHA_THRESH_STAT)       alpha = ALPHA_VAL_STAT;
    else if (speed_dps < ALPHA_THRESH_SLOW)  alpha = ALPHA_VAL_SLOW;
    else if (speed_dps < ALPHA_THRESH_MOD)   alpha = ALPHA_VAL_MOD;
    else                                     alpha = ALPHA_VAL_FAST;

    if ((state & STATE_MASK) == STATE_PREDICTION)       alpha *= 0.7f;
    if (state & FLAG_OCCLUSION_RECOVERY)                alpha *= 0.5f;
    if (state & FLAG_FAST_MOTION)                       alpha = fmaxf(alpha, 0.7f);

    return alpha;
}

static float apply_accel_limit(float step, float prev_step, float max_change)
{
    float delta = step - prev_step;
    if (delta >  max_change) delta =  max_change;
    if (delta < -max_change) delta = -max_change;
    return prev_step + delta;
}

/* ── Watchdog helpers ─────────────────────────────────────────────────── */

static void coast_decelerate(float dt_since)
{
    /* Coast: maintain last velocity vector but decelerate towards zero */
    float decay = fmaxf(0.0f, 1.0f - (dt_since - WDT_COAST_S) / (WDT_HOLD_S - WDT_COAST_S));
    
    // s_smoothed = s_current + s_prev_dp * decay
    float decay_step[2];
    dsps_mulc_f32(s_prev_dp, decay_step, 2, decay, 1, 1);
    dsps_add_f32(s_current, decay_step, s_smoothed, 2, 1, 1, 1);

    float dz_u, dzp_u, coast_max_vel, acc_u;
    comm_get_control_params(&dz_u, &dzp_u, &coast_max_vel, &acc_u);
    float max_v = coast_max_vel * DT_CTRL;
    float dp[2];
    dsps_sub_f32(s_smoothed, s_current, dp, 2, 1, 1, 1);
    dp[0] = clampf(dp[0], -max_v, max_v);
    dp[1] = clampf(dp[1], -max_v, max_v);

    dsps_add_f32(s_current, dp, s_current, 2, 1, 1, 1);

    float pan_off, tilt_off;
    comm_get_offsets(&pan_off, &tilt_off);
    servo_set_angle(0, s_current[0] + pan_off);
    servo_set_angle(1, s_current[1] + tilt_off);
}

static void return_to_center(void)
{
    gpio_set_level(PIN_LASER, s_laser_on ? 1 : 0);
    /* Lamp: debounced off — only extinguish after sustained no-target */
    int64_t now_us = esp_timer_get_time();
    if (s_lamp_on && (now_us - s_lamp_last_target_us) > LAMP_OFF_DELAY_US) {
        s_lamp_on = false;
        gpio_set_level(PIN_LAMP, 1);  /* active-low: HIGH = OFF */
    }
    float step = HOME_SPEED_DPS * DT_CTRL;

    if (fabsf(s_current[0] - SERVO_CENTER_DEG) > step)
        s_current[0] += (s_current[0] > SERVO_CENTER_DEG) ? -step : step;
    else
        s_current[0] = SERVO_CENTER_DEG;

    if (fabsf(s_current[1] - SERVO_CENTER_DEG) > step)
        s_current[1] += (s_current[1] > SERVO_CENTER_DEG) ? -step : step;
    else
        s_current[1] = SERVO_CENTER_DEG;

    s_smoothed[0] = s_current[0];
    s_smoothed[1] = s_current[1];
    s_prev_dp[0] = 0.0f;
    s_prev_dp[1] = 0.0f;

    float pan_off, tilt_off;
    comm_get_offsets(&pan_off, &tilt_off);
    servo_set_angle(0, s_current[0] + pan_off);
    servo_set_angle(1, s_current[1] + tilt_off);
}

/* ── Main 200 Hz control callback ─────────────────────────────────────── */

static void control_loop_cb(void *arg)
{
    tracking_command_t cmd;
    if (!comm_get_latest_command(&cmd)) {
        return;  /* No command received yet */
    }

    /* Laser control via protocol bit 6 (default ON until first host command). */
    s_laser_on = (cmd.state & FLAG_LASER_ON) != 0;
    gpio_set_level(PIN_LASER, s_laser_on ? 1 : 0);
    /* Lamp: debounced — ON immediately when target acquired, OFF after 3s delay */
    {
        bool target_now = (cmd.state & FLAG_TARGET_ACQUIRED) != 0;
        int64_t now_lamp = esp_timer_get_time();
        if (target_now) {
            s_lamp_last_target_us = now_lamp;
            if (!s_lamp_on) {
                s_lamp_on = true;
                gpio_set_level(PIN_LAMP, 0);  /* active-low: LOW = ON */
            }
        } else if (s_lamp_on && (now_lamp - s_lamp_last_target_us) > LAMP_OFF_DELAY_US) {
            s_lamp_on = false;
            gpio_set_level(PIN_LAMP, 1);  /* active-low: HIGH = OFF */
        }
    }
    /* Relay control via protocol bit 7 */
    gpio_set_level(PIN_RELAY, (cmd.state & FLAG_RELAY_ON) ? 0 : 1);  /* active-low */

    int64_t now_us = esp_timer_get_time();
    float dt_since = (now_us - cmd.received_at_us) / 1e6f;

    /* ── Watchdog ── */
    if (dt_since > WDT_HOME_S) {
        return_to_center();
        return;
    }
    if (dt_since > WDT_HOLD_S) {
        /* Hold position, do nothing */
        return;
    }
    if (dt_since > WDT_COAST_S) {
        coast_decelerate(dt_since);
        return;
    }

    /* ── Normal operation ── */

    /* Extrapolate target position using velocity and staleness.
     * PC sends a PI-corrected angle offset from center; the integral term on
     * the PC side compensates for the camera-on-gimbal feedback halving. */
    float target[2] = {
        SERVO_CENTER_DEG + cmd.pan_deg  + cmd.pan_vel_dps  * dt_since,
        SERVO_CENTER_DEG + cmd.tilt_deg + cmd.tilt_vel_dps * dt_since
    };

#if LEAD_COMP_ENABLED
    /* Filtered lead compensation: low-pass on acceleration, then feedforward */
    {
        float vel[2] = { cmd.pan_vel_dps, cmd.tilt_vel_dps };
        float raw_accel[2];
        dsps_sub_f32(vel, s_prev_vel, raw_accel, 2, 1, 1, 1);
        /* Low-pass filter on acceleration to suppress noise */
        float alpha_lc = LEAD_COMP_ALPHA;
        float filt_new[2];
        dsps_mulc_f32(raw_accel, filt_new, 2, alpha_lc, 1, 1);
        float filt_old[2];
        dsps_mulc_f32(s_filtered_accel, filt_old, 2, 1.0f - alpha_lc, 1, 1);
        dsps_add_f32(filt_new, filt_old, s_filtered_accel, 2, 1, 1, 1);
        /* Feedforward: add filtered acceleration × gain to target */
        float lead[2];
        dsps_mulc_f32(s_filtered_accel, lead, 2, LEAD_COMP_GAIN, 1, 1);
        dsps_add_f32(target, lead, target, 2, 1, 1, 1);
        s_prev_vel[0] = vel[0];
        s_prev_vel[1] = vel[1];
    }
#endif

    /* Clamp to servo range */
    target[0] = clampf(target[0], 0.0f, SERVO_RANGE_DEG);
    target[1] = clampf(target[1], 0.0f, SERVO_RANGE_DEG);

    /* ── Fetch NVS-backed control parameters (once per loop iteration) ─── */
    float dz_base, dz_pred, ctrl_max_vel, ctrl_max_acc;
    comm_get_control_params(&dz_base, &dz_pred, &ctrl_max_vel, &ctrl_max_acc);

    /* ── Per-state adaptive dead-zone filter (BEFORE EMA) ── */
    {
        float dz = dz_base;
        if ((cmd.state & STATE_MASK) == STATE_PREDICTION) dz = dz_pred;
        if (cmd.state & FLAG_FAST_MOTION) dz = 0.0f;  /* no dead-zone during fast motion */
        if (cmd.quality & QUALITY_FLAG_MANUAL) dz = 0.0f;  /* manual: no dead-zone */
        if (dz > 0.0f) {
            if (fabsf(target[0] - s_current[0]) < dz)
                target[0] = s_current[0];
            if (fabsf(target[1] - s_current[1]) < dz)
                target[1] = s_current[1];
        }
    }

    /* ── Vectorized Adaptive EMA ── */
    float speed = sqrtf(cmd.pan_vel_dps * cmd.pan_vel_dps +
                        cmd.tilt_vel_dps * cmd.tilt_vel_dps);
    float alpha = compute_adaptive_alpha(speed, cmd.state, cmd.quality);
    
    float target_scaled[2];
    float smoothed_scaled[2];
    // target * alpha
    dsps_mulc_f32(target, target_scaled, 2, alpha, 1, 1);
    // smoothed * (1 - alpha)
    dsps_mulc_f32(s_smoothed, smoothed_scaled, 2, 1.0f - alpha, 1, 1);
    // smoothed = target_scaled + smoothed_scaled
    dsps_add_f32(target_scaled, smoothed_scaled, s_smoothed, 2, 1, 1, 1);

    /* ── Velocity limiting ── */
    float vel_mult = 1.0f, acc_mult = 1.0f;
    if (cmd.state & FLAG_FAST_MOTION) { vel_mult = 1.5f; acc_mult = 2.0f; }
    float max_step = ctrl_max_vel * vel_mult * DT_CTRL;
    float dp[2];
    dsps_sub_f32(s_smoothed, s_current, dp, 2, 1, 1, 1);
    dp[0] = clampf(dp[0], -max_step, max_step);
    dp[1] = clampf(dp[1], -max_step, max_step);

    /* ── Acceleration limiting ── */
    float max_accel_step = ctrl_max_acc * acc_mult * DT_CTRL;
    dp[0] = apply_accel_limit(dp[0], s_prev_dp[0], max_accel_step);
    dp[1] = apply_accel_limit(dp[1], s_prev_dp[1], max_accel_step);
    s_prev_dp[0] = dp[0];
    s_prev_dp[1] = dp[1];

    /* ── Update position and drive servos ── */
    dsps_add_f32(s_current, dp, s_current, 2, 1, 1, 1);

    float pan_off, tilt_off;
    comm_get_offsets(&pan_off, &tilt_off);
    servo_set_angle(0, s_current[0] + pan_off);
    servo_set_angle(1, s_current[1] + tilt_off);
}

/* ── Public API ───────────────────────────────────────────────────────── */

esp_err_t control_start(void)
{
    esp_timer_create_args_t timer_args = {
        .callback = control_loop_cb,
        .arg = NULL,
        .dispatch_method = ESP_TIMER_TASK,
        .name = "ctrl_200hz",
        .skip_unhandled_events = true,
    };

    esp_err_t err = esp_timer_create(&timer_args, &s_timer);
    if (err != ESP_OK) {
        ESP_LOGE(TAG, "Failed to create control timer: %s", esp_err_to_name(err));
        return err;
    }

    err = esp_timer_start_periodic(s_timer, CONTROL_PERIOD_US);
    if (err != ESP_OK) {
        ESP_LOGE(TAG, "Failed to start control timer: %s", esp_err_to_name(err));
        return err;
    }

    ESP_LOGI(TAG, "Control loop started at %d Hz with Vector Acceleration", TAFFF_CONTROL_HZ);
    return ESP_OK;
}

void control_stop(void)
{
    if (s_timer) {
        esp_timer_stop(s_timer);
        esp_timer_delete(s_timer);
        s_timer = NULL;
    }
}
