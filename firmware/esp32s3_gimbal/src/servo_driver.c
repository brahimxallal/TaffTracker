#include "servo_driver.h"
#include "config.h"
#include "driver/ledc.h"
#include "esp_log.h"

static const char *TAG = "servo";

static const int gpio_map[2] = { PIN_SERVO_PAN, PIN_SERVO_TILT };

static inline uint32_t angle_to_duty(float deg)
{
    if (deg < 0.0f) deg = 0.0f;
    if (deg > SERVO_RANGE_DEG) deg = SERVO_RANGE_DEG;
    float pulse_us = SERVO_MIN_US + (deg / SERVO_RANGE_DEG) * (SERVO_MAX_US - SERVO_MIN_US);
    return (uint32_t)(pulse_us * LEDC_MAX_DUTY / SERVO_PERIOD_US + 0.5f);
}

esp_err_t servo_init(void)
{
    /* One timer shared by both channels */
    ledc_timer_config_t timer_cfg = {
        .speed_mode      = LEDC_LOW_SPEED_MODE,
        .duty_resolution = LEDC_TIMER_14_BIT,
        .timer_num       = LEDC_TIMER_0,
        .freq_hz         = LEDC_FREQ_HZ,
        .clk_cfg         = LEDC_AUTO_CLK,
    };
    esp_err_t err = ledc_timer_config(&timer_cfg);
    if (err != ESP_OK) {
        ESP_LOGE(TAG, "LEDC timer config failed: %s", esp_err_to_name(err));
        return err;
    }

    uint32_t pan_duty  = angle_to_duty(SERVO_CENTER_DEG + PAN_OFFSET_DEG);
    uint32_t tilt_duty = angle_to_duty(SERVO_CENTER_DEG + TILT_OFFSET_DEG);
    uint32_t duties[2] = { pan_duty, tilt_duty };

    for (int ch = 0; ch < 2; ch++) {
        ledc_channel_config_t ch_cfg = {
            .gpio_num   = gpio_map[ch],
            .speed_mode = LEDC_LOW_SPEED_MODE,
            .channel    = (ledc_channel_t)ch,
            .timer_sel  = LEDC_TIMER_0,
            .duty       = duties[ch],
            .hpoint     = 0,
        };
        err = ledc_channel_config(&ch_cfg);
        if (err != ESP_OK) {
            ESP_LOGE(TAG, "LEDC channel %d config failed: %s", ch, esp_err_to_name(err));
            return err;
        }
    }

    ESP_LOGI(TAG, "Servos initialised — pan=GPIO%d, tilt=GPIO%d, pan_duty=%lu, tilt_duty=%lu",
             PIN_SERVO_PAN, PIN_SERVO_TILT, (unsigned long)pan_duty, (unsigned long)tilt_duty);
    return ESP_OK;
}

void servo_set_angle(int channel, float degrees)
{
    if (channel < 0 || channel > 1) return;
    uint32_t duty = angle_to_duty(degrees);
    ledc_set_duty(LEDC_LOW_SPEED_MODE, (ledc_channel_t)channel, duty);
    ledc_update_duty(LEDC_LOW_SPEED_MODE, (ledc_channel_t)channel);
}
