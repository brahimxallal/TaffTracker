#pragma once

#include "esp_err.h"

/**
 * Initialise two LEDC channels for pan/tilt servo on GPIO 11/12.
 * Both servos driven to center (90 deg) on init.
 */
esp_err_t servo_init(void);

/**
 * Set servo angle (0-180 degrees). Clamps out-of-range values.
 * @param channel 0 = pan, 1 = tilt
 */
void servo_set_angle(int channel, float degrees);
