#pragma once

#include "esp_err.h"

/**
 * Start the 200 Hz control loop (esp_timer on Core 1).
 * Reads latest command from comm, applies Layer 3 filters, drives servos.
 */
esp_err_t control_start(void);

/**
 * Stop the control loop.
 */
void control_stop(void);
