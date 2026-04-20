#pragma once

#include <stdbool.h>
#include <stddef.h>

#include "config.h"
#include "esp_err.h"

/**
 * Parse raw bytes for a valid v2 packet (0xBB header, checksum match).
 * Returns true and populates *out on success.
 */
bool packet_parse(const uint8_t *data, size_t len, tracking_packet_v2_t *out);

/**
 * Convert a parsed packet into a runtime tracking command.
 */
void packet_to_command(const tracking_packet_v2_t *pkt, tracking_command_t *cmd);

/**
 * Start the USB CDC receive task (pinned to Core 0).
 */
esp_err_t comm_usb_start(void);

/**
 * Start the WiFi UDP receive task (pinned to Core 0).
 * Connects to the configured SSID, then listens on UDP_PORT.
 */
esp_err_t comm_wifi_start(void);

/**
 * Read the latest tracking command (thread-safe, lock-free for reader).
 * Returns false if no command has been received yet.
 */
bool comm_get_latest_command(tracking_command_t *out);

/**
 * Initialize NVS and load calibration offsets. Called early in startup.
 * Returns ESP_OK on success.
 */
esp_err_t comm_nvs_init(void);

/**
 * Get the current calibration offsets (degrees). Thread-safe.
 */
void comm_get_offsets(float *pan_deg, float *tilt_deg);

/**
 * Get NVS-backed control parameters. Thread-safe.
 */
void comm_get_control_params(float *dz_deg, float *dzp_deg, float *max_vel, float *max_acc);
