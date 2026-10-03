#pragma once
#include <stdint.h>
#include "esp_err.h"
typedef int nvs_handle_t;
#define NVS_READONLY 0
#define NVS_READWRITE 1
static inline esp_err_t nvs_open(const char *ns, int mode, nvs_handle_t *h) { (void)ns; (void)mode; *h = 1; return ESP_OK; }
static inline esp_err_t nvs_get_i16(nvs_handle_t h, const char *k, int16_t *v) { (void)h; (void)k; (void)v; return ESP_FAIL; }
static inline esp_err_t nvs_set_i16(nvs_handle_t h, const char *k, int16_t v) { (void)h; (void)k; (void)v; return ESP_OK; }
static inline esp_err_t nvs_erase_key(nvs_handle_t h, const char *k) { (void)h; (void)k; return ESP_OK; }
static inline esp_err_t nvs_commit(nvs_handle_t h) { (void)h; return ESP_OK; }
static inline void nvs_close(nvs_handle_t h) { (void)h; }
static inline esp_err_t nvs_flash_init(void) { return ESP_OK; }
static inline esp_err_t nvs_flash_erase(void) { return ESP_OK; }
