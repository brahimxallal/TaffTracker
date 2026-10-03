#pragma once
#include <stdint.h>
#include "esp_err.h"
typedef struct { int dummy; } wifi_init_config_t;
#define WIFI_INIT_CONFIG_DEFAULT() { 0 }
typedef struct { struct { uint8_t ssid[32]; uint8_t password[64]; } sta; } wifi_config_t;
typedef enum { WIFI_IF_STA = 0 } wifi_interface_t;
typedef enum { WIFI_MODE_STA = 1 } wifi_mode_t;
typedef enum { WIFI_PS_NONE = 0 } wifi_ps_type_t;
enum { WIFI_EVENT_STA_DISCONNECTED = 5 };
static inline esp_err_t esp_wifi_init(const wifi_init_config_t *c) { (void)c; return ESP_OK; }
static inline esp_err_t esp_wifi_set_mode(wifi_mode_t m) { (void)m; return ESP_OK; }
static inline esp_err_t esp_wifi_set_config(wifi_interface_t i, wifi_config_t *c) { (void)i; (void)c; return ESP_OK; }
static inline esp_err_t esp_wifi_start(void) { return ESP_OK; }
static inline esp_err_t esp_wifi_set_ps(wifi_ps_type_t t) { (void)t; return ESP_OK; }
static inline esp_err_t esp_wifi_connect(void) { return ESP_OK; }
