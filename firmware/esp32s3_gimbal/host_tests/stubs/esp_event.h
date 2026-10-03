#pragma once
#include <stdint.h>
#include "esp_err.h"
typedef const char *esp_event_base_t;
static esp_event_base_t WIFI_EVENT = "WIFI_EVENT";
static esp_event_base_t IP_EVENT = "IP_EVENT";
#define ESP_EVENT_ANY_ID (-1)
enum { IP_EVENT_STA_GOT_IP = 0 };
typedef void (*esp_event_handler_t)(void *arg, esp_event_base_t base, int32_t id, void *data);
typedef struct { struct { struct { uint32_t addr; } ip; } ip_info; } ip_event_got_ip_t;
#define IPSTR "%u.%u.%u.%u"
#define IP2STR(a) ((unsigned)((a)->addr & 0xff)), ((unsigned)(((a)->addr >> 8) & 0xff)), \
                  ((unsigned)(((a)->addr >> 16) & 0xff)), ((unsigned)(((a)->addr >> 24) & 0xff))
static inline esp_err_t esp_event_loop_create_default(void) { return ESP_OK; }
static inline esp_err_t esp_event_handler_register(esp_event_base_t b, int32_t id, esp_event_handler_t h, void *arg)
{ (void)b; (void)id; (void)h; (void)arg; return ESP_OK; }
