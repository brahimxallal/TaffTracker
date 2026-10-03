#pragma once
#include <stdint.h>
#include <stdbool.h>
#include "esp_err.h"
int64_t esp_timer_get_time(void);                 /* mocked by the harness */
typedef void *esp_timer_handle_t;
typedef enum { ESP_TIMER_TASK = 0 } esp_timer_dispatch_t;
typedef struct {
    void (*callback)(void *);
    void *arg;
    esp_timer_dispatch_t dispatch_method;
    const char *name;
    bool skip_unhandled_events;
} esp_timer_create_args_t;
static inline esp_err_t esp_timer_create(const esp_timer_create_args_t *a, esp_timer_handle_t *h) { (void)a; *h = (void *)1; return ESP_OK; }
static inline esp_err_t esp_timer_start_periodic(esp_timer_handle_t h, uint64_t p) { (void)h; (void)p; return ESP_OK; }
static inline esp_err_t esp_timer_stop(esp_timer_handle_t h) { (void)h; return ESP_OK; }
static inline esp_err_t esp_timer_delete(esp_timer_handle_t h) { (void)h; return ESP_OK; }
