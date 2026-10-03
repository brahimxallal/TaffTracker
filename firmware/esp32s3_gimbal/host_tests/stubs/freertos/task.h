#pragma once
#include "freertos/FreeRTOS.h"
typedef void (*TaskFunction_t)(void *);
static inline void vTaskDelay(TickType_t t) { (void)t; }
static inline void vTaskDelete(void *h) { (void)h; }
static inline BaseType_t xTaskCreatePinnedToCore(TaskFunction_t f, const char *n, uint32_t s, void *a, int p, void *h, int c)
{ (void)f; (void)n; (void)s; (void)a; (void)p; (void)h; (void)c; return pdPASS; }
