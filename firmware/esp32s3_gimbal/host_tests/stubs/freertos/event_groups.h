#pragma once
#include "freertos/FreeRTOS.h"
typedef void *EventGroupHandle_t;
static inline EventGroupHandle_t xEventGroupCreate(void) { return (void *)1; }
static inline EventBits_t xEventGroupClearBits(EventGroupHandle_t g, EventBits_t b) { (void)g; (void)b; return 0; }
static inline EventBits_t xEventGroupSetBits(EventGroupHandle_t g, EventBits_t b) { (void)g; (void)b; return 0; }
static inline EventBits_t xEventGroupWaitBits(EventGroupHandle_t g, EventBits_t b, BaseType_t c, BaseType_t a, TickType_t t)
{ (void)g; (void)b; (void)c; (void)a; (void)t; return 1; }
