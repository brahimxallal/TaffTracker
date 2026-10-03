#pragma once
#include <stdint.h>
#include <stdbool.h>
typedef int BaseType_t;
typedef uint32_t TickType_t;
typedef uint32_t EventBits_t;
typedef int portMUX_TYPE;
#define portMUX_INITIALIZER_UNLOCKED 0
#define portENTER_CRITICAL(m) ((void)(m))
#define portEXIT_CRITICAL(m) ((void)(m))
#define pdMS_TO_TICKS(ms) ((TickType_t)(ms))
#define pdPASS 1
#define pdFALSE 0
#define pdTRUE 1
#define BIT0 1u
