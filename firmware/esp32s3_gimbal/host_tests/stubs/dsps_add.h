#pragma once
#include "esp_err.h"
static inline esp_err_t dsps_add_f32(const float *a, const float *b, float *out, int len, int s1, int s2, int so)
{ for (int i = 0; i < len; i++) out[i * so] = a[i * s1] + b[i * s2]; return ESP_OK; }
