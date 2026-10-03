#pragma once
#include "esp_err.h"
static inline esp_err_t dsps_mulc_f32(const float *in, float *out, int len, float c, int si, int so)
{ for (int i = 0; i < len; i++) out[i * so] = in[i * si] * c; return ESP_OK; }
