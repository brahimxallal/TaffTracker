#pragma once
#include <stdint.h>
#include "esp_err.h"
esp_err_t gpio_set_level(int gpio_num, uint32_t level);   /* mocked by the harness */
