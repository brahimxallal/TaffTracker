#pragma once
#include <stddef.h>
#include "freertos/FreeRTOS.h"
#include "esp_err.h"
typedef struct { uint32_t rx_buffer_size; uint32_t tx_buffer_size; } usb_serial_jtag_driver_config_t;
int usb_serial_jtag_read_bytes(void *buf, uint32_t length, TickType_t ticks);   /* mocked by the harness */
static inline int usb_serial_jtag_write_bytes(const void *src, size_t size, TickType_t ticks)
{ (void)src; (void)ticks; return (int)size; }
static inline esp_err_t usb_serial_jtag_driver_install(usb_serial_jtag_driver_config_t *c) { (void)c; return ESP_OK; }
