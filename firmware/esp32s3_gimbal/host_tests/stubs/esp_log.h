#pragma once
static inline void esp_log_stub(const char *tag, const char *fmt, ...) __attribute__((format(printf, 2, 3)));
static inline void esp_log_stub(const char *tag, const char *fmt, ...) { (void)tag; (void)fmt; }
#define ESP_LOGI(tag, fmt, ...) esp_log_stub(tag, fmt, ##__VA_ARGS__)
#define ESP_LOGW(tag, fmt, ...) esp_log_stub(tag, fmt, ##__VA_ARGS__)
#define ESP_LOGE(tag, fmt, ...) esp_log_stub(tag, fmt, ##__VA_ARGS__)
