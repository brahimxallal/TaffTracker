#pragma once
typedef int esp_err_t;
#define ESP_OK 0
#define ESP_FAIL (-1)
#define ESP_ERR_NVS_NO_FREE_PAGES 0x1102
#define ESP_ERR_NVS_NEW_VERSION_FOUND 0x1110
#define ESP_ERR_ESP_NETIF_DHCP_ALREADY_STOPPED 0x5008
static inline const char *esp_err_to_name(esp_err_t e) { (void)e; return "ERR"; }
