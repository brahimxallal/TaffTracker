#pragma once
#include <stdint.h>
#include "esp_err.h"
typedef struct esp_netif_obj esp_netif_t;
typedef struct { uint32_t addr; } esp_ip4_addr_t;
typedef struct { esp_ip4_addr_t ip; esp_ip4_addr_t netmask; esp_ip4_addr_t gw; } esp_netif_ip_info_t;
typedef struct { struct { union { esp_ip4_addr_t ip4; } u_addr; uint8_t type; } ip; } esp_netif_dns_info_t;
#define ESP_IPADDR_TYPE_V4 0
typedef enum { ESP_NETIF_DNS_MAIN = 0 } esp_netif_dns_type_t;
static inline esp_err_t esp_netif_init(void) { return ESP_OK; }
static inline esp_netif_t *esp_netif_create_default_wifi_sta(void) { return (esp_netif_t *)1; }
static inline esp_err_t esp_netif_str_to_ip4(const char *s, esp_ip4_addr_t *a) { (void)s; (void)a; return ESP_OK; }
static inline esp_err_t esp_netif_dhcpc_stop(esp_netif_t *n) { (void)n; return ESP_OK; }
static inline esp_err_t esp_netif_set_ip_info(esp_netif_t *n, const esp_netif_ip_info_t *i) { (void)n; (void)i; return ESP_OK; }
static inline esp_err_t esp_netif_set_dns_info(esp_netif_t *n, esp_netif_dns_type_t t, esp_netif_dns_info_t *d)
{ (void)n; (void)t; (void)d; return ESP_OK; }
