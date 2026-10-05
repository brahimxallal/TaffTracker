#include "comm_task.h"
#include "config.h"
#include "wifi_diag.h"

#include <stdio.h>
#include <string.h>
#include "freertos/FreeRTOS.h"
#include "freertos/task.h"
#include "freertos/event_groups.h"
#include "driver/usb_serial_jtag.h"
#include "esp_log.h"
#include "esp_timer.h"
#include "esp_wifi.h"
#include "esp_event.h"
#include "esp_netif.h"
#include "nvs_flash.h"
#include "lwip/sockets.h"

static const char *TAG = "comm";

/* ── NVS-backed calibration offsets ──────────────────────────────────── */

static float s_pan_offset_deg  = PAN_OFFSET_DEG;   /* compile-time default */
static float s_tilt_offset_deg = TILT_OFFSET_DEG;

/* ── NVS-backed control parameters (runtime-tunable) ─────────────────── */
static float s_dead_zone_deg     = DEAD_ZONE_DEG;
static float s_dead_zone_pred_deg = DEAD_ZONE_PRED_DEG;
static float s_max_vel_dps       = MAX_VEL_DPS;
static float s_max_accel_dpss    = MAX_ACCEL_DPSS;

void comm_get_offsets(float *pan_deg, float *tilt_deg)
{
    *pan_deg  = s_pan_offset_deg;
    *tilt_deg = s_tilt_offset_deg;
}

void comm_get_control_params(float *dz_deg, float *dzp_deg, float *max_vel, float *max_acc)
{
    *dz_deg   = s_dead_zone_deg;
    *dzp_deg  = s_dead_zone_pred_deg;
    *max_vel  = s_max_vel_dps;
    *max_acc  = s_max_accel_dpss;
}

static void nvs_load_offsets(void)
{
    nvs_handle_t h;
    if (nvs_open(NVS_NAMESPACE, NVS_READONLY, &h) == ESP_OK) {
        int16_t pan_cd, tilt_cd;
        if (nvs_get_i16(h, NVS_KEY_PAN_OFFSET, &pan_cd) == ESP_OK &&
            nvs_get_i16(h, NVS_KEY_TILT_OFFSET, &tilt_cd) == ESP_OK) {
            s_pan_offset_deg  = pan_cd / 100.0f;
            s_tilt_offset_deg = tilt_cd / 100.0f;
            ESP_LOGI(TAG, "NVS offsets loaded: pan=%.2f° tilt=%.2f°",
                     s_pan_offset_deg, s_tilt_offset_deg);
        } else {
            ESP_LOGI(TAG, "No NVS offsets found, using defaults: pan=%.1f° tilt=%.1f°",
                     s_pan_offset_deg, s_tilt_offset_deg);
        }
        /* Load runtime-tunable control parameters */
        int16_t val;
        if (nvs_get_i16(h, NVS_KEY_DEAD_ZONE, &val) == ESP_OK)
            s_dead_zone_deg = val / 100.0f;
        if (nvs_get_i16(h, NVS_KEY_DEAD_ZONE_P, &val) == ESP_OK)
            s_dead_zone_pred_deg = val / 100.0f;
        if (nvs_get_i16(h, NVS_KEY_MAX_VEL, &val) == ESP_OK)
            s_max_vel_dps = (float)val;
        if (nvs_get_i16(h, NVS_KEY_MAX_ACCEL, &val) == ESP_OK)
            s_max_accel_dpss = (float)val;
        ESP_LOGI(TAG, "Control params: dz=%.2f° dzp=%.2f° vel=%.0f acc=%.0f",
                 s_dead_zone_deg, s_dead_zone_pred_deg, s_max_vel_dps, s_max_accel_dpss);
        nvs_close(h);
    }
}

static esp_err_t nvs_save_offsets(float pan_deg, float tilt_deg)
{
    nvs_handle_t h;
    esp_err_t err = nvs_open(NVS_NAMESPACE, NVS_READWRITE, &h);
    if (err != ESP_OK) return err;
    int16_t pan_cd  = (int16_t)(pan_deg * 100.0f + 0.5f);
    int16_t tilt_cd = (int16_t)(tilt_deg * 100.0f + 0.5f);
    nvs_set_i16(h, NVS_KEY_PAN_OFFSET, pan_cd);
    nvs_set_i16(h, NVS_KEY_TILT_OFFSET, tilt_cd);
    err = nvs_commit(h);
    nvs_close(h);
    if (err == ESP_OK) {
        s_pan_offset_deg  = pan_deg;
        s_tilt_offset_deg = tilt_deg;
        ESP_LOGI(TAG, "NVS offsets saved: pan=%.2f° tilt=%.2f°", pan_deg, tilt_deg);
    }
    return err;
}

static esp_err_t nvs_erase_offsets(void)
{
    nvs_handle_t h;
    esp_err_t err = nvs_open(NVS_NAMESPACE, NVS_READWRITE, &h);
    if (err != ESP_OK) return err;
    nvs_erase_key(h, NVS_KEY_PAN_OFFSET);
    nvs_erase_key(h, NVS_KEY_TILT_OFFSET);
    err = nvs_commit(h);
    nvs_close(h);
    s_pan_offset_deg  = PAN_OFFSET_DEG;
    s_tilt_offset_deg = TILT_OFFSET_DEG;
    ESP_LOGI(TAG, "NVS offsets erased, defaults restored: pan=%.1f° tilt=%.1f°",
             s_pan_offset_deg, s_tilt_offset_deg);
    return err;
}

esp_err_t comm_nvs_init(void)
{
    esp_err_t err = nvs_flash_init();
    if (err == ESP_ERR_NVS_NO_FREE_PAGES || err == ESP_ERR_NVS_NEW_VERSION_FOUND) {
        nvs_flash_erase();
        err = nvs_flash_init();
    }
    if (err == ESP_OK) {
        nvs_load_offsets();
    }
    return err;
}

/* ── Shared command buffer (written by comm tasks, read by control) ──── */

static portMUX_TYPE s_cmd_lock = portMUX_INITIALIZER_UNLOCKED;
static tracking_command_t s_latest_cmd;
static volatile bool s_cmd_valid = false;

/* Per-channel freshness: prefer USB when both channels are active */
#define CHANNEL_USB  0
#define CHANNEL_WIFI 1
#define USB_PRIORITY_WINDOW_US  50000  /* 50ms — use USB if fresh within this window */

static volatile int64_t s_channel_last_us[2] = {0, 0};

static void publish_command(const tracking_command_t *cmd, int channel)
{
    int64_t now_us = esp_timer_get_time();

    portENTER_CRITICAL(&s_cmd_lock);
    /* Channel priority: reject WiFi packet if USB was received recently */
    if (channel == CHANNEL_WIFI &&
        (now_us - s_channel_last_us[CHANNEL_USB]) < USB_PRIORITY_WINDOW_US) {
        portEXIT_CRITICAL(&s_cmd_lock);
        return;  /* Stale WiFi — USB is active, discard */
    }
    s_latest_cmd = *cmd;
    s_cmd_valid = true;
    s_channel_last_us[channel] = now_us;
    portEXIT_CRITICAL(&s_cmd_lock);
}

bool comm_get_latest_command(tracking_command_t *out)
{
    portENTER_CRITICAL(&s_cmd_lock);
    bool valid = s_cmd_valid;
    if (valid) {
        *out = s_latest_cmd;
    }
    portEXIT_CRITICAL(&s_cmd_lock);
    return valid;
}

/* ── Packet parsing ──────────────────────────────────────────────────── */

static uint16_t crc16_ccitt(const uint8_t *data, size_t len)
{
    uint16_t crc = 0xFFFF;
    for (size_t i = 0; i < len; i++) {
        crc ^= (uint16_t)data[i] << 8;
        for (int j = 0; j < 8; j++) {
            if (crc & 0x8000)
                crc = (crc << 1) ^ 0x1021;
            else
                crc = (crc << 1);
        }
    }
    return crc;
}

bool packet_parse(const uint8_t *data, size_t len, tracking_packet_v2_t *out)
{
    if (len < PROTO_PACKET_SIZE) return false;
    if (data[0] != PROTO_HEADER_V2) return false;

    uint16_t crc = crc16_ccitt(&data[1], PROTO_PACKET_SIZE - 3);
    uint16_t pkt_crc;
    memcpy(&pkt_crc, &data[PROTO_PACKET_SIZE - 2], sizeof(uint16_t));
    if (crc != pkt_crc) return false;

    memcpy(out, data, PROTO_PACKET_SIZE);
    return true;
}

/* ── Calibration packet parsing (USB only) ───────────────────────────── */

static bool cal_packet_parse(const uint8_t *data, size_t len, cal_packet_t *out)
{
    if (len < CAL_PACKET_SIZE) return false;
    if (data[0] != CAL_HEADER) return false;

    /* CRC16 over bytes 1..7 (cmd, pan_cd, tilt_cd, reserved) */
    uint16_t crc = crc16_ccitt(&data[1], CAL_PACKET_SIZE - 3);
    uint16_t pkt_crc;
    memcpy(&pkt_crc, &data[CAL_PACKET_SIZE - 2], sizeof(uint16_t));
    if (crc != pkt_crc) return false;

    memcpy(out, data, CAL_PACKET_SIZE);
    return true;
}

static void cal_send_response(uint8_t cmd, int16_t pan_cd, int16_t tilt_cd)
{
    uint8_t pkt[CAL_PACKET_SIZE];
    pkt[0] = CAL_HEADER;
    pkt[1] = cmd;
    memcpy(&pkt[2], &pan_cd, 2);
    memcpy(&pkt[4], &tilt_cd, 2);
    uint16_t reserved = 0;
    memcpy(&pkt[6], &reserved, 2);
    uint16_t crc = crc16_ccitt(&pkt[1], CAL_PACKET_SIZE - 3);
    memcpy(&pkt[8], &crc, 2);
    usb_serial_jtag_write_bytes(pkt, CAL_PACKET_SIZE, pdMS_TO_TICKS(100));
}

static void handle_cal_packet(const cal_packet_t *cal)
{
    int16_t pan_cd  = (int16_t)(s_pan_offset_deg * 100.0f);
    int16_t tilt_cd = (int16_t)(s_tilt_offset_deg * 100.0f);

    switch (cal->command) {
    case CAL_CMD_SET_OFFSETS:
        nvs_save_offsets(cal->pan_cd / 100.0f, cal->tilt_cd / 100.0f);
        pan_cd  = cal->pan_cd;
        tilt_cd = cal->tilt_cd;
        break;
    case CAL_CMD_GET_OFFSETS:
        /* Just respond with current offsets */
        break;
    case CAL_CMD_RESET_DEFAULTS:
        nvs_erase_offsets();
        pan_cd  = (int16_t)(PAN_OFFSET_DEG * 100.0f);
        tilt_cd = (int16_t)(TILT_OFFSET_DEG * 100.0f);
        break;
    default:
        ESP_LOGW(TAG, "Unknown calibration command: 0x%02x", cal->command);
        return;
    }
    cal_send_response(cal->command, pan_cd, tilt_cd);
}

static inline float clamp_cd(int16_t centideg, int16_t lo, int16_t hi)
{
    if (centideg < lo) centideg = lo;
    if (centideg > hi) centideg = hi;
    return centideg / 100.0f;
}

void packet_to_command(const tracking_packet_v2_t *pkt, tracking_command_t *cmd)
{
    /* Clamp angles to +/-110 deg (extended MG996R range) and velocity */
    cmd->pan_deg      = clamp_cd(pkt->pan,      -11000, 11000);
    cmd->tilt_deg     = clamp_cd(pkt->tilt,     -11000, 11000);
    cmd->pan_vel_dps  = clamp_cd(pkt->pan_vel,  -30000, 30000);
    cmd->tilt_vel_dps = clamp_cd(pkt->tilt_vel, -30000, 30000);
    cmd->sequence     = pkt->sequence;
    cmd->state        = pkt->state;
    cmd->confidence   = pkt->confidence;
    cmd->quality      = pkt->quality;
    cmd->latency_ms   = pkt->latency;
    cmd->received_at_us = esp_timer_get_time();
}

/* ── USB CDC receive task ────────────────────────────────────────────── */

static void usb_rx_task(void *arg)
{
    uint8_t buf[128];
    uint8_t frame[PROTO_PACKET_SIZE];  /* max(PROTO_PACKET_SIZE, CAL_PACKET_SIZE) */
    int frame_pos = 0;
    uint8_t frame_header = 0;
    int frame_expected = 0;

    ESP_LOGI(TAG, "USB CDC RX task started");

    while (1) {
        int n = usb_serial_jtag_read_bytes(buf, sizeof(buf), pdMS_TO_TICKS(1));
        if (n <= 0) continue;

        for (int i = 0; i < n; i++) {
            /* Start of new frame: detect header type */
            if (frame_pos == 0) {
                if (buf[i] == PROTO_HEADER_V2) {
                    frame_header = PROTO_HEADER_V2;
                    frame_expected = PROTO_PACKET_SIZE;
                } else if (buf[i] == CAL_HEADER) {
                    frame_header = CAL_HEADER;
                    frame_expected = CAL_PACKET_SIZE;
                } else {
                    continue;  /* scan for header */
                }
            }
            frame[frame_pos++] = buf[i];

            /* >= (not ==): after a resync the kept bytes can already exceed the new
             * frame's expected length (e.g. a 0xCC cal header found early in a corrupted
             * 21-byte frame). With == the completion test never fired again and
             * frame[] was overrun on the stack. */
            if (frame_pos >= frame_expected) {
                if (frame_header == PROTO_HEADER_V2) {
                    tracking_packet_v2_t pkt;
                    if (packet_parse(frame, PROTO_PACKET_SIZE, &pkt)) {
                        tracking_command_t cmd;
                        packet_to_command(&pkt, &cmd);
                        publish_command(&cmd, CHANNEL_USB);
                        frame_pos = 0;
                    } else {
                        /* Checksum failed — resync */
                        int resync = -1;
                        for (int j = 1; j < frame_expected; j++) {
                            if (frame[j] == PROTO_HEADER_V2 || frame[j] == CAL_HEADER) {
                                resync = j;
                                break;
                            }
                        }
                        if (resync >= 0) {
                            int keep = frame_expected - resync;
                            memmove(frame, frame + resync, keep);
                            frame_pos = keep;
                            frame_header = frame[0];
                            frame_expected = (frame_header == CAL_HEADER) ? CAL_PACKET_SIZE : PROTO_PACKET_SIZE;
                        } else {
                            frame_pos = 0;
                        }
                    }
                } else if (frame_header == CAL_HEADER) {
                    cal_packet_t cal;
                    if (cal_packet_parse(frame, CAL_PACKET_SIZE, &cal)) {
                        handle_cal_packet(&cal);
                        frame_pos = 0;
                    } else {
                        frame_pos = 0;  /* Bad CRC on cal packet, discard */
                    }
                } else {
                    frame_pos = 0;
                }
            }
        }
    }
}

esp_err_t comm_usb_start(void)
{
    usb_serial_jtag_driver_config_t cfg = {
        .rx_buffer_size = TAFFF_USB_RX_BUFFER,
        .tx_buffer_size = TAFFF_USB_TX_BUFFER,
    };
    esp_err_t err = usb_serial_jtag_driver_install(&cfg);
    if (err != ESP_OK) {
        ESP_LOGE(TAG, "USB SJ driver install failed: %s", esp_err_to_name(err));
        return err;
    }

    BaseType_t ret = xTaskCreatePinnedToCore(
        usb_rx_task, "usb_rx", 4096, NULL, 5, NULL, 0 /* Core 0 */
    );
    if (ret != pdPASS) {
        ESP_LOGE(TAG, "Failed to create USB RX task");
        return ESP_FAIL;
    }
    return ESP_OK;
}

/* ── WiFi event handling ─────────────────────────────────────────────── */

static EventGroupHandle_t s_wifi_event_group;
#define WIFI_CONNECTED_BIT BIT0

/* Why the last join attempt failed (0 = none yet) and how many attempts have failed since the
 * station last had an IP. Written by the event handler, read by the UDP task's heartbeat. */
static volatile uint8_t  s_last_disc_reason;
static volatile uint32_t s_disc_count;

static void wifi_event_handler(void *arg, esp_event_base_t base,
                               int32_t event_id, void *event_data)
{
    if (base == WIFI_EVENT && event_id == WIFI_EVENT_STA_DISCONNECTED) {
        const wifi_event_sta_disconnected_t *dis = (const wifi_event_sta_disconnected_t *)event_data;
        const uint8_t reason = (dis != NULL) ? dis->reason : 0;
        const bool changed = (reason != s_last_disc_reason);
        s_last_disc_reason = reason;
        s_disc_count++;
        /* The hint is long: print it on the first failure, when the reason changes and every
         * 30th attempt, so the 1 Hz retry loop does not bury the log. */
        if (changed || (s_disc_count % 30u) == 1u) {
            ESP_LOGW(TAG, "WiFi disconnected: reason=%u (%s), attempt %lu. %s Retrying in 1s...",
                     (unsigned)reason, wifi_reason_name(reason), (unsigned long)s_disc_count,
                     wifi_reason_hint(reason));
        } else {
            ESP_LOGW(TAG, "WiFi disconnected: reason=%u (%s), attempt %lu. Retrying in 1s...",
                     (unsigned)reason, wifi_reason_name(reason), (unsigned long)s_disc_count);
        }
        xEventGroupClearBits(s_wifi_event_group, WIFI_CONNECTED_BIT);
        vTaskDelay(pdMS_TO_TICKS(1000));
        esp_wifi_connect();
    } else if (base == IP_EVENT && event_id == IP_EVENT_STA_GOT_IP) {
        ip_event_got_ip_t *event = (ip_event_got_ip_t *)event_data;
        char ip[16];
        snprintf(ip, sizeof ip, IPSTR, IP2STR(&event->ip_info.ip));
        ESP_LOGI(TAG, "Got IP: %s (after %lu failed attempts)", ip, (unsigned long)s_disc_count);
        if (WIFI_STATIC_IP[0] != '\0' && strcmp(ip, WIFI_STATIC_IP) != 0) {
            ESP_LOGW(TAG, "A static IP %s was configured but the station got %s: the tracker "
                          "(comms.udp_host) will not find this board until they match.",
                     WIFI_STATIC_IP, ip);
        }
        s_disc_count = 0;
        xEventGroupSetBits(s_wifi_event_group, WIFI_CONNECTED_BIT);
    }
}

static esp_err_t apply_static_wifi_ip(esp_netif_t *netif)
{
    if (WIFI_STATIC_IP[0] == '\0') {
        return ESP_OK;
    }

    if (WIFI_GATEWAY[0] == '\0' || WIFI_NETMASK[0] == '\0') {
        ESP_LOGW(TAG, "Static WiFi IP requested but gateway/netmask missing; using DHCP");
        return ESP_OK;
    }

    esp_netif_ip_info_t ip_info = {0};
    if (esp_netif_str_to_ip4(WIFI_STATIC_IP, &ip_info.ip) != ESP_OK ||
        esp_netif_str_to_ip4(WIFI_GATEWAY, &ip_info.gw) != ESP_OK ||
        esp_netif_str_to_ip4(WIFI_NETMASK, &ip_info.netmask) != ESP_OK) {
        ESP_LOGW(TAG, "Invalid static WiFi IP settings; using DHCP");
        return ESP_OK;
    }

    esp_err_t err = esp_netif_dhcpc_stop(netif);
    if (err != ESP_OK && err != ESP_ERR_ESP_NETIF_DHCP_ALREADY_STOPPED) {
        ESP_LOGW(TAG, "Failed to stop WiFi DHCP client: %s", esp_err_to_name(err));
        return err;
    }

    err = esp_netif_set_ip_info(netif, &ip_info);
    if (err != ESP_OK) {
        ESP_LOGW(TAG, "Failed to set static WiFi IP: %s", esp_err_to_name(err));
        return err;
    }

    if (WIFI_DNS[0] != '\0') {
        esp_netif_dns_info_t dns_info = {0};
        if (esp_netif_str_to_ip4(WIFI_DNS, &dns_info.ip.u_addr.ip4) == ESP_OK) {
            dns_info.ip.type = ESP_IPADDR_TYPE_V4;
            esp_netif_set_dns_info(netif, ESP_NETIF_DNS_MAIN, &dns_info);
        } else {
            ESP_LOGW(TAG, "Invalid static WiFi DNS '%s'; continuing without DNS override", WIFI_DNS);
        }
    }

    ESP_LOGI(TAG, "Using static WiFi IP %s gateway %s netmask %s",
             WIFI_STATIC_IP, WIFI_GATEWAY, WIFI_NETMASK);
    return ESP_OK;
}

/* ── WiFi UDP receive task ───────────────────────────────────────────── */

static void udp_rx_task(void *arg)
{
    /* Cold WiFi association can exceed 10s; keep UDP alive until STA has an IP. */
    while (1) {
        EventBits_t bits = xEventGroupWaitBits(s_wifi_event_group, WIFI_CONNECTED_BIT,
                            pdFALSE, pdTRUE, pdMS_TO_TICKS(10000));
        if (bits & WIFI_CONNECTED_BIT) {
            break;
        }
        ESP_LOGW(TAG, "WiFi not connected yet (last reason %u %s, %lu failed attempts); "
                      "UDP RX still waiting",
                 (unsigned)s_last_disc_reason, wifi_reason_name(s_last_disc_reason),
                 (unsigned long)s_disc_count);
    }

    int sock = socket(AF_INET, SOCK_DGRAM, IPPROTO_UDP);
    if (sock < 0) {
        ESP_LOGE(TAG, "Socket creation failed");
        vTaskDelete(NULL);
        return;
    }

    struct sockaddr_in addr = {
        .sin_family = AF_INET,
        .sin_port = htons(UDP_PORT),
        .sin_addr.s_addr = htonl(INADDR_ANY),
    };
    if (bind(sock, (struct sockaddr *)&addr, sizeof(addr)) < 0) {
        ESP_LOGE(TAG, "Socket bind failed");
        close(sock);
        vTaskDelete(NULL);
        return;
    }

    /* 1ms receive timeout — keep tight for low-latency control */
    struct timeval tv = { .tv_sec = 0, .tv_usec = 1000 };
    setsockopt(sock, SOL_SOCKET, SO_RCVTIMEO, &tv, sizeof(tv));

    uint8_t buf[64];
    ESP_LOGI(TAG, "UDP RX listening on port %d", UDP_PORT);

    while (1) {
        int n = recvfrom(sock, buf, sizeof(buf), 0, NULL, NULL);
        if (n == PROTO_PACKET_SIZE) {
            tracking_packet_v2_t pkt;
            if (packet_parse(buf, n, &pkt)) {
                tracking_command_t cmd;
                packet_to_command(&pkt, &cmd);
                publish_command(&cmd, CHANNEL_WIFI);
            }
        }
    }
}

/* Log what WiFi configuration this build carries. Lengths only: the SSID and password must never
 * reach the serial log, which ends up pasted into chats and bug reports. */
static void wifi_log_config(const char *ssid, const char *pass, const char *static_ip)
{
    if (ssid[0] == '\0') {
        ESP_LOGW(TAG, "No WiFi SSID was compiled in (firmware/esp32s3_gimbal/wifi_secrets.env "
                      "missing or empty when this build was made); WiFi cannot join.");
    }
    ESP_LOGI(TAG, "WiFi config: ssid_len=%u pass_len=%u ip=%s",
             (unsigned)strlen(ssid), (unsigned)strlen(pass),
             static_ip[0] != '\0' ? static_ip : "DHCP");
}

esp_err_t comm_wifi_start(void)
{
    s_wifi_event_group = xEventGroupCreate();

    /* NVS is already initialized by comm_nvs_init() in app_main */

    wifi_log_config(WIFI_SSID, WIFI_PASS, WIFI_STATIC_IP);

    esp_err_t err;

    err = esp_netif_init();
    if (err != ESP_OK) return err;
    err = esp_event_loop_create_default();
    if (err != ESP_OK) return err;
    esp_netif_t *sta_netif = esp_netif_create_default_wifi_sta();
    if (sta_netif == NULL) return ESP_FAIL;
    err = apply_static_wifi_ip(sta_netif);
    if (err != ESP_OK) return err;

    wifi_init_config_t wifi_cfg = WIFI_INIT_CONFIG_DEFAULT();
    err = esp_wifi_init(&wifi_cfg);
    if (err != ESP_OK) return err;

    err = esp_event_handler_register(WIFI_EVENT, ESP_EVENT_ANY_ID,
                                     &wifi_event_handler, NULL);
    if (err != ESP_OK) return err;
    err = esp_event_handler_register(IP_EVENT, IP_EVENT_STA_GOT_IP,
                                     &wifi_event_handler, NULL);
    if (err != ESP_OK) return err;

    wifi_config_t sta_cfg = {
        .sta = {
            .ssid = WIFI_SSID,
            .password = WIFI_PASS,
        },
    };
    err = esp_wifi_set_mode(WIFI_MODE_STA);
    if (err != ESP_OK) return err;
    err = esp_wifi_set_config(WIFI_IF_STA, &sta_cfg);
    if (err != ESP_OK) return err;
    err = esp_wifi_start();
    if (err != ESP_OK) return err;
    esp_wifi_set_ps(WIFI_PS_NONE);  /* Best-effort: disable power save */
    err = esp_wifi_connect();
    if (err != ESP_OK) {
        ESP_LOGW(TAG, "WiFi connect failed: %s — continuing USB-only", esp_err_to_name(err));
        return err;
    }

    ESP_LOGI(TAG, "WiFi STA connecting (PS=NONE)...");

    BaseType_t ret = xTaskCreatePinnedToCore(
        udp_rx_task, "udp_rx", 4096, NULL, 5, NULL, 0 /* Core 0 */
    );
    return (ret == pdPASS) ? ESP_OK : ESP_FAIL;
}
