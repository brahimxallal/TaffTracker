#pragma once
/* Names and hints for WiFi station disconnect reasons, so the serial log says WHY a join fails.
 *
 * The numbers are ESP-IDF's wifi_err_reason_t: IEEE 802.11 reason codes (1-66) plus Espressif's
 * 200+ range. They are written as numbers on purpose: this header must compile on every ESP-IDF
 * version and in the host test harness, which has no real esp_wifi headers. The numeric code is
 * always logged next to the name, so a wrong or missing name never hides the real code.
 */
#include <stdint.h>

static inline const char *wifi_reason_name(uint8_t reason)
{
    switch (reason) {
    case 2:   return "AUTH_EXPIRE";
    case 3:   return "AUTH_LEAVE";
    case 4:   return "ASSOC_EXPIRE";
    case 5:   return "ASSOC_TOOMANY";
    case 8:   return "ASSOC_LEAVE";
    case 14:  return "MIC_FAILURE";
    case 15:  return "4WAY_HANDSHAKE_TIMEOUT";
    case 200: return "BEACON_TIMEOUT";
    case 201: return "NO_AP_FOUND";
    case 202: return "AUTH_FAIL";
    case 203: return "ASSOC_FAIL";
    case 204: return "HANDSHAKE_TIMEOUT";
    case 205: return "CONNECTION_FAIL";
    case 206: return "AP_TSF_RESET";
    case 207: return "ROAMING";
    case 210: return "NO_AP_FOUND_W_COMPATIBLE_SECURITY";
    case 211: return "NO_AP_FOUND_IN_AUTHMODE_THRESHOLD";
    case 212: return "NO_AP_FOUND_IN_RSSI_THRESHOLD";
    default:  return "UNKNOWN";
    }
}

/* One line on what usually causes this reason. Never NULL, never empty. */
static inline const char *wifi_reason_hint(uint8_t reason)
{
    switch (reason) {
    case 201:
    case 212:
        return "SSID not seen: check its spelling and case, that the router's 2.4 GHz WiFi is on "
               "(the ESP32-S3 has no 5 GHz), that its channel is 1-11, and the range.";
    case 210:
    case 211:
        return "Network seen but its security mode is not accepted: use WPA2 or WPA2/WPA3 mixed "
               "mode on the router.";
    case 202:
    case 15:
    case 204:
    case 14:
        return "Most likely a wrong password: fix wifi_secrets.env and rebuild. A WPA3-only router "
               "can also cause this.";
    case 203:
    case 5:
        return "The router refused the join: client limit, MAC filter or an incompatible 802.11 "
               "mode.";
    case 200:
    case 205:
    case 206:
    case 2:
    case 4:
        return "Link lost or unstable: check the signal strength and the board's power supply.";
    case 3:
    case 8:
    case 207:
        return "The router dropped the station; it retries by itself.";
    default:
        return "See ESP-IDF wifi_err_reason_t for this code.";
    }
}
