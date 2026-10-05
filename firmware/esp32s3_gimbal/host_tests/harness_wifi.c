/* Host-side harness for the WiFi event handling in the REAL comm_task.c (included below).
 * When a join fails the board must say why (reason code, name, hint), keep retrying, and reset
 * its failure count once it has an IP; a static IP that DHCP overrode must be called out.
 * ESP-IDF calls are replaced by tiny recorders; ESP_LOGW/ESP_LOGI output is captured so the
 * tests can assert on what a person reading the serial monitor would actually see. */
#include <stdarg.h>
#include <stdbool.h>
#include <stdio.h>
#include <string.h>

#include "driver/usb_serial_jtag.h"
#include "esp_event.h"
#include "esp_log.h"
#include "esp_timer.h"
#include "esp_wifi.h"
#include "freertos/FreeRTOS.h"
#include "freertos/event_groups.h"

/* Build-time WiFi settings, defined before config.h (included by comm_task.c) so they win over
 * its empty defaults. The static IP exercises the "DHCP overrode it" warning; the SSID and
 * password are canaries that must never show up in any log line. */
#define CONFIG_TAFFF_WIFI_STATIC_IP "192.168.1.12"
#define CONFIG_TAFFF_WIFI_SSID "HomeNet-canary"
#define CONFIG_TAFFF_WIFI_PASS "hunter2-canary!"

int64_t esp_timer_get_time(void) { return 0; }
int usb_serial_jtag_read_bytes(void *buf, uint32_t length, TickType_t ticks)
{
    (void)buf; (void)length; (void)ticks;
    return 0;
}

/* Recorders: defined after the stub headers and before the code under test. */
static int g_connect_calls;
static EventBits_t g_bits;
static esp_err_t fake_wifi_connect(void)
{
    g_connect_calls++;
    return ESP_OK;
}
#define esp_wifi_connect() fake_wifi_connect()
#define xEventGroupSetBits(g, b) (g_bits |= (b))
#define xEventGroupClearBits(g, b) (g_bits &= ~(EventBits_t)(b))

static char g_log[8192];
static void log_clear(void) { g_log[0] = '\0'; }
static void log_capture(const char *fmt, ...) __attribute__((format(printf, 1, 2)));
static void log_capture(const char *fmt, ...)
{
    size_t used = strlen(g_log);
    va_list ap;
    va_start(ap, fmt);
    vsnprintf(g_log + used, sizeof g_log - used, fmt, ap);
    va_end(ap);
}
#undef ESP_LOGE
#undef ESP_LOGW
#undef ESP_LOGI
#define ESP_LOGE(tag, fmt, ...) ((void)(tag), log_capture(fmt "\n", ##__VA_ARGS__))
#define ESP_LOGW(tag, fmt, ...) ((void)(tag), log_capture(fmt "\n", ##__VA_ARGS__))
#define ESP_LOGI(tag, fmt, ...) ((void)(tag), log_capture(fmt "\n", ##__VA_ARGS__))

#include COMM_TASK_C /* real code under test */

static int g_fail;

static void check(const char *name, bool ok)
{
    printf("  %-66s %s\n", name, ok ? "PASS" : "FAIL");
    if (!ok) {
        g_fail++;
        printf("    log was:\n%s\n", g_log);
    }
}

static bool logged(const char *needle) { return strstr(g_log, needle) != NULL; }

static void reset_state(void)
{
    s_last_disc_reason = 0;
    s_disc_count = 0;
    g_connect_calls = 0;
    g_bits = WIFI_CONNECTED_BIT;
    log_clear();
}

static void emit_disconnect(uint8_t reason)
{
    wifi_event_sta_disconnected_t ev;
    memset(&ev, 0, sizeof ev);
    ev.reason = reason;
    wifi_event_handler(NULL, WIFI_EVENT, WIFI_EVENT_STA_DISCONNECTED, &ev);
}

/* addr is in network byte order read as a little-endian word: 192.168.1.12 -> 0x0C01A8C0. */
static void emit_got_ip(uint32_t addr)
{
    ip_event_got_ip_t ev;
    memset(&ev, 0, sizeof ev);
    ev.ip_info.ip.addr = addr;
    wifi_event_handler(NULL, IP_EVENT, IP_EVENT_STA_GOT_IP, &ev);
}

static void test_reason_names_and_hints(void)
{
    check("201 is NO_AP_FOUND", !strcmp(wifi_reason_name(201), "NO_AP_FOUND"));
    check("202 is AUTH_FAIL", !strcmp(wifi_reason_name(202), "AUTH_FAIL"));
    check("15 is 4WAY_HANDSHAKE_TIMEOUT", !strcmp(wifi_reason_name(15), "4WAY_HANDSHAKE_TIMEOUT"));
    check("205 is CONNECTION_FAIL", !strcmp(wifi_reason_name(205), "CONNECTION_FAIL"));
    check("200 is BEACON_TIMEOUT", !strcmp(wifi_reason_name(200), "BEACON_TIMEOUT"));
    check("211 is NO_AP_FOUND_IN_AUTHMODE_THRESHOLD",
          !strcmp(wifi_reason_name(211), "NO_AP_FOUND_IN_AUTHMODE_THRESHOLD"));
    check("an unlisted code is UNKNOWN", !strcmp(wifi_reason_name(99), "UNKNOWN"));

    bool all_non_empty = true;
    for (int r = 0; r < 256; r++) {
        const char *h = wifi_reason_hint((uint8_t)r);
        if (h == NULL || h[0] == '\0') all_non_empty = false;
    }
    check("every reason code (0-255) has a non-empty hint", all_non_empty);
    check("NO_AP_FOUND hint points at the 2.4 GHz band", strstr(wifi_reason_hint(201), "2.4 GHz"));
    check("AUTH_FAIL hint points at the password", strstr(wifi_reason_hint(202), "password"));
    check("handshake timeout hint points at the password", strstr(wifi_reason_hint(15), "password"));
}

static void test_disconnects_are_tracked_and_retried(void)
{
    reset_state();
    emit_disconnect(201);
    check("first failure: reason recorded", s_last_disc_reason == 201);
    check("first failure: counted", s_disc_count == 1);
    check("first failure: connected bit cleared", (g_bits & WIFI_CONNECTED_BIT) == 0);
    check("first failure: reconnect requested", g_connect_calls == 1);
    check("first failure: log names the reason", logged("reason=201 (NO_AP_FOUND)"));

    emit_disconnect(201);
    emit_disconnect(201);
    check("repeated failures are counted and retried", s_disc_count == 3 && g_connect_calls == 3);

    emit_disconnect(202);
    check("a changed reason is recorded", s_last_disc_reason == 202 && s_disc_count == 4);

    log_clear();
    emit_got_ip(0x0C01A8C0u); /* 192.168.1.12 */
    check("got IP: connected bit set", (g_bits & WIFI_CONNECTED_BIT) != 0);
    check("got IP: failure count reset", s_disc_count == 0);
    check("got IP: log reports the address and the failed attempts",
          logged("Got IP: 192.168.1.12 (after 4 failed attempts)"));

    reset_state();
    wifi_event_handler(NULL, WIFI_EVENT, WIFI_EVENT_STA_DISCONNECTED, NULL);
    check("a disconnect event without data is survived and retried",
          s_disc_count == 1 && s_last_disc_reason == 0 && g_connect_calls == 1);
}

static void test_hint_is_not_repeated_every_second(void)
{
    reset_state();
    emit_disconnect(201);
    check("first failure prints the hint", logged("2.4 GHz"));

    log_clear();
    emit_disconnect(201);
    check("same reason again: short line, no hint",
          logged("reason=201 (NO_AP_FOUND), attempt 2.") && !logged("2.4 GHz"));

    for (int i = 0; i < 28; i++) emit_disconnect(201); /* attempts 3..30 */
    log_clear();
    emit_disconnect(201); /* attempt 31 */
    check("every 30th attempt prints the hint again", logged("attempt 31.") && logged("2.4 GHz"));

    log_clear();
    emit_disconnect(202);
    check("a new reason prints its own hint at once", logged("password"));
}

static void test_static_ip_mismatch_is_called_out(void)
{
    reset_state();
    emit_got_ip(0x0C01A8C0u); /* 192.168.1.12: what was configured */
    check("matching address: no warning", !logged("static IP"));

    reset_state();
    emit_got_ip(0x1501A8C0u); /* 192.168.1.21: DHCP answered instead */
    check("different address: warns with both addresses",
          logged("static IP 192.168.1.12") && logged("got 192.168.1.21"));
    check("different address: the link is still treated as up", (g_bits & WIFI_CONNECTED_BIT) != 0);
}

static void test_config_log_has_lengths_only(void)
{
    reset_state();
    wifi_log_config("HomeNet-canary", "hunter2-canary!", "192.168.1.12");
    check("config log shows the lengths and the static IP",
          logged("ssid_len=14 pass_len=15 ip=192.168.1.12"));
    check("config log never shows the SSID", !logged("HomeNet") && !logged("canary"));
    check("config log never shows the password", !logged("hunter2"));
    check("no 'missing SSID' warning when an SSID is present", !logged("No WiFi SSID"));

    reset_state();
    wifi_log_config("", "", "");
    check("an empty SSID is called out", logged("No WiFi SSID was compiled in"));
    check("no static IP is reported as DHCP", logged("ssid_len=0 pass_len=0 ip=DHCP"));
}

static void test_start_never_logs_the_credentials(void)
{
    reset_state();
    esp_err_t err = comm_wifi_start();
    check("comm_wifi_start runs to the end against the stubs", err == ESP_OK);
    check("start logs the config lengths", logged("ssid_len=14 pass_len=15 ip=192.168.1.12"));
    check("start never logs the SSID", !logged("HomeNet") && !logged("canary"));
    check("start never logs the password", !logged("hunter2"));
}

int main(void)
{
    s_wifi_event_group = (EventGroupHandle_t)1;
    test_reason_names_and_hints();
    test_disconnects_are_tracked_and_retried();
    test_hint_is_not_repeated_every_second();
    test_static_ip_mismatch_is_called_out();
    test_config_log_has_lengths_only();
    test_start_never_logs_the_credentials();
    printf("%s (%d failing)\n", g_fail ? "RESULT: FAIL" : "RESULT: PASS", g_fail);
    return g_fail ? 1 : 0;
}
