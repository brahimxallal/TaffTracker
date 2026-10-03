/* Host-side harness for the REAL control_task.c: checks relay/laser pin behaviour
 * of control_loop_cb() as a function of command freshness. */
#include <stdbool.h>
#include <stdio.h>
#include <string.h>

#include "comm_task.h"
#include "driver/gpio.h"
#include "esp_timer.h"

static int g_level[64];
static int64_t g_now_us;
static tracking_command_t g_cmd;

esp_err_t gpio_set_level(int pin, uint32_t level) { g_level[pin] = (int)level; return ESP_OK; }
int64_t esp_timer_get_time(void) { return g_now_us; }
bool comm_get_latest_command(tracking_command_t *out) { *out = g_cmd; return true; }
void comm_get_offsets(float *p, float *t) { *p = 0.0f; *t = 0.0f; }
void comm_get_control_params(float *a, float *b, float *c, float *d) { *a = 0.15f; *b = 0.30f; *c = 300.0f; *d = 500.0f; }
void servo_set_angle(int ch, float deg) { (void)ch; (void)deg; }

#include CONTROL_TASK_C /* real code under test */

static int run(const char *name, uint8_t state, double age_s, int expect_relay_level)
{
    memset(&g_cmd, 0, sizeof g_cmd);
    g_cmd.state = state;
    g_now_us = 100LL * 1000000LL;                              /* arbitrary "now" = 100 s */
    g_cmd.received_at_us = g_now_us - (int64_t)(age_s * 1e6);  /* packet received age_s ago */
    g_level[PIN_RELAY] = -1; g_level[PIN_LASER] = -1;
    control_loop_cb(NULL);
    int laser_expect = (state & FLAG_LASER_ON) ? 1 : 0;
    bool ok = g_level[PIN_RELAY] == expect_relay_level && g_level[PIN_LASER] == laser_expect;
    printf("  %-52s relay_level=%d (want %d) laser=%d (want %d)  %s\n", name,
           g_level[PIN_RELAY], expect_relay_level, g_level[PIN_LASER], laser_expect, ok ? "PASS" : "FAIL");
    return ok ? 0 : 1;
}

int main(void)
{
    int fails = 0;
    /* relay is active-low: level 0 = energised (ON), 1 = released (OFF) */
    fails += run("fresh packet, relay bit set        (age 10 ms)", FLAG_RELAY_ON | FLAG_LASER_ON, 0.010, 0);
    fails += run("no relay bit                       (age 10 ms)", FLAG_LASER_ON, 0.010, 1);
    fails += run("relay bit, jittery link            (age 0.40 s)", FLAG_RELAY_ON | FLAG_LASER_ON, 0.400, 0);
    fails += run("relay bit, host silent just over   (age 0.60 s)", FLAG_RELAY_ON | FLAG_LASER_ON, 0.600, 1);
    fails += run("relay bit, host silent coast stage (age 1.50 s)", FLAG_RELAY_ON, 1.500, 1);
    fails += run("relay bit, host long gone          (age 30 s)", FLAG_RELAY_ON | FLAG_LASER_ON, 30.0, 1);
    printf("%s (%d failing)\n", fails ? "RESULT: FAIL" : "RESULT: PASS", fails);
    return fails ? 1 : 0;
}
