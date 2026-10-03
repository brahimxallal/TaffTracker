/* Host-side harness: drives the REAL firmware usb_rx_task() (comm_task.c, included below)
 * with synthetic byte streams, under AddressSanitizer/UBSan.
 *   harness crafted         -> one hand-built stream that exposes the resync bug
 *   harness valid           -> functional checks (valid frames, noise, recovery)
 *   harness fuzz <n> <seed> -> n random streams biased toward header bytes
 */
#include <setjmp.h>
#include <stdio.h>
#include <stdlib.h>
#include <string.h>

#include "driver/usb_serial_jtag.h"
#include "esp_timer.h"

int64_t esp_timer_get_time(void) { static int64_t t; return t += 1000; }

static jmp_buf g_done;
static const uint8_t *g_in;
static size_t g_in_len, g_in_pos, g_chunk = 7;

int usb_serial_jtag_read_bytes(void *buf, uint32_t length, TickType_t ticks)
{
    (void)ticks;
    if (g_in_pos >= g_in_len) longjmp(g_done, 1);
    size_t n = g_in_len - g_in_pos;
    if (n > g_chunk) n = g_chunk;
    if (n > length) n = length;
    memcpy(buf, g_in + g_in_pos, n);
    g_in_pos += n;
    return (int)n;
}

#include COMM_TASK_C /* real code under test */

static void run_stream(const uint8_t *data, size_t len, size_t chunk)
{
    s_cmd_valid = false;
    memset(&s_latest_cmd, 0, sizeof s_latest_cmd);
    g_in = data; g_in_len = len; g_in_pos = 0; g_chunk = chunk ? chunk : 1;
    if (setjmp(g_done) == 0) usb_rx_task(NULL);
}

static size_t put_v2(uint8_t *out, uint16_t seq, int16_t pan_cd, uint8_t state, bool good_crc)
{
    tracking_packet_v2_t p; memset(&p, 0, sizeof p);
    p.header = PROTO_HEADER_V2; p.sequence = seq; p.pan = pan_cd; p.state = state;
    uint8_t raw[PROTO_PACKET_SIZE]; memcpy(raw, &p, sizeof raw);
    uint16_t crc = crc16_ccitt(&raw[1], PROTO_PACKET_SIZE - 3);
    if (!good_crc) crc ^= 0x5A5A;
    memcpy(&raw[PROTO_PACKET_SIZE - 2], &crc, 2);
    memcpy(out, raw, sizeof raw);
    return sizeof raw;
}

static int check_last_seq(const char *name, int expect)
{
    tracking_command_t c;
    bool ok = comm_get_latest_command(&c);
    int got = ok ? c.sequence : -1;
    printf("  %-44s last_seq=%d expect=%d  %s\n", name, got, expect, got == expect ? "PASS" : "FAIL");
    return got == expect ? 0 : 1;
}

static int test_crafted(void)
{
    /* v2 header, then 0xCC at index 1 (a cal header), 19 zero bytes -> CRC fails ->
     * resync keeps 20 bytes and switches to the 10-byte cal frame. Then 40 more bytes. */
    uint8_t s[21 + 40 + 21]; size_t n = 0;
    s[n++] = PROTO_HEADER_V2; s[n++] = CAL_HEADER; memset(&s[n], 0, 19); n += 19;
    memset(&s[n], 0x00, 40); n += 40;
    n += put_v2(&s[n], 44, 1000, 0, true);
    run_stream(s, n, 5);
    return check_last_seq("crafted resync-into-cal stream (+valid tail)", 44);
}

static int test_valid(void)
{
    int fails = 0; uint8_t s[2048]; size_t n;
    n = 0; for (int i = 1; i <= 10; i++) n += put_v2(&s[n], (uint16_t)i, (int16_t)(i * 100), 0, true);
    run_stream(s, n, 7); fails += check_last_seq("10 valid frames", 10);

    n = 0; memset(s, 0x11, 37); n = 37; n += put_v2(&s[n], 42, 0, 0, true);
    run_stream(s, n, 3); fails += check_last_seq("noise prefix then valid frame", 42);

    n = 0; n += put_v2(&s[n], 7, 0, 0, false);       /* bad-CRC frame, no embedded headers except [0] */
    n += put_v2(&s[n], 43, 0, 0, true);
    run_stream(s, n, 21); fails += check_last_seq("bad-CRC frame then valid frame", 43);

    n = 0; s[n++] = CAL_HEADER; memset(&s[n], 0x22, 9); n += 9;   /* bad cal frame then valid v2 */
    n += put_v2(&s[n], 45, 0, 0, true);
    run_stream(s, n, 4); fails += check_last_seq("bad cal frame then valid frame", 45);
    return fails;
}

static uint32_t rng;
static uint32_t rnd(void) { rng ^= rng << 13; rng ^= rng >> 17; rng ^= rng << 5; return rng; }

static int test_fuzz(long iters, uint32_t seed)
{
    rng = seed ? seed : 1; uint8_t s[1500];
    for (long it = 0; it < iters; it++) {
        size_t n = 1 + rnd() % sizeof s;
        for (size_t i = 0; i < n; i++) {
            uint32_t r = rnd() % 8;
            s[i] = r == 0 ? PROTO_HEADER_V2 : r == 1 ? CAL_HEADER : (uint8_t)rnd();
        }
        run_stream(s, n, 1 + rnd() % 40);
    }
    printf("  fuzz: %ld random streams, seed=%u -> no memory error\n", iters, seed);
    return 0;
}

int main(int argc, char **argv)
{
    const char *mode = argc > 1 ? argv[1] : "valid";
    if (!strcmp(mode, "crafted")) return test_crafted();
    if (!strcmp(mode, "valid")) return test_valid();
    if (!strcmp(mode, "fuzz")) return test_fuzz(argc > 2 ? atol(argv[2]) : 20000, argc > 3 ? (uint32_t)atol(argv[3]) : 1);
    fprintf(stderr, "unknown mode\n"); return 2;
}
