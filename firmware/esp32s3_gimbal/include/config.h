#pragma once

#include <stdint.h>

/* ── GPIO ─────────────────────────────────────────────────────────────── */
#define PIN_SERVO_PAN   12
#define PIN_SERVO_TILT  11
#define PIN_LASER       14
#define PIN_LAMP         6          /* auxiliary lamp, HIGH when target acquired */
#define LAMP_OFF_DELAY_US  3000000   /* 3s before lamp turns off (relay protection) */
#define PIN_RELAY        4

/* ── LEDC PWM (14-bit, 50 Hz) ────────────────────────────────────────── */
#define LEDC_RESOLUTION     14              /* bits                        */
#define LEDC_MAX_DUTY       ((1 << LEDC_RESOLUTION) - 1)  /* 16383        */
#define LEDC_FREQ_HZ        50
#define SERVO_PERIOD_US      20000
/* Servo pulse & range — override via platformio.ini build_flags.
   MG996R standard: 500–2500 µs = 180° travel. */
#ifndef TAFFF_SERVO_MIN_US
#define TAFFF_SERVO_MIN_US   500
#endif
#ifndef TAFFF_SERVO_MAX_US
#define TAFFF_SERVO_MAX_US   2500
#endif
#ifndef TAFFF_SERVO_RANGE_DEG
#define TAFFF_SERVO_RANGE_DEG 180.0f
#endif
#define SERVO_MIN_US         TAFFF_SERVO_MIN_US
#define SERVO_MAX_US         TAFFF_SERVO_MAX_US
#define SERVO_RANGE_DEG      TAFFF_SERVO_RANGE_DEG
#define SERVO_CENTER_DEG     (SERVO_RANGE_DEG / 2.0f)

/* ── Servo calibration offsets (degrees, set during first-time alignment) */
#ifndef TAFFF_PAN_OFFSET_DEG
#define TAFFF_PAN_OFFSET_DEG  -65.0f
#endif
#ifndef TAFFF_TILT_OFFSET_DEG
#define TAFFF_TILT_OFFSET_DEG 7.0f
#endif
#define PAN_OFFSET_DEG       TAFFF_PAN_OFFSET_DEG
#define TILT_OFFSET_DEG      TAFFF_TILT_OFFSET_DEG

/* ── Control loop ─────────────────────────────────────────────────────── */
#ifndef TAFFF_CONTROL_HZ
#define TAFFF_CONTROL_HZ     200
#endif
#define CONTROL_PERIOD_US    (1000000 / TAFFF_CONTROL_HZ)  /* 5000 us     */
#define DT_CTRL              (1.0f / TAFFF_CONTROL_HZ)     /* 0.005 s     */

/* ── Layer 3 filter defaults ──────────────────────────────────────────── */
/* PC-side Kalman+1Euro already provides smooth commands; firmware EMA    */
/* should be fast enough to follow without adding significant phase lag.  */
/* Vel/accel limits are safety nets, not primary smoothing.               */
#define DEAD_ZONE_DEG        0.15f          /* narrow: PC handles deadband (0.5°) */
#define DEAD_ZONE_PRED_DEG   0.30f          /* wider during prediction      */
#define MAX_VEL_DPS          300.0f         /* MG996R sustained speed       */
#define MAX_ACCEL_DPSS       500.0f         /* moderate jerk limit          */

/* ── Lead compensation ────────────────────────────────────────────────── */
/* DISABLED for ego-motion mount: velocity feedforward amplifies pixel    */
/* noise into servo jitter when camera moves with gimbal. Re-enable only  */
/* after implementing pixel-velocity smoothing + ego-motion subtraction.  */
#define LEAD_COMP_ENABLED    0
#define LEAD_COMP_ALPHA      0.3f           /* low-pass on acceleration     */
#define LEAD_COMP_GAIN       0.005f         /* feedforward gain (seconds)   */

/* ── Adaptive EMA speed thresholds (deg/s) ──────────────────── */
#define ALPHA_THRESH_STAT    5.0f
#define ALPHA_THRESH_SLOW    30.0f
#define ALPHA_THRESH_MOD     90.0f

/* EMA alphas: faster than before — PC sends pre-smoothed commands via   */
/* Kalman + 1Euro, so firmware EMA needs less filtering. Lower values     */
/* added ~60ms of lag at rest, creating oscillation in the feedback loop. */
#define ALPHA_VAL_STAT       0.20f          /* was 0.08: τ=25ms (from 62ms) */
#define ALPHA_VAL_SLOW       0.45f          /* was 0.30 */
#define ALPHA_VAL_MOD        0.70f          /* was 0.55 */
#define ALPHA_VAL_FAST       0.90f          /* was 0.75: near pass-through */

/* ── Manual mode flag (quality byte MSB from PC) ──────────────────────── */
#define QUALITY_FLAG_MANUAL  0x80           /* bypass EMA + dead-zone for manual commands */

/* ── Watchdog timeouts (seconds) ──────────────────────────────────────── */
#define WDT_COAST_S          0.100f
#define WDT_HOLD_S           0.500f
#define WDT_HOME_S           2.000f
#define HOME_SPEED_DPS       10.0f

/* ── Communication ────────────────────────────────────────────────────── */
#ifndef TAFFF_USB_RX_BUFFER
#define TAFFF_USB_RX_BUFFER  2048
#endif
#ifndef TAFFF_USB_TX_BUFFER
#define TAFFF_USB_TX_BUFFER  512
#endif

/* WiFi credentials: set via sdkconfig menuconfig or build flags, NOT here. */
#ifndef WIFI_SSID
#define WIFI_SSID            CONFIG_TAFFF_WIFI_SSID
#endif
#ifndef WIFI_PASS
#define WIFI_PASS            CONFIG_TAFFF_WIFI_PASS
#endif
#define UDP_PORT             6000

/* ── Protocol v2 (21 bytes, header 0xBB) ──────────────────────────────── */
#define PROTO_HEADER_V2      0xBB
#define PROTO_PACKET_SIZE    21

/* ── Calibration protocol (10 bytes, header 0xCC, USB only) ──────────── */
#define CAL_HEADER           0xCC
#define CAL_PACKET_SIZE      10
#define CAL_CMD_SET_OFFSETS  0x01
#define CAL_CMD_GET_OFFSETS  0x02
#define CAL_CMD_RESET_DEFAULTS 0x03

/* NVS namespace and keys for calibration data */
#define NVS_NAMESPACE        "tafff_cal"
#define NVS_KEY_PAN_OFFSET   "pan_off_cd"
#define NVS_KEY_TILT_OFFSET  "tilt_off_cd"
#define NVS_KEY_DEAD_ZONE    "dz_cd"
#define NVS_KEY_DEAD_ZONE_P  "dzp_cd"
#define NVS_KEY_MAX_VEL      "maxvel_cd"
#define NVS_KEY_MAX_ACCEL    "maxacc_cd"

/* State flag bits (byte 16) */
#define STATE_MASK           0x03
#define STATE_LOST           0x00
#define STATE_PREDICTION     0x01
#define STATE_MEASUREMENT    0x02
#define STATE_CENTER         0x03
#define FLAG_TARGET_ACQUIRED     (1 << 2)
#define FLAG_HIGH_CONFIDENCE     (1 << 3)
#define FLAG_FAST_MOTION         (1 << 4)
#define FLAG_OCCLUSION_RECOVERY  (1 << 5)
#define FLAG_LASER_ON            (1 << 6)
#define FLAG_RELAY_ON            (1 << 7)

/* ── Packed protocol v2 structure ─────────────────────────────────────── */
typedef struct __attribute__((packed)) {
    uint8_t  header;        /* 0xBB                       */
    uint16_t sequence;
    uint32_t timestamp_ms;
    int16_t  pan;           /* centidegrees               */
    int16_t  tilt;          /* centidegrees               */
    int16_t  pan_vel;       /* centidegrees / sec          */
    int16_t  tilt_vel;      /* centidegrees / sec          */
    uint8_t  confidence;    /* 0-255                      */
    uint8_t  state;         /* bit flags                  */
    uint8_t  quality;       /* 0-255                      */
    uint8_t  latency;       /* PC pipeline ms, capped 255 */
    uint16_t checksum;      /* CRC16 of bytes 1-18        */
    } __attribute__((packed)) tracking_packet_v2_t;

    _Static_assert(sizeof(tracking_packet_v2_t) == PROTO_PACKET_SIZE,
               "tracking_packet_v2_t must be exactly 21 bytes");

/* ── Packed calibration packet (10 bytes, USB only) ───────────────────── */
typedef struct __attribute__((packed)) {
    uint8_t  header;        /* 0xCC                       */
    uint8_t  command;       /* CAL_CMD_*                  */
    int16_t  pan_cd;        /* pan offset centidegrees    */
    int16_t  tilt_cd;       /* tilt offset centidegrees   */
    uint16_t reserved;
    uint16_t checksum;      /* CRC16 of bytes 1..7        */
} __attribute__((packed)) cal_packet_t;

_Static_assert(sizeof(cal_packet_t) == CAL_PACKET_SIZE,
               "cal_packet_t must be exactly 10 bytes");

/* ── Runtime command shared between comm and control tasks ────────────── */
typedef struct {
    float    pan_deg;       /* target pan in degrees       */
    float    tilt_deg;      /* target tilt in degrees      */
    float    pan_vel_dps;   /* pan velocity deg/s          */
    float    tilt_vel_dps;  /* tilt velocity deg/s         */
    uint16_t sequence;      /* packet sequence for new-cmd detection */
    uint8_t  state;         /* state flags byte            */
    uint8_t  confidence;    /* 0-255                      */
    uint8_t  quality;       /* quality byte (MSB = QUALITY_FLAG_MANUAL) */
    uint8_t  latency_ms;    /* PC pipeline latency ms      */
    int64_t  received_at_us;/* esp_timer_get_time() stamp  */
} tracking_command_t;