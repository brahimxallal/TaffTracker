#include "freertos/FreeRTOS.h"
#include "freertos/task.h"
#include "esp_log.h"

#include "driver/gpio.h"

#include "config.h"
#include "servo_driver.h"
#include "comm_task.h"
#include "control_task.h"

static const char *TAG = "tafff";

void app_main(void)
{
    ESP_LOGI(TAG, "╔══════════════════════════════════════╗");
    ESP_LOGI(TAG, "║  TAFFF Gimbal Tracker — ESP32-S3     ║");
    ESP_LOGI(TAG, "║  Pan=GPIO%d  Tilt=GPIO%d  Ctrl=%dHz  ║",
             PIN_SERVO_PAN, PIN_SERVO_TILT, TAFFF_CONTROL_HZ);
    ESP_LOGI(TAG, "╚══════════════════════════════════════╝");

    /* 0a. Laser GPIO init (default ON, protocol bit 6 controls at runtime) */
    gpio_config_t laser_cfg = {
        .pin_bit_mask = (1ULL << PIN_LASER) | (1ULL << PIN_LAMP),
        .mode         = GPIO_MODE_OUTPUT,
        .pull_up_en   = GPIO_PULLUP_DISABLE,
        .pull_down_en = GPIO_PULLDOWN_DISABLE,
        .intr_type    = GPIO_INTR_DISABLE,
    };
    ESP_ERROR_CHECK(gpio_config(&laser_cfg));
    gpio_set_level(PIN_LASER, 1);
    gpio_set_level(PIN_LAMP, 1);  /* lamp OFF (active-low relay) */
    ESP_LOGI(TAG, "Laser ON (GPIO%d), Lamp OFF (GPIO%d)", PIN_LASER, PIN_LAMP);

    /* 0b. Relay GPIO init (default OFF, protocol bit 7 controls at runtime) */
    gpio_config_t relay_cfg = {
        .pin_bit_mask = 1ULL << PIN_RELAY,
        .mode         = GPIO_MODE_OUTPUT,
        .pull_up_en   = GPIO_PULLUP_DISABLE,
        .pull_down_en = GPIO_PULLDOWN_DISABLE,
        .intr_type    = GPIO_INTR_DISABLE,
    };
    ESP_ERROR_CHECK(gpio_config(&relay_cfg));
    gpio_set_level(PIN_RELAY, 1);  /* active-low: 1 = OFF */
    ESP_LOGI(TAG, "Relay OFF (GPIO%d)", PIN_RELAY);

    /* 1. Initialise servos (drives to center on start) */
    ESP_ERROR_CHECK(servo_init());

    /* 2. Init NVS and load saved calibration offsets */
    ESP_ERROR_CHECK(comm_nvs_init());

    /* 3. Start USB CDC communication (always active) */
    ESP_ERROR_CHECK(comm_usb_start());

#if defined(CONFIG_ESP_WIFI_ENABLED) && (!defined(TAFFF_UART_INGEST) || TAFFF_UART_INGEST == 0)
    /* 4. Start WiFi UDP as secondary channel (optional) */
    esp_err_t wifi_err = comm_wifi_start();
    if (wifi_err != ESP_OK) {
        ESP_LOGW(TAG, "WiFi start failed (%s), USB-only mode", esp_err_to_name(wifi_err));
    }
#endif

    /* 5. Start 200 Hz control loop (esp_timer on Core 1) */
    ESP_ERROR_CHECK(control_start());

    ESP_LOGI(TAG, "All systems go. Waiting for tracking packets...");

    /* app_main returns — FreeRTOS deletes this task, freeing stack memory.
     * All work continues in comm tasks (Core 0) and control timer (Core 1). */
}
