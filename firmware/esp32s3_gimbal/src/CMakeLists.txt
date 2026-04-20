FILE(GLOB_RECURSE app_sources ${CMAKE_SOURCE_DIR}/src/*.c)

idf_component_register(
    SRCS ${app_sources}
    INCLUDE_DIRS "${CMAKE_SOURCE_DIR}/include"
    REQUIRES
        esp_common
        esp_driver_ledc
        esp_driver_usb_serial_jtag
        esp_event
        esp_netif
        esp_timer
        esp_wifi
        freertos
        log
        lwip
        nvs_flash
        esp-dsp
)
