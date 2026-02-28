/*
 * SafetyRobot - ESP32 Basic Initialization
 * LED Blink on GPIO 2 (onboard LED for most ESP32 dev boards)
 */

#include <stdio.h>
#include "freertos/FreeRTOS.h"
#include "freertos/task.h"
#include "driver/gpio.h"
#include "esp_log.h"

#define LED_GPIO    GPIO_NUM_2
#define BLINK_DELAY_MS  500

static const char *TAG = "SafetyRobot";

static void led_init(void)
{
    gpio_config_t io_conf = {
        .pin_bit_mask = (1ULL << LED_GPIO),
        .mode         = GPIO_MODE_OUTPUT,
        .pull_up_en   = GPIO_PULLUP_DISABLE,
        .pull_down_en = GPIO_PULLDOWN_DISABLE,
        .intr_type    = GPIO_INTR_DISABLE,
    };
    gpio_config(&io_conf);
}

static void blink_task(void *arg)
{
    uint8_t state = 0;

    while (1) {
        state ^= 1;
        gpio_set_level(LED_GPIO, state);
        ESP_LOGI(TAG, "LED %s", state ? "ON" : "OFF");
        vTaskDelay(pdMS_TO_TICKS(BLINK_DELAY_MS));
    }
}

void app_main(void)
{
    ESP_LOGI(TAG, "SafetyRobot initializing...");

    led_init();

    xTaskCreate(blink_task, "blink_task", 2048, NULL, 5, NULL);

    ESP_LOGI(TAG, "Init complete.");
}
