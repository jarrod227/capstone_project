/* USER CODE BEGIN Header */
/**
  ******************************************************************************
  * @file           : main.c
  * @brief          : Main program body
  *
  * EOG Cursor Control - STM32 Firmware
  *
  * Data acquisition node: reads EOG (ADC) + IMU gyro (I2C),
  * transmits CSV packets over UART-DMA at 200Hz.
  *
  * Hardware connections:
  *   - AD8232 #1 OUT -> PA0 (ADC1 Channel 1) - vertical EOG
  *   - AD8232 #2 OUT -> PA4 (ADC2 Channel 1) - horizontal EOG
  *   - MPU9250 SDA   -> PB7 (I2C1)
  *   - MPU9250 SCL   -> PB6 (I2C1)
  *   - USB           -> USART2 (virtual COM port)
  *
  * Output format: "timestamp,eog_v,eog_h,gyro_x,gyro_y,gyro_z\r\n"
  *
  * NOTE: This file only contains USER CODE sections.
  *       CubeMX generates the full file from the .ioc configuration.
  *       Copy the USER CODE blocks into the CubeMX-generated main.c.
  *       In CubeMX peripheral init order, DMA must come BEFORE USART2.
  *
  ******************************************************************************
  */
/* USER CODE END Header */
/* Includes ------------------------------------------------------------------*/
#include "main.h"
/* USER CODE BEGIN Includes */
#include "mpu9250.h"
#include <stdio.h>
#include <string.h>
/* USER CODE END Includes */

/* USER CODE BEGIN PV */
static char tx_buf[2][80];                /* Ping-pong buffers for DMA */
static volatile uint8_t tx_idx = 0;       /* Index of buffer being filled */
static volatile uint8_t dma_busy = 0;     /* 1 while DMA transfer in progress */
static volatile uint8_t dma_stuck = 0;    /* Ticks since DMA started (watchdog) */
static volatile uint8_t tick_200hz = 0;   /* Set by TIM6 ISR every 5ms */
static MPU9250_GyroRaw gyro_data;
/* USER CODE END PV */

int main(void)
{
  /* USER CODE BEGIN 2 */

    /* Initialize IMU */
    uint8_t imu_ok = 0;
    if (MPU9250_Init(&hi2c1) == HAL_OK) {
        imu_ok = 1;
    } else {
        HAL_GPIO_WritePin(GPIOA, GPIO_PIN_5, GPIO_PIN_SET);
    }

    HAL_Delay(500);

    /* Start 200Hz timer — drives the main loop via interrupt flag */
    HAL_TIM_Base_Start_IT(&htim6);

  /* USER CODE END 2 */

  /* USER CODE BEGIN WHILE */
    while (1)
    {
        /* Wait for TIM6 interrupt (precise 5ms period) */
        while (!tick_200hz) {}
        tick_200hz = 0;

        uint32_t timestamp = HAL_GetTick();

        /* --- Read vertical EOG via ADC1 --- */
        HAL_ADC_Start(&hadc1);
        HAL_ADC_PollForConversion(&hadc1, 10);
        uint16_t eog_v = HAL_ADC_GetValue(&hadc1);
        HAL_ADC_Stop(&hadc1);

        /* --- Read horizontal EOG via ADC2 --- */
        HAL_ADC_Start(&hadc2);
        HAL_ADC_PollForConversion(&hadc2, 10);
        uint16_t eog_h = HAL_ADC_GetValue(&hadc2);
        HAL_ADC_Stop(&hadc2);

        /* --- Read IMU gyroscope via I2C --- */
        if (imu_ok && MPU9250_ReadGyro(&hi2c1, &gyro_data) != HAL_OK) {
            gyro_data.x = 0;
            gyro_data.y = 0;
            gyro_data.z = 0;
        }

        /* --- Format CSV into current fill buffer --- */
        int len = snprintf(tx_buf[tx_idx], sizeof(tx_buf[0]),
                           "%lu,%u,%u,%d,%d,%d\r\n",
                           timestamp,
                           eog_v,
                           eog_h,
                           gyro_data.x,
                           gyro_data.y,
                           gyro_data.z);

        /* --- DMA watchdog: force recovery if stuck > 10ms (2 ticks) --- */
        if (dma_busy) {
            dma_stuck++;
            if (dma_stuck > 2) {
                HAL_UART_AbortTransmit(&huart2);
                dma_busy = 0;
                dma_stuck = 0;
            }
        }

        /* --- Transmit via DMA (non-blocking) --- */
        if (!dma_busy) {
            uint8_t send_idx = tx_idx;
            tx_idx ^= 1;
            dma_busy = 1;
            dma_stuck = 0;
            if (HAL_UART_Transmit_DMA(&huart2, (uint8_t *)tx_buf[send_idx], len) != HAL_OK) {
                HAL_UART_AbortTransmit(&huart2);
                dma_busy = 0;
            }
        }
    }
    /* USER CODE END WHILE */
}

/* USER CODE BEGIN 4 */

/* IMPORTANT: These three callbacks are required for DMA mode.
 * Copy them into USER CODE BEGIN 4 / USER CODE END 4 in
 * the CubeMX-generated main.c. Without them:
 *   - tick_200hz stays 0 -> main loop hangs
 *   - dma_busy stays 1   -> only first frame ever sent
 */

void HAL_UART_TxCpltCallback(UART_HandleTypeDef *huart)
{
    if (huart->Instance == USART2) {
        dma_busy = 0;
        dma_stuck = 0;
    }
}

void HAL_UART_ErrorCallback(UART_HandleTypeDef *huart)
{
    if (huart->Instance == USART2) {
        __HAL_UART_CLEAR_OREFLAG(huart);
        __HAL_UART_CLEAR_FEFLAG(huart);
        __HAL_UART_CLEAR_NEFLAG(huart);
        __HAL_UART_CLEAR_PEFLAG(huart);
        HAL_UART_AbortTransmit(huart);
        dma_busy = 0;
        dma_stuck = 0;
    }
}

void HAL_TIM_PeriodElapsedCallback(TIM_HandleTypeDef *htim)
{
    if (htim->Instance == TIM6) {
        tick_200hz = 1;
    }
}

/* USER CODE END 4 */
