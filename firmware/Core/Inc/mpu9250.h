/**
 * @file mpu9250.h
 * @brief MPU9250 IMU driver for STM32 HAL (I2C)
 *
 * Reads raw gyroscope data from MPU9250 via I2C.
 * Only gyroscope is used for head motion tracking.
 */

#ifndef MPU9250_H
#define MPU9250_H

#include "main.h"
#include <stdint.h>

/* MPU9250 I2C address (AD0 = LOW) */
#define MPU9250_ADDR        (0x68 << 1)

/* Register addresses */
#define MPU9250_WHO_AM_I    0x75
#define MPU9250_PWR_MGMT_1  0x6B
#define MPU9250_GYRO_CONFIG 0x1B
#define MPU9250_GYRO_XOUT_H 0x43
#define MPU9250_CONFIG      0x1A
#define MPU9250_SMPLRT_DIV  0x19

/* Expected WHO_AM_I response */
#define MPU9250_WHO_AM_I_VAL 0x71

/* Gyroscope full-scale range */
typedef enum {
    GYRO_FS_250DPS  = 0x00,
    GYRO_FS_500DPS  = 0x08,
    GYRO_FS_1000DPS = 0x10,
    GYRO_FS_2000DPS = 0x18
} MPU9250_GyroFS;

/* Gyro raw data structure */
typedef struct {
    int16_t x;
    int16_t y;
    int16_t z;
} MPU9250_GyroRaw;

/**
 * @brief Initialize MPU9250 sensor
 * @param hi2c Pointer to I2C handle
 * @return HAL_OK on success, HAL_ERROR on failure
 */
HAL_StatusTypeDef MPU9250_Init(I2C_HandleTypeDef *hi2c);

/**
 * @brief Verify MPU9250 identity via WHO_AM_I register
 * @param hi2c Pointer to I2C handle
 * @return HAL_OK if device responds correctly
 */
HAL_StatusTypeDef MPU9250_WhoAmI(I2C_HandleTypeDef *hi2c);

/**
 * @brief Read raw gyroscope data (all 3 axes)
 * @param hi2c Pointer to I2C handle
 * @param gyro Pointer to output structure
 * @return HAL_OK on success
 */
HAL_StatusTypeDef MPU9250_ReadGyro(I2C_HandleTypeDef *hi2c, MPU9250_GyroRaw *gyro);

#endif /* MPU9250_H */
