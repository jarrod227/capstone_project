/**
 * @file mpu9250.c
 * @brief MPU9250 IMU driver implementation
 */

#include "mpu9250.h"

#define I2C_TIMEOUT 100  /* ms */

/**
 * Write a single byte to an MPU9250 register
 */
static HAL_StatusTypeDef MPU9250_WriteReg(I2C_HandleTypeDef *hi2c,
                                           uint8_t reg, uint8_t val)
{
    return HAL_I2C_Mem_Write(hi2c, MPU9250_ADDR, reg,
                             I2C_MEMADD_SIZE_8BIT, &val, 1, I2C_TIMEOUT);
}

/**
 * Read bytes from MPU9250 registers
 */
static HAL_StatusTypeDef MPU9250_ReadRegs(I2C_HandleTypeDef *hi2c,
                                           uint8_t reg, uint8_t *buf,
                                           uint16_t len)
{
    return HAL_I2C_Mem_Read(hi2c, MPU9250_ADDR, reg,
                            I2C_MEMADD_SIZE_8BIT, buf, len, I2C_TIMEOUT);
}

HAL_StatusTypeDef MPU9250_WhoAmI(I2C_HandleTypeDef *hi2c)
{
    uint8_t id = 0;
    HAL_StatusTypeDef status;

    status = MPU9250_ReadRegs(hi2c, MPU9250_WHO_AM_I, &id, 1);
    if (status != HAL_OK) return status;

    /* MPU9250 should return 0x71, MPU6050 returns 0x68 */
    if (id != MPU9250_WHO_AM_I_VAL && id != 0x68) {
        return HAL_ERROR;
    }

    return HAL_OK;
}

HAL_StatusTypeDef MPU9250_Init(I2C_HandleTypeDef *hi2c)
{
    HAL_StatusTypeDef status;

    /* Wake up: clear sleep bit in PWR_MGMT_1 */
    status = MPU9250_WriteReg(hi2c, MPU9250_PWR_MGMT_1, 0x00);
    if (status != HAL_OK) return status;

    HAL_Delay(100);  /* Wait for sensor to stabilize */

    /* Verify device identity */
    status = MPU9250_WhoAmI(hi2c);
    if (status != HAL_OK) return status;

    /* Enable DLPF: ~42Hz bandwidth, Fs = 1kHz (reduces gyro noise) */
    status = MPU9250_WriteReg(hi2c, MPU9250_CONFIG, 0x03);
    if (status != HAL_OK) return status;

    /* Set sample rate divider: 200Hz = 1kHz / (1 + 4)
     * NOTE: SMPLRT_DIV uses 1kHz base only when DLPF is enabled
     * (DLPF_CFG = 1-6). Without DLPF the base is 8kHz. */
    status = MPU9250_WriteReg(hi2c, MPU9250_SMPLRT_DIV, 0x04);
    if (status != HAL_OK) return status;

    /* Configure gyroscope: +/- 500 dps */
    status = MPU9250_WriteReg(hi2c, MPU9250_GYRO_CONFIG, GYRO_FS_500DPS);
    if (status != HAL_OK) return status;

    return HAL_OK;
}

HAL_StatusTypeDef MPU9250_ReadGyro(I2C_HandleTypeDef *hi2c,
                                    MPU9250_GyroRaw *gyro)
{
    uint8_t buf[6];
    HAL_StatusTypeDef status;

    /* Read 6 bytes starting from GYRO_XOUT_H (0x43) */
    status = MPU9250_ReadRegs(hi2c, MPU9250_GYRO_XOUT_H, buf, 6);
    if (status != HAL_OK) return status;

    /* Combine high and low bytes (big-endian from MPU) */
    gyro->x = (int16_t)((buf[0] << 8) | buf[1]);
    gyro->y = (int16_t)((buf[2] << 8) | buf[3]);
    gyro->z = (int16_t)((buf[4] << 8) | buf[5]);

    return HAL_OK;
}
