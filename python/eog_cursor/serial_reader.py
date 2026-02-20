"""
Serial data reader for STM32 sensor data.

Reads CSV-formatted lines from the STM32 UART and parses them
into structured data packets.
"""

import logging
import time
from dataclasses import dataclass

from . import config

logger = logging.getLogger(__name__)


@dataclass
class SensorPacket:
    """Single data packet from STM32.

    Contains dual-channel EOG data:
      eog_v: vertical   EOG channel (12-bit ADC, 0-4095)
      eog_h: horizontal EOG channel (12-bit ADC, 0-4095)
    """
    timestamp: int      # ms since STM32 boot
    eog_v: int          # 12-bit ADC value, vertical EOG channel
    eog_h: int          # 12-bit ADC value, horizontal EOG channel
    gyro_x: int         # Raw gyroscope X
    gyro_y: int         # Raw gyroscope Y
    gyro_z: int         # Raw gyroscope Z
    pc_time: float      # PC-side timestamp (time.time())


class SerialReader:
    """Reads and parses sensor data from STM32 via USB serial."""

    def __init__(self, port=None, baudrate=None):
        self.port = port or config.SERIAL_PORT
        self.baudrate = baudrate or config.SERIAL_BAUDRATE
        self.ser = None
        self._error_count = 0

    def connect(self):
        """Open serial connection to STM32."""
        import serial  # lazy import: not needed for simulate/replay modes

        logger.info(f"Connecting to {self.port} at {self.baudrate} baud...")
        self.ser = serial.Serial(
            port=self.port,
            baudrate=self.baudrate,
            timeout=config.SERIAL_TIMEOUT
        )
        # Flush stale data
        self.ser.reset_input_buffer()
        logger.info("Serial connection established.")

    def disconnect(self):
        """Close serial connection."""
        if self.ser and self.ser.is_open:
            self.ser.close()
            logger.info("Serial connection closed.")

    def read_packet(self) -> SensorPacket | None:
        """
        Read and parse one line from serial.

        Returns:
            SensorPacket if parsing succeeds, None if line is malformed.

        Expected format: "timestamp,eog_v,eog_h,gyro_x,gyro_y,gyro_z\\r\\n"
        """
        if not self.ser or not self.ser.is_open:
            raise ConnectionError("Serial port not connected")

        try:
            raw_line = self.ser.readline()
            if not raw_line:
                return None

            line = raw_line.decode("ascii", errors="ignore").strip()
            if not line:
                return None

            parts = line.split(",")
            if len(parts) == 6:
                # Dual-channel: timestamp,eog_v,eog_h,gx,gy,gz
                return SensorPacket(
                    timestamp=int(parts[0]),
                    eog_v=int(parts[1]),
                    eog_h=int(parts[2]),
                    gyro_x=int(parts[3]),
                    gyro_y=int(parts[4]),
                    gyro_z=int(parts[5]),
                    pc_time=time.time()
                )
            elif len(parts) == 5:
                # Legacy single-channel: timestamp,eog_v,gx,gy,gz
                return SensorPacket(
                    timestamp=int(parts[0]),
                    eog_v=int(parts[1]),
                    eog_h=config.EOG_BASELINE,
                    gyro_x=int(parts[2]),
                    gyro_y=int(parts[3]),
                    gyro_z=int(parts[4]),
                    pc_time=time.time()
                )
            else:
                self._error_count += 1
                if self._error_count % 100 == 1:
                    logger.warning(f"Malformed line ({self._error_count} total): {line!r}")
                return None

        except (ValueError, UnicodeDecodeError) as e:
            self._error_count += 1
            if self._error_count % 100 == 1:
                logger.warning(f"Parse error ({self._error_count} total): {e}")
            return None

    def stream(self):
        """Generator that yields SensorPackets continuously."""
        while True:
            packet = self.read_packet()
            if packet is not None:
                yield packet

    def __enter__(self):
        self.connect()
        return self

    def __exit__(self, exc_type, exc_val, exc_tb):
        self.disconnect()
        return False
