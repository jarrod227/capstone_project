"""
CSV file replay source for offline demo and debugging.

Reads a previously recorded CSV data file and replays it as if
the data were streaming from the STM32 in real time.

Supports two modes:
  - Real-time: Plays back at the original sample rate (200Hz)
  - Fast: Replays as fast as possible (for batch processing)
"""

import time
import logging

import pandas as pd

from . import config
from .serial_reader import SensorPacket

logger = logging.getLogger(__name__)


class CSVReplaySource:
    """
    Replays sensor data from a CSV file.

    Mimics the SerialReader/HardwareSimulator interface so it can
    be used as a drop-in replacement in the pipeline.
    """

    def __init__(self, csv_path: str, realtime: bool = True, loop: bool = False):
        """
        Args:
            csv_path: Path to CSV file with columns:
                      timestamp, eog_v, eog_h, gyro_x, gyro_y, gyro_z [, label]
            realtime: If True, replay at original sample rate.
                      If False, yield as fast as possible.
            loop:     If True, loop the file continuously.
        """
        self.csv_path = csv_path
        self.realtime = realtime
        self.loop = loop
        self.data = None

    def load(self):
        """Load the CSV file into memory."""
        self.data = pd.read_csv(self.csv_path)

        # Backward compatibility: rename legacy 'eog' column to 'eog_v'
        if 'eog' in self.data.columns and 'eog_v' not in self.data.columns:
            self.data.rename(columns={'eog': 'eog_v'}, inplace=True)
        if 'eog_h' not in self.data.columns:
            self.data['eog_h'] = config.EOG_BASELINE

        required = {'eog_v', 'gyro_x', 'gyro_y', 'gyro_z'}
        missing = required - set(self.data.columns)
        if missing:
            raise ValueError(f"CSV missing required columns: {missing}")

        if 'timestamp' not in self.data.columns:
            self.data['timestamp'] = range(0, len(self.data) * 5, 5)

        logger.info(f"Loaded {len(self.data)} samples from {self.csv_path}")

        if 'label' in self.data.columns:
            counts = self.data['label'].value_counts()
            logger.info(f"Labels: {dict(counts)}")

    def stream(self):
        """
        Generator that yields SensorPackets from the CSV data.

        If realtime=True, sleeps between samples to match 200Hz rate.
        """
        if self.data is None:
            self.load()

        while True:
            start_time = time.time()

            for idx, row in self.data.iterrows():
                packet = SensorPacket(
                    timestamp=int(row['timestamp']),
                    eog_v=int(row['eog_v']),
                    eog_h=int(row['eog_h']),
                    gyro_x=int(row['gyro_x']),
                    gyro_y=int(row['gyro_y']),
                    gyro_z=int(row['gyro_z']),
                    pc_time=time.time()
                )
                yield packet

                if self.realtime:
                    # Calculate expected time for next sample
                    elapsed = idx + 1
                    expected_time = start_time + elapsed * config.SAMPLE_PERIOD
                    sleep_time = expected_time - time.time()
                    if sleep_time > 0:
                        time.sleep(sleep_time)

            if not self.loop:
                break

            logger.info("Replay loop: restarting from beginning")

    @property
    def duration_seconds(self) -> float:
        """Total duration of the recording in seconds."""
        if self.data is not None:
            return len(self.data) / config.SAMPLE_RATE
        return 0.0

    @property
    def num_samples(self) -> int:
        """Total number of samples."""
        if self.data is not None:
            return len(self.data)
        return 0
