#!/usr/bin/env python3
"""
Real-time signal visualization for debugging.

Displays live EOG and IMU data streams in matplotlib plots.
Useful for:
  - Verifying hardware connections
  - Setting threshold values
  - Observing signal patterns

Usage:
    python -m scripts.visualize --port COM4
    python -m scripts.visualize --simulate
"""

import argparse
import os
import sys
from collections import deque

import numpy as np

# Add parent directory to path
sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

from eog_cursor import config


def run_visualization(source, window_seconds: float = 5.0):
    """
    Run real-time matplotlib visualization of sensor data.

    Shows three subplots:
      1. Raw vertical EOG (eog_v) with threshold lines
      2. Raw horizontal EOG (eog_h) with threshold lines
      3. IMU gyroscope (3 axes)
    """
    import matplotlib.pyplot as plt
    import matplotlib.animation as animation

    window_size = int(window_seconds * config.SAMPLE_RATE)
    eog_v_buf = deque(maxlen=window_size)
    eog_h_buf = deque(maxlen=window_size)
    gx_buf = deque(maxlen=window_size)
    gy_buf = deque(maxlen=window_size)
    gz_buf = deque(maxlen=window_size)

    stream = source.stream()

    fig, (ax1, ax2, ax3) = plt.subplots(3, 1, figsize=(12, 8))
    fig.suptitle("EOG Cursor Control - Live Signal Monitor (Dual Channel)")

    # Initialize lines
    time_axis = np.linspace(-window_seconds, 0, window_size)

    line_eog_v, = ax1.plot([], [], 'b-', linewidth=0.8, label='Vertical EOG (eog_v)')
    ax1.axhline(y=config.BLINK_THRESHOLD, color='r', linestyle='--',
                alpha=0.7, label=f'Blink threshold ({config.BLINK_THRESHOLD})')
    ax1.axhline(y=config.LOOK_UP_THRESHOLD, color='orange', linestyle='--',
                alpha=0.5, label=f'Look-up ({config.LOOK_UP_THRESHOLD})')
    ax1.set_ylabel("ADC Value")
    ax1.set_title("Vertical EOG (blinks, up/down)")
    ax1.set_ylim(0, 4095)
    ax1.legend(loc='upper right', fontsize=8)
    ax1.grid(True, alpha=0.3)

    line_eog_h, = ax2.plot([], [], 'm-', linewidth=0.8, label='Horizontal EOG (eog_h)')
    ax2.axhline(y=config.LOOK_RIGHT_THRESHOLD, color='orange', linestyle='--',
                alpha=0.5, label=f'Look-right ({config.LOOK_RIGHT_THRESHOLD})')
    ax2.axhline(y=config.LOOK_LEFT_THRESHOLD, color='cyan', linestyle='--',
                alpha=0.5, label=f'Look-left ({config.LOOK_LEFT_THRESHOLD})')
    ax2.set_ylabel("ADC Value")
    ax2.set_title("Horizontal EOG (left/right)")
    ax2.set_ylim(0, 4095)
    ax2.legend(loc='upper right', fontsize=8)
    ax2.grid(True, alpha=0.3)

    line_gx, = ax3.plot([], [], 'r-', linewidth=0.8, label='Gyro X (pitch)')
    line_gy, = ax3.plot([], [], 'g-', linewidth=0.8, label='Gyro Y (yaw)')
    line_gz, = ax3.plot([], [], 'b-', linewidth=0.8, label='Gyro Z (roll)')
    ax3.axhline(y=config.GYRO_DEADZONE, color='gray', linestyle=':', alpha=0.5)
    ax3.axhline(y=-config.GYRO_DEADZONE, color='gray', linestyle=':', alpha=0.5)
    ax3.set_ylabel("Raw Gyro")
    ax3.set_xlabel("Time (s)")
    ax3.set_title("IMU Gyroscope")
    ax3.legend(loc='upper right', fontsize=8)
    ax3.grid(True, alpha=0.3)

    def init():
        for ax in (ax1, ax2, ax3):
            ax.set_xlim(-window_seconds, 0)
        return line_eog_v, line_eog_h, line_gx, line_gy, line_gz

    def update(frame):
        # Read a batch of samples
        for _ in range(10):
            try:
                packet = next(stream)
                eog_v_buf.append(packet.eog_v)
                eog_h_buf.append(packet.eog_h)
                gx_buf.append(packet.gyro_x)
                gy_buf.append(packet.gyro_y)
                gz_buf.append(packet.gyro_z)
            except StopIteration:
                break

        n = len(eog_v_buf)
        if n < 2:
            return line_eog_v, line_eog_h, line_gx, line_gy, line_gz

        t = np.linspace(-n / config.SAMPLE_RATE, 0, n)

        line_eog_v.set_data(t, list(eog_v_buf))
        line_eog_h.set_data(t, list(eog_h_buf))

        line_gx.set_data(t, list(gx_buf))
        line_gy.set_data(t, list(gy_buf))
        line_gz.set_data(t, list(gz_buf))

        # Auto-scale gyro
        all_gyro = list(gx_buf) + list(gy_buf) + list(gz_buf)
        if all_gyro:
            g_max = max(abs(min(all_gyro)), abs(max(all_gyro)), 1000) * 1.2
            ax3.set_ylim(-g_max, g_max)

        return line_eog_v, line_eog_h, line_gx, line_gy, line_gz

    ani = animation.FuncAnimation(
        fig, update, init_func=init,
        interval=50,  # 20 FPS display update
        blit=False, cache_frame_data=False
    )

    plt.tight_layout()
    plt.show()


def main():
    parser = argparse.ArgumentParser(description="Real-time signal visualization")
    parser.add_argument("--port", default=config.SERIAL_PORT,
                        help=f"Serial port (default: {config.SERIAL_PORT})")
    parser.add_argument("--simulate", action="store_true",
                        help="Use simulator instead of hardware")
    parser.add_argument("--window", type=float, default=5.0,
                        help="Display window in seconds (default: 5)")
    args = parser.parse_args()

    if args.simulate:
        from eog_cursor.simulator import HardwareSimulator
        source = HardwareSimulator()
    else:
        from eog_cursor.serial_reader import SerialReader
        source = SerialReader(port=args.port)
        source.connect()

    try:
        run_visualization(source, window_seconds=args.window)
    except KeyboardInterrupt:
        print("\nVisualization stopped.")
    finally:
        if hasattr(source, 'disconnect'):
            source.disconnect()


if __name__ == "__main__":
    main()
