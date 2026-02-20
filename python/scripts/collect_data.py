#!/usr/bin/env python3
"""
EOG Training Data Collection Script

Records timestamped, labeled EOG + IMU data for SVM training.
Uses keyboard shortcuts to label data in real time.

Usage:
    python -m scripts.collect_data --port COM4
    python -m scripts.collect_data --simulate  # No hardware needed

Controls:
    0 = idle          1 = blink         2 = double_blink
    3 = long_blink    4 = look_up       5 = look_down
    6 = look_left     7 = look_right    ESC = stop and save

Labeling procedure:
    The subject presses a label key BEFORE performing the gesture,
    holds it during the gesture, then presses 0 (idle) after.
    For blink events (which are very fast, ~100-500ms), press the
    label key ~1s before blinking, perform the blink, then press 0.
    Example: press 2 → wait 1s → double blink → wait 1s → press 0.
"""

import argparse
import csv
import os
import sys
import time

# Add parent directory to path
sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

from eog_cursor import config
from eog_cursor.serial_reader import SerialReader
from eog_cursor.simulator import HardwareSimulator

# Label mapping — subject presses key before gesture, holds during, presses 0 after
LABEL_KEYS = {
    '0': 'idle',
    '1': 'blink',
    '2': 'double_blink',
    '3': 'long_blink',
    '4': 'look_up',
    '5': 'look_down',
    '6': 'look_left',
    '7': 'look_right',
}


def run_collection(source, output_path: str):
    """
    Collect labeled data from a sensor source.

    Args:
        source: SerialReader or HardwareSimulator instance
        output_path: CSV file path for output
    """
    from pynput import keyboard

    current_label = 'idle'
    running = True
    sample_count = 0
    label_counts = {label: 0 for label in config.ML_CLASSES}

    def on_press(key):
        nonlocal current_label, running
        try:
            if hasattr(key, 'char') and key.char in LABEL_KEYS:
                current_label = LABEL_KEYS[key.char]
            elif key == keyboard.Key.esc:
                running = False
        except AttributeError:
            pass

    listener = keyboard.Listener(on_press=on_press)
    listener.daemon = True
    listener.start()

    os.makedirs(os.path.dirname(output_path), exist_ok=True)

    print("=" * 60)
    print("EOG Data Collection")
    print("=" * 60)
    print("Label keys: 0=idle 1=blink 2=double_blink 3=long_blink")
    print("            4=up 5=down 6=left 7=right")
    print("Tip: press label key BEFORE gesture, press 0 after.")
    print(f"Output: {output_path}")
    print("Press ESC to stop.\n")

    with open(output_path, 'w', newline='') as f:
        writer = csv.writer(f)
        writer.writerow(['timestamp', 'eog_v', 'eog_h', 'gyro_x', 'gyro_y', 'gyro_z', 'label'])

        for packet in source.stream():
            if not running:
                break

            writer.writerow([
                packet.timestamp,
                packet.eog_v,
                packet.eog_h,
                packet.gyro_x,
                packet.gyro_y,
                packet.gyro_z,
                current_label
            ])
            f.flush()

            sample_count += 1
            if current_label in label_counts:
                label_counts[current_label] += 1

            if sample_count % 200 == 0:  # Print status every second
                elapsed = sample_count / config.SAMPLE_RATE
                status = f"\r[{elapsed:.0f}s] Samples: {sample_count} | "
                status += f"Label: {current_label:>10} | "
                status += " ".join(f"{k}:{v}" for k, v in label_counts.items() if v > 0)
                print(status, end="", flush=True)

    listener.stop()

    print(f"\n\nCollection complete!")
    print(f"Total samples: {sample_count}")
    print(f"Saved to: {output_path}")
    print("\nLabel distribution:")
    for label, count in label_counts.items():
        if count > 0:
            pct = count / sample_count * 100 if sample_count > 0 else 0
            print(f"  {label:>12}: {count:>6} ({pct:.1f}%)")


def main():
    parser = argparse.ArgumentParser(description="Collect labeled EOG training data")
    parser.add_argument("--port", default=config.SERIAL_PORT,
                        help=f"Serial port (default: {config.SERIAL_PORT})")
    parser.add_argument("--baudrate", type=int, default=config.SERIAL_BAUDRATE,
                        help=f"Baud rate (default: {config.SERIAL_BAUDRATE})")
    parser.add_argument("--simulate", action="store_true",
                        help="Use keyboard simulator instead of hardware")
    parser.add_argument("--output", default=None,
                        help="Output CSV path (default: auto-generated)")
    args = parser.parse_args()

    # Generate output filename with timestamp
    if args.output is None:
        timestamp = time.strftime("%Y%m%d_%H%M%S")
        args.output = os.path.join(config.COLLECT_OUTPUT_DIR,
                                   f"eog_session_{timestamp}.csv")

    if args.simulate:
        source = HardwareSimulator()
    else:
        source = SerialReader(port=args.port, baudrate=args.baudrate)
        source.connect()

    try:
        run_collection(source, args.output)
    except KeyboardInterrupt:
        print("\nInterrupted by user.")
    finally:
        if isinstance(source, SerialReader):
            source.disconnect()


if __name__ == "__main__":
    main()
