#!/usr/bin/env python3
"""
EOG Cursor Control System - Main Entry Point

Runs the complete pipeline: serial read -> signal processing -> cursor control.

Modes:
  threshold  - Simple threshold-based control (recommended first)
  statespace - Physics-based state-space model with inertia
  ml         - SVM-based EOG classification

Usage:
    # With hardware (Linux: /dev/ttyACM0, Windows: COM4):
    python main.py --port /dev/ttyACM0 --mode threshold
    python main.py --port /dev/ttyACM0 --mode statespace
    python main.py --port /dev/ttyACM0 --mode ml

    # Without hardware (simulator):
    python main.py --simulate --mode threshold
    python main.py --simulate --mode statespace
    python main.py --simulate --mode ml

    # Replay from CSV (offline):
    python main.py --replay ../data/raw/demo_replay.csv --mode threshold
    python main.py --replay ../data/raw/demo_replay.csv --mode statespace
    python main.py --replay ../data/raw/demo_replay.csv --mode ml
"""

import argparse
import logging
import sys
import time

from eog_cursor import config
from eog_cursor.cursor_control import ThresholdController, StateSpaceController
from eog_cursor.signal_processing import EOGLowPassFilter, GyroCalibrator, GyroKalmanFilter3Axis


def setup_logging(verbose: bool = False):
    """Configure logging output."""
    level = logging.DEBUG if verbose else logging.INFO
    logging.basicConfig(
        level=level,
        format="%(asctime)s [%(levelname)s] %(name)s: %(message)s",
        datefmt="%H:%M:%S"
    )


def run_threshold_mode(source, calibrator=None, kalman=None):
    """Run cursor control with threshold-based event detection."""
    controller = ThresholdController()

    if config.EOG_LOWPASS_ENABLED:
        filter_v = EOGLowPassFilter()
        filter_h = EOGLowPassFilter()
    else:
        filter_v = None
        filter_h = None

    print("Threshold mode active.")
    print(f"  Blink threshold:  {config.BLINK_THRESHOLD}")
    print(f"  Double blink:     2 blinks within {config.DOUBLE_BLINK_WINDOW}s → left click")
    print(f"  Long blink:       hold >={config.LONG_BLINK_MIN_DURATION}s → right click")
    print(f"  Scroll:           eye gaze + head tilt fusion")
    print(f"  Window switch:    head roll flick (gyro_z)")

    for packet in source.stream():
        eog_v = float(packet.eog_v)
        eog_h = float(packet.eog_h)
        if filter_v:
            eog_v = filter_v.filter_sample(eog_v)
            eog_h = filter_h.filter_sample(eog_h)
        gx, gy, gz = packet.gyro_x, packet.gyro_y, packet.gyro_z
        if kalman:
            gx, gy, gz = kalman.update(gx, gy, gz)
        elif calibrator:
            gx, gy, gz = calibrator.correct(gx, gy, gz)
        controller.update(eog_v, eog_h, gx, gy, gz)


def run_statespace_mode(source, calibrator=None, kalman=None):
    """Run cursor control with state-space physics model."""
    controller = StateSpaceController()

    if config.EOG_LOWPASS_ENABLED:
        filter_v = EOGLowPassFilter()
        filter_h = EOGLowPassFilter()
    else:
        filter_v = None
        filter_h = None

    print("State-space mode active.")
    print(f"  Velocity retain: {config.SS_VELOCITY_RETAIN}")
    print(f"  Sensitivity:  {config.SS_SENSITIVITY}")
    print(f"  Deadzone:     {config.GYRO_DEADZONE}")

    for packet in source.stream():
        eog_v = float(packet.eog_v)
        eog_h = float(packet.eog_h)
        if filter_v:
            eog_v = filter_v.filter_sample(eog_v)
            eog_h = filter_h.filter_sample(eog_h)
        gx, gy, gz = packet.gyro_x, packet.gyro_y, packet.gyro_z
        if kalman:
            gx, gy, gz = kalman.update(gx, gy, gz)
        elif calibrator:
            gx, gy, gz = calibrator.correct(gx, gy, gz)
        controller.update(eog_v, eog_h, gx, gy, gz)


def run_ml_mode(source, calibrator=None, kalman=None):
    """Run cursor control with ML-based EOG classification + sensor fusion."""
    from eog_cursor.ml_classifier import EOGClassifier

    classifier = EOGClassifier()
    if not classifier.load():
        print("ERROR: No trained model found.")
        print("Run training first: python -m scripts.train_model --generate-demo")
        sys.exit(1)

    controller = StateSpaceController()
    deadzone = config.GYRO_DEADZONE

    print("ML mode active.")
    print(f"  Model: {config.ML_MODEL_PATH}")
    print(f"  Window size: {config.ML_WINDOW_SIZE}")
    print(f"  Sensor fusion: scroll and back/fwd require eye + head agreement")

    import pyautogui
    pyautogui.FAILSAFE = False
    pyautogui.PAUSE = 0

    # Per-action cooldowns (consistent with threshold/statespace modes)
    last_blink_time = 0.0
    last_scroll_time = 0.0
    last_nav_time = 0.0
    last_prediction = "idle"  # Remember last ML result for continuous suppression

    for packet in source.stream():
        # ML classification uses raw EOG values (must match training data)
        prediction = classifier.predict(float(packet.eog_v), float(packet.eog_h))

        # predict() returns None for 19 out of 20 samples (ML_WINDOW_STEP).
        # Remember last non-None prediction so cursor suppression is continuous.
        if prediction is not None:
            last_prediction = prediction

        now = time.time()
        gx, gy = packet.gyro_x, packet.gyro_y
        gz = packet.gyro_z
        if kalman:
            gx, gy, gz = kalman.update(gx, gy, gz)
        elif calibrator:
            gx, gy, gz = calibrator.correct(gx, gy, gz)

        if prediction and prediction != "idle":
            action = None

            if prediction == "double_blink":
                if now - last_blink_time > config.DOUBLE_BLINK_COOLDOWN:
                    pyautogui.click(_pause=False)
                    action = "left click"
                    last_blink_time = now
            elif prediction == "long_blink":
                if now - last_blink_time > config.LONG_BLINK_COOLDOWN:
                    pyautogui.click(button='right', _pause=False)
                    action = "right click"
                    last_blink_time = now

            # Scroll fusion: ML detects eye gaze + check head tilt
            elif prediction == "look_up" and gx < -deadzone:
                if now - last_scroll_time > config.SCROLL_COOLDOWN:
                    amount = max(1, int(abs(gx) / deadzone * config.SCROLL_AMOUNT))
                    pyautogui.scroll(amount, _pause=False)
                    action = f"scroll up {amount} lines (eye + head fusion)"
                    last_scroll_time = now
            elif prediction == "look_down" and gx > deadzone:
                if now - last_scroll_time > config.SCROLL_COOLDOWN:
                    amount = max(1, int(abs(gx) / deadzone * config.SCROLL_AMOUNT))
                    pyautogui.scroll(-amount, _pause=False)
                    action = f"scroll down {amount} lines (eye + head fusion)"
                    last_scroll_time = now

            # Back/forward fusion: ML detects eye gaze + check head turn
            elif prediction == "look_left" and gy < -deadzone:
                if now - last_nav_time > config.HORIZONTAL_GAZE_COOLDOWN:
                    pyautogui.hotkey('alt', 'left', _pause=False)
                    action = "browser back (eye + head fusion)"
                    last_nav_time = now
            elif prediction == "look_right" and gy > deadzone:
                if now - last_nav_time > config.HORIZONTAL_GAZE_COOLDOWN:
                    pyautogui.hotkey('alt', 'right', _pause=False)
                    action = "browser forward (eye + head fusion)"
                    last_nav_time = now

            if action:
                logging.getLogger(__name__).info(f"ML: {prediction} -> {action}")

        # IMU controls cursor via state-space controller.
        # Use last_prediction (not prediction) for suppression — prediction is
        # None for 19/20 samples, but suppression must be continuous.
        # Head roll suppression is handled by the controller internally (from gz).
        #
        # When ML detects horizontal gaze (look_left/look_right), pass real
        # gx/gz so the controller's nod/roll detectors can work while the
        # cursor is frozen via cursor_frozen_override.
        cursor_frozen = last_prediction in ("look_left", "look_right")
        if cursor_frozen:
            # Cursor frozen by override; pass real gx/gz for nod/roll detection
            cursor_gx = gx
            cursor_gy = gy
        elif last_prediction != "idle":
            cursor_gx = 0
            cursor_gy = 0
            # Zero velocity to freeze cursor during action
            controller.state[1] = 0
            controller.state[3] = 0
        else:
            cursor_gx = gx
            cursor_gy = gy

        controller.update(
            config.EOG_BASELINE, config.EOG_BASELINE,
            cursor_gx, cursor_gy, gz,
            cursor_frozen_override=cursor_frozen
        )


def main():
    parser = argparse.ArgumentParser(
        description="EOG Cursor Control System",
        formatter_class=argparse.RawDescriptionHelpFormatter,
        epilog="""
Examples:
  python main.py --simulate --mode threshold          # Live simulator demo
  python main.py --port /dev/ttyACM0 --mode statespace # Hardware with physics model
  python main.py --port /dev/ttyACM0 --mode ml        # Hardware with ML classifier
  python main.py --replay ../data/raw/demo_replay.csv # Offline replay from CSV
  python main.py --replay data.csv --replay-loop      # Loop replay continuously
  python main.py --simulate --mode ml                  # ML mode with simulator
        """
    )
    parser.add_argument("--mode", choices=["threshold", "statespace", "ml"],
                        default="threshold",
                        help="Control mode (default: threshold)")
    parser.add_argument("--port", default=config.SERIAL_PORT,
                        help=f"Serial port (default: {config.SERIAL_PORT})")
    parser.add_argument("--baudrate", type=int, default=config.SERIAL_BAUDRATE,
                        help=f"Baud rate (default: {config.SERIAL_BAUDRATE})")
    parser.add_argument("--simulate", action="store_true",
                        help="Use keyboard simulator (no hardware needed)")
    parser.add_argument("--replay", metavar="CSV_FILE",
                        help="Replay a recorded CSV file as data source")
    parser.add_argument("--replay-fast", action="store_true",
                        help="Replay CSV at maximum speed (not real-time)")
    parser.add_argument("--replay-loop", action="store_true",
                        help="Loop CSV replay continuously")
    parser.add_argument("--verbose", "-v", action="store_true",
                        help="Enable debug logging")

    # Tuning parameters
    parser.add_argument("--sensitivity", type=float, default=None,
                        help="Override cursor sensitivity")
    parser.add_argument("--velocity-retain", type=float, default=None,
                        help="Override velocity retention factor (state-space mode)")
    parser.add_argument("--deadzone", type=int, default=None,
                        help="Override gyro deadzone threshold")
    parser.add_argument("--blink-threshold", type=int, default=None,
                        help="Override blink detection threshold")

    args = parser.parse_args()
    setup_logging(args.verbose)

    # Apply parameter overrides
    if args.sensitivity is not None:
        config.CURSOR_SENSITIVITY = args.sensitivity
        config.SS_SENSITIVITY = args.sensitivity
    if args.velocity_retain is not None:
        config.SS_VELOCITY_RETAIN = args.velocity_retain
    if args.deadzone is not None:
        config.GYRO_DEADZONE = args.deadzone
    if args.blink_threshold is not None:
        config.BLINK_THRESHOLD = args.blink_threshold

    # Create data source
    if args.replay:
        from eog_cursor.csv_replay import CSVReplaySource
        source = CSVReplaySource(
            csv_path=args.replay,
            realtime=not args.replay_fast,
            loop=args.replay_loop
        )
        source.load()
    elif args.simulate:
        from eog_cursor.simulator import HardwareSimulator
        source = HardwareSimulator()
    else:
        from eog_cursor.serial_reader import SerialReader
        source = SerialReader(port=args.port, baudrate=args.baudrate)
        source.connect()

    print("=" * 60)
    print("  EOG Cursor Control System")
    print("=" * 60)
    print(f"  Mode:   {args.mode}")
    if args.replay:
        source_str = f"Replay: {args.replay} ({source.num_samples} samples, {source.duration_seconds:.1f}s)"
        if args.replay_fast:
            source_str += " [FAST]"
        if args.replay_loop:
            source_str += " [LOOP]"
    elif args.simulate:
        source_str = "Simulator"
    else:
        source_str = args.port
    print(f"  Source: {source_str}")
    print()

    print("  Actions:")
    print("    Cursor move:    IMU Gyro X/Y")
    print("    Left click:     Double blink (2 quick blinks)")
    print("    Right click:    Long blink (hold eyes closed)")
    print("    Double click:   Look left/right → double head nod (cursor frozen + gyro_x)")
    print("    Scroll up/down: Eye gaze + head tilt (fusion)")
    print("    Back/Fwd:       Eye left/right + head turn (fusion)")
    print("    Window switch:  Look left/right → head roll flick (cursor frozen + gyro_z)")
    print()

    if args.simulate:
        print("  Simulator keys:")
        print("    Arrow keys      → head motion (cursor move)")
        print("    Space (tap x2)  → double blink (left click)")
        print("    Space (hold)    → long blink (right click)")
        print("    N (tap x2)      → double head nod (double click)")
        print("    U + Arrow Up    → look up + head up (scroll up)")
        print("    D + Arrow Down  → look down + head down (scroll down)")
        print("    L + Arrow Left  → look left + head left (browser back)")
        print("    R + Arrow Right → look right + head right (browser fwd)")
        print("    W               → head roll flick (window switch)")
        print("    Q / Escape      → quit")
        print()

    print("  Press Ctrl+C to stop.")
    print("=" * 60)

    # --- Gyro calibration + Kalman filter (skip for simulator) ---
    calibrator = None
    kalman = None
    if not args.simulate:
        print()
        print("  Calibrating gyroscope — keep the device STILL...")
        calibrator = GyroCalibrator()
        bx, by, bz = calibrator.calibrate(source)
        print(f"  Bias: gx={bx:.1f}  gy={by:.1f}  gz={bz:.1f}")
        print(f"  Deadzone: {config.GYRO_DEADZONE}")

        # Initialize Kalman filter with startup bias for faster convergence
        kalman = GyroKalmanFilter3Axis()
        kalman.set_initial_bias(bx, by, bz)
        print(f"  Kalman filter: Q_ω={config.KALMAN_Q_OMEGA}  Q_b={config.KALMAN_Q_BIAS}  R={config.KALMAN_R}")
        print("  Calibration done. Kalman filter active. Starting control loop.")
        print()

    try:
        if args.mode == "threshold":
            run_threshold_mode(source, calibrator, kalman)
        elif args.mode == "statespace":
            run_statespace_mode(source, calibrator, kalman)
        elif args.mode == "ml":
            run_ml_mode(source, calibrator, kalman)
    except KeyboardInterrupt:
        print("\nStopped by user.")
    finally:
        if hasattr(source, 'disconnect'):
            source.disconnect()
        if hasattr(source, 'stop'):
            source.stop()


if __name__ == "__main__":
    main()
