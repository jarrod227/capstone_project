"""
Configuration parameters for the EOG cursor control system.

All tunable parameters are centralized here for easy adjustment.

EOG Signal Model (12-bit ADC, dual channel, two AD8232 modules):
  Vertical channel (eog_v):  electrodes above/below one eye
    Baseline ~2048 | Blink/Up > 2048 | Down < 2048
  Horizontal channel (eog_h): electrodes at left/right outer canthi
    Baseline ~2048 | Right > 2048 | Left < 2048

  Electrode placement:
         [REF]  Forehead (reference/ground)
    [L]  [V+]  [R]   L/R = horizontal pair, V+/V- = vertical pair
         [V-]
"""

# --- Serial Communication ---
SERIAL_PORT = "/dev/ttyACM0"  # Linux default (Nucleo); Windows: COM4
SERIAL_BAUDRATE = 115200
SERIAL_TIMEOUT = 1.0          # seconds

# --- Sampling ---
SAMPLE_RATE = 200             # Hz (must match STM32 firmware)
SAMPLE_PERIOD = 1.0 / SAMPLE_RATE  # 5ms

# --- EOG Signal Processing ---
EOG_BASELINE = 2048           # 12-bit ADC midpoint (1.65V)
EOG_LOWPASS_CUTOFF = 30.0     # Hz - removes EMG/power line noise, preserves DC baseline
EOG_LOWPASS_ORDER = 4         # Butterworth filter order
EOG_LOWPASS_ENABLED = True    # Always on — no effect on clean data, removes noise on hardware

# --- IMU Deadzone ---
GYRO_DEADZONE = 300           # Raw gyro threshold (below = noise, ignore after bias removal)

# --- IMU Calibration (startup bias removal) ---
GYRO_CALIBRATION_SAMPLES = 400  # Number of samples for bias estimation (2s at 200Hz)
GYRO_CALIBRATION_DISCARD = 50   # Discard first N samples (sensor settling)

# --- IMU Kalman Filter (runtime bias tracking) ---
# State: [angular_velocity, bias] per axis
# Gyro reading = angular_velocity + bias + noise
KALMAN_Q_OMEGA = 1000.0   # Process noise for angular velocity (changes fast, trust measurement)
KALMAN_Q_BIAS = 0.001     # Process noise for bias (drifts very slowly)
KALMAN_R = 500.0           # Measurement noise (gyro sensor noise variance)

# --- Cursor Movement (IMU) ---
CURSOR_SENSITIVITY = 0.01     # Gyro-to-pixel scaling factor

# --- Cursor Control (State-Space Model) ---
SS_VELOCITY_RETAIN = 0.95     # Velocity retention per step (0.8=quick stop, 0.99=long glide)
SS_SENSITIVITY = 0.05         # Input gain (gives ~half threshold speed + glide)
SS_DT = SAMPLE_PERIOD         # Time step for state equations

# --- Blink Detection (vertical EOG) ---
# A "blink" is eog_v > BLINK_THRESHOLD (large positive spike above baseline)
BLINK_THRESHOLD = 3000        # ADC value above which = blink detected
BLINK_MIN_DURATION = 0.05     # seconds - minimum blink duration (debounce)
BLINK_MAX_DURATION = 0.25     # seconds - max duration for a normal blink

# --- Double Blink → Left Click ---
DOUBLE_BLINK_WINDOW = 0.6     # seconds - two blinks within this window = double blink
DOUBLE_BLINK_COOLDOWN = 0.8   # seconds - prevent re-trigger after double blink

# --- Long Blink → Right Click ---
LONG_BLINK_MIN_DURATION = 0.4 # seconds - blink held longer than this = long blink
LONG_BLINK_MAX_DURATION = 2.5 # seconds - cap to avoid accidental triggers
LONG_BLINK_COOLDOWN = 1.0     # seconds - prevent re-trigger after long blink

# --- Vertical Gaze (eog_v) → Scroll fusion ---
LOOK_UP_THRESHOLD = 2800      # eog_v > this = looking up (sustained, not blink)
LOOK_DOWN_THRESHOLD = 1200    # eog_v < this = looking down

# --- Horizontal Gaze (eog_h) + Head Turn (gy) → Back/Forward (eye-head fusion) ---
LOOK_RIGHT_THRESHOLD = 2800   # eog_h > this = looking right
LOOK_LEFT_THRESHOLD = 1200    # eog_h < this = looking left
HORIZONTAL_GAZE_COOLDOWN = 1.0  # seconds between back/fwd triggers
# Back:    eye left (eog_h < LOOK_LEFT_THRESHOLD) + head left (gy < -GYRO_DEADZONE)
# Forward: eye right (eog_h > LOOK_RIGHT_THRESHOLD) + head right (gy > GYRO_DEADZONE)

# --- Scroll: Eye + Head Fusion ---
SCROLL_COOLDOWN = 0.08        # seconds - min interval between scroll events
SCROLL_AMOUNT = 30            # base scroll clicks per event (scales with head speed)

# --- Window Switch: Head Roll Flick (gyro_z) ---
HEAD_ROLL_THRESHOLD = 3000    # Raw gyro_z threshold for roll flick detection
HEAD_ROLL_COOLDOWN = 1.0      # seconds - prevent re-trigger
HEAD_ROLL_SUPPRESS_DURATION = 0.3   # seconds - suppress cursor after head roll
HEAD_ROLL_MAX_DURATION = 0.3  # seconds - gz must return below threshold within this time

# --- Double Click: Double Head Nod (gyro_x) ---
DOUBLE_NOD_THRESHOLD = 3000   # Raw gyro_x threshold for nod detection
DOUBLE_NOD_MAX_DURATION = 0.3 # seconds - single nod must be shorter than this
DOUBLE_NOD_WINDOW = 0.8       # seconds - two nods within this window = double click
DOUBLE_NOD_COOLDOWN = 1.0     # seconds - prevent re-trigger

# --- ML Model ---
import os as _os
_PACKAGE_DIR = _os.path.dirname(_os.path.dirname(_os.path.abspath(__file__)))
ML_MODEL_PATH = _os.path.join(_PACKAGE_DIR, "models", "eog_model.pkl")
ML_SCALER_PATH = _os.path.join(_PACKAGE_DIR, "models", "eog_scaler.pkl")
ML_WINDOW_SIZE = 100          # samples per classification window (0.5s at 200Hz)
ML_WINDOW_STEP = 20           # step between windows (0.1s at 200Hz)
ML_CLASSES = ["idle", "blink", "double_blink", "long_blink",
              "look_up", "look_down", "look_left", "look_right"]

# --- Data Collection ---
COLLECT_OUTPUT_DIR = "data/raw"

# --- Simulator ---
SIM_NOISE_STD = 50            # ADC noise standard deviation
SIM_GYRO_NOISE_STD = 100      # Gyro noise standard deviation
