"""
Signal processing utilities for EOG data.

Provides low-pass filtering for hardware noise reduction
and sliding window buffer for ML feature extraction.
"""

import numpy as np
from scipy.signal import butter, sosfilt, sosfilt_zi

from . import config


class EOGLowPassFilter:
    """
    Real-time low-pass filter for EOG signals.

    Removes high-frequency noise (EMG, power line) while preserving
    the DC baseline needed for threshold comparisons and mean-based
    ML features. Intended for hardware data where AD8232 analog
    filtering alone may be insufficient.

    Uses Butterworth IIR filter with second-order sections (SOS)
    for numerical stability.
    """

    def __init__(self, cutoff=None, fs=None, order=None):
        cutoff = cutoff or config.EOG_LOWPASS_CUTOFF
        fs = fs or config.SAMPLE_RATE
        order = order or config.EOG_LOWPASS_ORDER

        nyquist = fs / 2.0
        normalized_cutoff = cutoff / nyquist
        self.sos = butter(order, normalized_cutoff, btype="low", output="sos")
        self._zi_template = sosfilt_zi(self.sos)
        self.zi = None  # Scaled on first sample

    def filter_sample(self, sample: float) -> float:
        """Filter a single sample, maintaining internal state."""
        if self.zi is None:
            # Scale initial conditions by first sample to avoid startup transient.
            # sosfilt_zi returns steady-state zi for input=1.0; multiplying by the
            # actual first sample makes the filter start at the correct DC level.
            self.zi = self._zi_template * sample
        filtered, self.zi = sosfilt(self.sos, [sample], zi=self.zi)
        return filtered[0]

    def reset(self):
        """Reset filter state (re-initializes on next sample)."""
        self.zi = None


class GyroCalibrator:
    """
    Startup gyroscope bias calibration.

    Collects samples while the device is stationary, computes the
    average bias per axis, and subtracts it from subsequent readings.
    This eliminates the static offset that causes cursor drift at rest.
    """

    def __init__(self, num_samples=None, discard=None):
        self.num_samples = num_samples or config.GYRO_CALIBRATION_SAMPLES
        self.discard = discard or config.GYRO_CALIBRATION_DISCARD
        self.bias_x = 0.0
        self.bias_y = 0.0
        self.bias_z = 0.0
        self.calibrated = False

    def calibrate(self, source):
        """
        Run calibration by collecting samples from the data source.

        The device must be stationary during this period.
        Returns (bias_x, bias_y, bias_z).
        """
        samples_x = []
        samples_y = []
        samples_z = []
        total = self.discard + self.num_samples
        count = 0

        for packet in source.stream():
            count += 1
            if count <= self.discard:
                continue
            samples_x.append(packet.gyro_x)
            samples_y.append(packet.gyro_y)
            samples_z.append(packet.gyro_z)
            if len(samples_x) >= self.num_samples:
                break

        self.bias_x = float(np.mean(samples_x))
        self.bias_y = float(np.mean(samples_y))
        self.bias_z = float(np.mean(samples_z))
        self.calibrated = True
        return self.bias_x, self.bias_y, self.bias_z

    def correct(self, gx: int, gy: int, gz: int):
        """Subtract bias from raw gyro readings."""
        return (
            int(round(gx - self.bias_x)),
            int(round(gy - self.bias_y)),
            int(round(gz - self.bias_z)),
        )


class GyroKalmanFilter:
    """
    Kalman filter for single-axis gyroscope bias tracking.

    Separates true angular velocity from slowly-drifting bias using a
    2-state model:
        state  x = [omega, bias]
        measurement  z = omega + bias + noise

    The key insight: bias changes slowly (small Q_bias) while angular
    velocity changes quickly (large Q_omega). When the gyro reads a
    sustained offset, the filter gradually attributes it to bias drift
    rather than real motion — no explicit stillness detection needed.
    """

    def __init__(self, q_omega=None, q_bias=None, r=None):
        q_omega = q_omega if q_omega is not None else config.KALMAN_Q_OMEGA
        q_bias = q_bias if q_bias is not None else config.KALMAN_Q_BIAS
        r = r if r is not None else config.KALMAN_R

        # State vector [omega, bias]
        self.x = np.array([0.0, 0.0])

        # State covariance
        self.P = np.array([
            [1000.0, 0.0],
            [0.0, 1000.0],
        ])

        # State transition: omega has no memory, bias persists
        self.F = np.array([
            [0.0, 0.0],
            [0.0, 1.0],
        ])

        # Process noise
        self.Q = np.array([
            [q_omega, 0.0],
            [0.0, q_bias],
        ])

        # Measurement matrix: z = omega + bias
        self.H = np.array([[1.0, 1.0]])

        # Measurement noise
        self.R = np.array([[r]])

    def update(self, z: float) -> float:
        """
        Process one raw gyro sample, return estimated true angular velocity.

        Args:
            z: Raw gyro reading (includes true angular velocity + bias + noise)

        Returns:
            Estimated true angular velocity (bias removed)
        """
        # --- Predict ---
        x_pred = self.F @ self.x
        P_pred = self.F @ self.P @ self.F.T + self.Q

        # --- Update ---
        # Innovation
        y = z - (self.H @ x_pred)[0]

        # Innovation covariance
        S = (self.H @ P_pred @ self.H.T + self.R)[0, 0]

        # Kalman gain
        K = (P_pred @ self.H.T) / S  # (2,1)

        # State update
        self.x = x_pred + K.flatten() * y

        # Covariance update
        I = np.eye(2)
        self.P = (I - K @ self.H) @ P_pred

        # Return estimated angular velocity (state[0])
        return self.x[0]

    def get_bias(self) -> float:
        """Return current bias estimate."""
        return self.x[1]

    def set_initial_bias(self, bias: float):
        """
        Initialize bias from startup calibration.

        Gives the filter a head start so it doesn't need to converge
        from zero during the first few seconds.
        """
        self.x[1] = bias
        # Reduce bias uncertainty since we have a calibration estimate
        self.P[1, 1] = 100.0


class GyroKalmanFilter3Axis:
    """
    Three independent Kalman filters for gyro X, Y, Z axes.

    Convenience wrapper that applies GyroKalmanFilter to each axis
    and returns corrected (gx, gy, gz) tuple matching the interface
    of GyroCalibrator.correct().
    """

    def __init__(self, q_omega=None, q_bias=None, r=None):
        self.kf_x = GyroKalmanFilter(q_omega, q_bias, r)
        self.kf_y = GyroKalmanFilter(q_omega, q_bias, r)
        self.kf_z = GyroKalmanFilter(q_omega, q_bias, r)

    def set_initial_bias(self, bx: float, by: float, bz: float):
        """Initialize all axes from startup calibration."""
        self.kf_x.set_initial_bias(bx)
        self.kf_y.set_initial_bias(by)
        self.kf_z.set_initial_bias(bz)

    def update(self, gx: float, gy: float, gz: float):
        """
        Process raw gyro readings, return bias-corrected values.

        Returns:
            (corrected_gx, corrected_gy, corrected_gz) as integers
        """
        ox = self.kf_x.update(gx)
        oy = self.kf_y.update(gy)
        oz = self.kf_z.update(gz)
        return int(round(ox)), int(round(oy)), int(round(oz))

    def get_bias(self):
        """Return current bias estimates for all axes."""
        return self.kf_x.get_bias(), self.kf_y.get_bias(), self.kf_z.get_bias()


class SlidingWindow:
    """
    Sliding window buffer for feature extraction.

    Maintains a fixed-size window of recent samples,
    used for EOG pattern classification.
    """

    def __init__(self, size=None):
        self.size = size or config.ML_WINDOW_SIZE
        self.buffer = np.zeros(self.size)
        self.count = 0

    def push(self, value: float):
        """Add a sample, shifting the window."""
        self.buffer = np.roll(self.buffer, -1)
        self.buffer[-1] = value
        self.count += 1

    def is_full(self) -> bool:
        """Check if window has been fully populated at least once."""
        return self.count >= self.size

    def get(self) -> np.ndarray:
        """Return current window contents."""
        return self.buffer.copy()

    def reset(self):
        """Clear the window."""
        self.buffer = np.zeros(self.size)
        self.count = 0
