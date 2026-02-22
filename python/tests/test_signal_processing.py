"""
Tests for signal processing modules.

Verifies low-pass filter, sliding window operation, Kalman filter,
feature extraction, and state-space controller math. Can be run without hardware.
"""

import sys
import os
import unittest

import numpy as np

sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

from eog_cursor.signal_processing import EOGLowPassFilter, SlidingWindow, GyroKalmanFilter, GyroKalmanFilter3Axis
from eog_cursor.feature_extraction import extract_features, FEATURE_NAMES
from eog_cursor import config


class TestEOGLowPassFilter(unittest.TestCase):
    """Test low-pass filter for EOG signals."""

    def setUp(self):
        self.filt = EOGLowPassFilter(cutoff=30, fs=200, order=4)

    def test_preserves_dc(self):
        """Low-pass filter should preserve DC baseline (unlike bandpass)."""
        dc_value = 2048.0
        outputs = []
        for _ in range(2000):
            outputs.append(self.filt.filter_sample(dc_value))
        # DC should be preserved (within 1% of input)
        self.assertAlmostEqual(outputs[-1], dc_value, delta=dc_value * 0.01)

    def test_passes_low_frequency(self):
        """Filter should pass signals well below cutoff."""
        # 5 Hz sine wave (well within 30 Hz cutoff)
        t = np.arange(0, 2, 1/200)
        signal = 1000 * np.sin(2 * np.pi * 5 * t)
        outputs = [self.filt.filter_sample(s) for s in signal]
        # After settling (skip first 200 samples), amplitude should be >80%
        self.assertGreater(np.max(np.abs(outputs[200:])), 800)

    def test_attenuates_high_frequency(self):
        """Filter should attenuate signals above cutoff."""
        # 80 Hz sine wave (above 30 Hz cutoff)
        filt = EOGLowPassFilter(cutoff=30, fs=200, order=4)
        t = np.arange(0, 2, 1/200)
        signal = 1000 * np.sin(2 * np.pi * 80 * t)
        outputs = [filt.filter_sample(s) for s in signal]
        # High frequency should be heavily attenuated
        self.assertLess(np.max(np.abs(outputs[200:])), 200)

    def test_reset(self):
        """Reset should clear filter state."""
        for _ in range(100):
            self.filt.filter_sample(3000.0)
        self.filt.reset()
        result = self.filt.filter_sample(2048.0)
        self.assertIsNotNone(result)


class TestSlidingWindow(unittest.TestCase):
    """Test sliding window buffer."""

    def test_starts_empty(self):
        """New window should not be full."""
        win = SlidingWindow(size=10)
        self.assertFalse(win.is_full())

    def test_becomes_full(self):
        """Window should report full after enough samples."""
        win = SlidingWindow(size=5)
        for i in range(5):
            win.push(float(i))
        self.assertTrue(win.is_full())

    def test_sliding_behavior(self):
        """Window should slide, keeping most recent values."""
        win = SlidingWindow(size=3)
        for i in range(5):
            win.push(float(i))
        data = win.get()
        np.testing.assert_array_equal(data, [2.0, 3.0, 4.0])

    def test_get_returns_copy(self):
        """get() should return a copy, not a reference."""
        win = SlidingWindow(size=3)
        for i in range(3):
            win.push(float(i))
        data = win.get()
        data[0] = 999  # Modify the copy
        self.assertNotEqual(win.get()[0], 999)

    def test_reset(self):
        """Reset should clear the window."""
        win = SlidingWindow(size=3)
        for i in range(3):
            win.push(float(i))
        win.reset()
        self.assertFalse(win.is_full())
        np.testing.assert_array_equal(win.get(), [0, 0, 0])


class TestGyroKalmanFilter(unittest.TestCase):
    """Test Kalman filter for gyroscope bias tracking."""

    def test_tracks_constant_bias(self):
        """Filter should converge on a constant bias when input is pure bias."""
        kf = GyroKalmanFilter(q_omega=1000.0, q_bias=0.001, r=500.0)
        # Feed constant reading of 300 (simulating pure bias, no real motion)
        for _ in range(2000):
            omega = kf.update(300.0)
        # After convergence, estimated omega should be near 0 (it's all bias)
        self.assertAlmostEqual(omega, 0.0, delta=30.0)
        # Estimated bias should be near 300
        self.assertAlmostEqual(kf.get_bias(), 300.0, delta=30.0)

    def test_passes_real_motion(self):
        """Filter should pass through real angular velocity on top of bias."""
        kf = GyroKalmanFilter(q_omega=1000.0, q_bias=0.001, r=500.0)
        # Let filter converge on bias=200 first
        for _ in range(2000):
            kf.update(200.0)
        # Now add a real motion signal: bias(200) + omega(500) = 700
        outputs = []
        for _ in range(50):
            outputs.append(kf.update(700.0))
        # The filter should detect nonzero angular velocity
        self.assertGreater(abs(outputs[-1]), 100.0)

    def test_initial_bias_speeds_convergence(self):
        """Setting initial bias should reduce convergence time."""
        kf = GyroKalmanFilter(q_omega=1000.0, q_bias=0.001, r=500.0)
        kf.set_initial_bias(300.0)
        # Even on first sample, bias estimate should be close to 300
        omega = kf.update(300.0)
        self.assertAlmostEqual(omega, 0.0, delta=50.0)

    def test_bias_drift_tracking(self):
        """Filter should track slow bias drift over time."""
        kf = GyroKalmanFilter(q_omega=1000.0, q_bias=0.001, r=500.0)
        # Bias starts at 100, then drifts to 400 linearly over 4000 samples
        for i in range(4000):
            bias = 100.0 + (300.0 * i / 4000.0)
            kf.update(bias)  # Pure bias, no real motion
        # Bias estimate should be somewhere tracking toward 400
        self.assertGreater(kf.get_bias(), 200.0)

    def test_3axis_wrapper(self):
        """3-axis wrapper should correct all axes independently."""
        kf3 = GyroKalmanFilter3Axis(q_omega=1000.0, q_bias=0.001, r=500.0)
        kf3.set_initial_bias(100.0, 200.0, 300.0)
        # Feed readings that are pure bias
        for _ in range(500):
            gx, gy, gz = kf3.update(100.0, 200.0, 300.0)
        # All outputs should be near zero (all bias, no real motion)
        self.assertAlmostEqual(gx, 0, delta=50)
        self.assertAlmostEqual(gy, 0, delta=50)
        self.assertAlmostEqual(gz, 0, delta=50)

    def test_3axis_get_bias(self):
        """3-axis wrapper should report bias for all axes."""
        kf3 = GyroKalmanFilter3Axis()
        kf3.set_initial_bias(100.0, 200.0, 300.0)
        bx, by, bz = kf3.get_bias()
        self.assertAlmostEqual(bx, 100.0)
        self.assertAlmostEqual(by, 200.0)
        self.assertAlmostEqual(bz, 300.0)


class TestFeatureExtraction(unittest.TestCase):
    """Test feature extraction from EOG windows."""

    def test_output_length(self):
        """Feature vector should have expected length."""
        window = np.random.randn(100) * 500 + 1500
        features = extract_features(window)
        self.assertEqual(len(features), len(FEATURE_NAMES))

    def test_blink_has_high_amplitude(self):
        """Blink signal should have higher peak amplitude than idle."""
        idle_window = np.ones(100) * 1500 + np.random.randn(100) * 10
        blink_window = np.ones(100) * 1500
        blink_window[40:60] += 2000  # Spike

        idle_feats = extract_features(idle_window)
        blink_feats = extract_features(blink_window)

        # Peak amplitude is feature index 0
        self.assertGreater(blink_feats[0], idle_feats[0])

    def test_constant_signal(self):
        """Constant signal should have zero std and near-zero features."""
        window = np.ones(100) * 1500
        features = extract_features(window)
        # Std should be 0
        self.assertAlmostEqual(features[5], 0.0, places=5)
        # Peak amplitude should be 0
        self.assertAlmostEqual(features[0], 0.0, places=5)

    def test_no_nan_values(self):
        """Features should not contain NaN values."""
        window = np.random.randn(100) * 500 + 1500
        features = extract_features(window)
        self.assertFalse(np.any(np.isnan(features)))


class TestCursorControllers(unittest.TestCase):
    """Test cursor controller state logic (without actual mouse movement)."""

    def test_state_space_matrices(self):
        """State-space A matrix should apply velocity retention."""
        # Test the math directly without importing cursor_control
        # (which requires pyautogui at runtime)
        retain = config.SS_VELOCITY_RETAIN
        dt = config.SS_DT

        A = np.array([
            [1, dt,     0, 0],
            [0, retain, 0, 0],
            [0, 0,      1, dt],
            [0, 0,      0, retain]
        ])

        # State: [px, vx, py, vy]
        state = np.array([0.0, 100.0, 0.0, 100.0])
        new_state = A @ state

        # Velocity should be scaled by retention factor
        self.assertAlmostEqual(new_state[1], 100.0 * retain)
        self.assertAlmostEqual(new_state[3], 100.0 * retain)
        # Position should increase by velocity * dt
        self.assertAlmostEqual(new_state[0], 100.0 * dt)

    def test_state_space_velocity_decays_to_zero(self):
        """Repeated velocity retention should bring velocity near zero."""
        retain = config.SS_VELOCITY_RETAIN
        A = np.array([
            [1, config.SS_DT, 0, 0],
            [0, retain,       0, 0],
            [0, 0,            1, config.SS_DT],
            [0, 0,            0, retain]
        ])

        state = np.array([0.0, 1000.0, 0.0, 1000.0])
        for _ in range(200):
            state = A @ state
            state[0] = 0  # Reset position (as controller does)
            state[2] = 0

        # After 200 iterations, velocity should be near zero
        self.assertLess(abs(state[1]), 1.0)
        self.assertLess(abs(state[3]), 1.0)


if __name__ == "__main__":
    unittest.main()
