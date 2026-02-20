"""
Tests for EOG event detection: double blink, long blink, gaze, head roll, double nod.

All tests use synthetic time to ensure deterministic behavior.
"""

import sys
import os
import unittest

sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

from eog_cursor.event_detector import (
    BlinkDetector, GazeDetector, HorizontalGazeDetector, HeadRollDetector,
    DoubleNodDetector, BlinkState, EOGEvent,
)
from eog_cursor import config


class TestBlinkDetector(unittest.TestCase):
    """Test double blink and long blink detection state machine."""

    def setUp(self):
        self.det = BlinkDetector()
        self.t = 0.0  # Synthetic time

    def _advance(self, seconds):
        """Advance synthetic time."""
        self.t += seconds

    def _feed_idle(self, duration_s, eog=2048):
        """Feed idle EOG samples for a duration."""
        samples = int(duration_s * config.SAMPLE_RATE)
        result = EOGEvent.NONE
        for _ in range(samples):
            r = self.det.update(eog, self.t)
            if r != EOGEvent.NONE:
                result = r
            self._advance(config.SAMPLE_PERIOD)
        return result

    def _feed_blink(self, duration_s, eog=3500):
        """Feed blink-level EOG samples for a duration."""
        samples = int(duration_s * config.SAMPLE_RATE)
        result = EOGEvent.NONE
        for _ in range(samples):
            r = self.det.update(eog, self.t)
            if r != EOGEvent.NONE:
                result = r
            self._advance(config.SAMPLE_PERIOD)
        return result

    def test_double_blink_detected(self):
        """Two quick blinks within window should produce DOUBLE_BLINK."""
        # First blink (0.1s)
        self._feed_blink(0.1)
        # Gap between blinks (0.2s)
        self._feed_idle(0.2)
        # Second blink (0.1s)
        result = self._feed_blink(0.1)
        # After second blink ends, need to drop below threshold
        if result == EOGEvent.NONE:
            result = self._feed_idle(0.05)

        self.assertEqual(result, EOGEvent.DOUBLE_BLINK)

    def test_long_blink_detected(self):
        """Sustained blink >= LONG_BLINK_MIN_DURATION should produce LONG_BLINK on release."""
        self._feed_blink(config.LONG_BLINK_MIN_DURATION + 0.05)
        result = self._feed_idle(0.05)
        self.assertEqual(result, EOGEvent.LONG_BLINK)

    def test_single_blink_ignored(self):
        """A single short blink followed by timeout should produce NONE."""
        self._feed_blink(0.1)
        # Wait for double blink window to expire
        result = self._feed_idle(config.DOUBLE_BLINK_WINDOW + 0.1)
        # Single blink alone should not trigger anything
        self.assertEqual(result, EOGEvent.NONE)

    def test_very_short_blink_rejected(self):
        """Blink shorter than MIN_DURATION should be ignored as noise."""
        # Very short pulse (below min duration)
        self._feed_blink(config.BLINK_MIN_DURATION * 0.5)
        self._feed_idle(0.3)
        # Should go back to IDLE without triggering
        self.assertEqual(self.det.state, BlinkState.IDLE)

    def test_cooldown_prevents_retrigger(self):
        """Events within cooldown period should be suppressed."""
        # First double blink
        self._feed_blink(0.1)
        self._feed_idle(0.2)
        self._feed_blink(0.1)
        result1 = self._feed_idle(0.05)
        self.assertEqual(result1, EOGEvent.DOUBLE_BLINK)

        # Immediately try another double blink (within cooldown)
        self._feed_blink(0.1)
        self._feed_idle(0.2)
        self._feed_blink(0.1)
        result2 = self._feed_idle(0.05)
        # Should be suppressed
        self.assertEqual(result2, EOGEvent.NONE)

    def test_reset_clears_state(self):
        """Reset should return to IDLE state."""
        self._feed_blink(0.1)
        self.det.reset()
        self.assertEqual(self.det.state, BlinkState.IDLE)
        self.assertEqual(self.det.blink_count, 0)

    def test_idle_produces_no_events(self):
        """Pure idle signal should never trigger events."""
        result = self._feed_idle(2.0)
        self.assertEqual(result, EOGEvent.NONE)

    def test_long_blink_fires_on_release(self):
        """Long blink should fire when eyes OPEN, not while still closed."""
        # Hold blink for 0.5s — no event while held
        result_held = self._feed_blink(0.5)
        self.assertEqual(result_held, EOGEvent.NONE)
        # Release — event fires now
        result_release = self._feed_idle(0.05)
        self.assertEqual(result_release, EOGEvent.LONG_BLINK)

    def test_sustained_close_no_retrigger(self):
        """Eyes held closed for 2s then released should fire LONG_BLINK once."""
        # Hold for 2s (within MAX_DURATION of 2.5s)
        self._feed_blink(2.0)
        # Release
        result = self._feed_idle(0.05)
        self.assertEqual(result, EOGEvent.LONG_BLINK)

    def test_long_blink_max_duration_rejected(self):
        """Blink exceeding MAX_DURATION should not emit LONG_BLINK."""
        # Hold eyes closed past MAX_DURATION
        self._feed_blink(config.LONG_BLINK_MAX_DURATION + 0.5)
        # Release — too long, rejected
        result = self._feed_idle(0.05)
        self.assertEqual(result, EOGEvent.NONE)


class TestGazeDetector(unittest.TestCase):
    """Test sustained gaze direction detection."""

    def setUp(self):
        self.det = GazeDetector()
        self.t = 0.0

    def _advance(self, seconds):
        self.t += seconds

    def test_look_up_detected(self):
        """Sustained EOG above LOOK_UP_THRESHOLD should return LOOK_UP."""
        # Feed look-up level for >100ms
        eog = config.LOOK_UP_THRESHOLD + 100
        result = EOGEvent.NONE
        for _ in range(50):  # 50 samples = 250ms
            r = self.det.update(eog, self.t)
            if r != EOGEvent.NONE:
                result = r
            self._advance(config.SAMPLE_PERIOD)
        self.assertEqual(result, EOGEvent.LOOK_UP)

    def test_look_down_detected(self):
        """Sustained EOG below LOOK_DOWN_THRESHOLD should return LOOK_DOWN."""
        eog = config.LOOK_DOWN_THRESHOLD - 100
        result = EOGEvent.NONE
        for _ in range(50):
            r = self.det.update(eog, self.t)
            if r != EOGEvent.NONE:
                result = r
            self._advance(config.SAMPLE_PERIOD)
        self.assertEqual(result, EOGEvent.LOOK_DOWN)

    def test_blink_level_not_gaze(self):
        """EOG above BLINK_THRESHOLD should NOT be detected as gaze."""
        eog = config.BLINK_THRESHOLD + 500
        result = EOGEvent.NONE
        for _ in range(50):
            r = self.det.update(eog, self.t)
            if r != EOGEvent.NONE:
                result = r
            self._advance(config.SAMPLE_PERIOD)
        self.assertEqual(result, EOGEvent.NONE)

    def test_baseline_no_gaze(self):
        """EOG near baseline should produce NONE."""
        eog = config.EOG_BASELINE
        result = EOGEvent.NONE
        for _ in range(50):
            r = self.det.update(eog, self.t)
            if r != EOGEvent.NONE:
                result = r
            self._advance(config.SAMPLE_PERIOD)
        self.assertEqual(result, EOGEvent.NONE)

    def test_transient_not_detected(self):
        """Very brief gaze shift should not be detected (< min duration)."""
        # Only 2 samples of look-up (10ms, below 100ms threshold)
        for _ in range(2):
            self.det.update(config.LOOK_UP_THRESHOLD + 100, self.t)
            self._advance(config.SAMPLE_PERIOD)
        # Then back to baseline
        result = self.det.update(config.EOG_BASELINE, self.t)
        self.assertEqual(result, EOGEvent.NONE)


class TestHeadRollDetector(unittest.TestCase):
    """Test head roll flick detection from gyro_z."""

    def setUp(self):
        self.det = HeadRollDetector()
        self.t = 0.0

    def _flick(self, gz, duration=0.1):
        """Simulate a flick: spike for duration, then return to neutral."""
        # Spike phase
        steps = int(duration / config.SAMPLE_PERIOD)
        for _ in range(max(steps, 1)):
            self.det.update(gz, self.t)
            self.t += config.SAMPLE_PERIOD
        # Return to neutral — trigger happens here
        return self.det.update(0, self.t)

    def test_roll_detected(self):
        """Quick gz spike + return to neutral should trigger."""
        result = self._flick(config.HEAD_ROLL_THRESHOLD + 500, duration=0.1)
        self.assertEqual(result, "switch_window")

    def test_small_gz_ignored(self):
        """Small gyro_z should not trigger."""
        result = self._flick(500, duration=0.1)
        self.assertIsNone(result)

    def test_held_too_long_ignored(self):
        """gz held above threshold too long should not trigger."""
        duration = config.HEAD_ROLL_MAX_DURATION + 0.1
        result = self._flick(config.HEAD_ROLL_THRESHOLD + 500, duration=duration)
        self.assertIsNone(result)

    def test_cooldown_works(self):
        """Rapid gz flicks should be suppressed by cooldown."""
        r1 = self._flick(4000, duration=0.1)
        self.assertEqual(r1, "switch_window")

        self.t += 0.1  # within cooldown
        r2 = self._flick(4000, duration=0.1)
        self.assertIsNone(r2)

        self.t += config.HEAD_ROLL_COOLDOWN  # after cooldown
        r3 = self._flick(4000, duration=0.1)
        self.assertEqual(r3, "switch_window")

    def test_negative_roll(self):
        """Negative gyro_z (roll left) should also trigger."""
        result = self._flick(-(config.HEAD_ROLL_THRESHOLD + 500), duration=0.1)
        self.assertEqual(result, "switch_window")


class TestDoubleNodDetector(unittest.TestCase):
    """Test double head nod detection from gyro_x."""

    def setUp(self):
        self.det = DoubleNodDetector()
        self.t = 0.0

    def _nod(self, gx, duration=0.1):
        """Simulate a single nod: spike for duration, then return to neutral."""
        steps = int(duration / config.SAMPLE_PERIOD)
        for _ in range(max(steps, 1)):
            self.det.update(gx, self.t)
            self.t += config.SAMPLE_PERIOD
        return self.det.update(0, self.t)

    def test_double_nod_detected(self):
        """Two quick nods should trigger double_click."""
        self._nod(config.DOUBLE_NOD_THRESHOLD + 500, duration=0.1)
        self.t += 0.1  # gap between nods
        result = self._nod(config.DOUBLE_NOD_THRESHOLD + 500, duration=0.1)
        self.assertEqual(result, "double_click")

    def test_single_nod_ignored(self):
        """One nod alone should not trigger."""
        result = self._nod(config.DOUBLE_NOD_THRESHOLD + 500, duration=0.1)
        self.assertIsNone(result)

    def test_small_gx_ignored(self):
        """Small gyro_x should not trigger."""
        self._nod(500, duration=0.1)
        self.t += 0.1
        result = self._nod(500, duration=0.1)
        self.assertIsNone(result)

    def test_held_too_long_ignored(self):
        """Nod held too long should not count."""
        duration = config.DOUBLE_NOD_MAX_DURATION + 0.1
        self._nod(config.DOUBLE_NOD_THRESHOLD + 500, duration=duration)
        self.t += 0.1
        result = self._nod(config.DOUBLE_NOD_THRESHOLD + 500, duration=0.1)
        self.assertIsNone(result)  # First nod was invalid, so only one valid nod

    def test_window_expired(self):
        """Two nods too far apart should not trigger."""
        self._nod(config.DOUBLE_NOD_THRESHOLD + 500, duration=0.1)
        self.t += config.DOUBLE_NOD_WINDOW + 0.1  # exceed window
        result = self._nod(config.DOUBLE_NOD_THRESHOLD + 500, duration=0.1)
        self.assertIsNone(result)

    def test_cooldown_works(self):
        """Double nod during cooldown should not trigger."""
        # First double nod
        self._nod(config.DOUBLE_NOD_THRESHOLD + 500, duration=0.1)
        self.t += 0.1
        r1 = self._nod(config.DOUBLE_NOD_THRESHOLD + 500, duration=0.1)
        self.assertEqual(r1, "double_click")

        # Second double nod within cooldown
        self.t += 0.1
        self._nod(config.DOUBLE_NOD_THRESHOLD + 500, duration=0.1)
        self.t += 0.1
        r2 = self._nod(config.DOUBLE_NOD_THRESHOLD + 500, duration=0.1)
        self.assertIsNone(r2)

        # After cooldown
        self.t += config.DOUBLE_NOD_COOLDOWN
        self._nod(config.DOUBLE_NOD_THRESHOLD + 500, duration=0.1)
        self.t += 0.1
        r3 = self._nod(config.DOUBLE_NOD_THRESHOLD + 500, duration=0.1)
        self.assertEqual(r3, "double_click")


class TestHorizontalGazeDetector(unittest.TestCase):
    """Test horizontal gaze detection from eog_h channel."""

    def setUp(self):
        self.det = HorizontalGazeDetector()
        self.t = 0.0

    def _advance(self, seconds):
        self.t += seconds

    def test_look_right_detected(self):
        """Sustained eog_h above LOOK_RIGHT_THRESHOLD should return LOOK_RIGHT."""
        eog_h = config.LOOK_RIGHT_THRESHOLD + 100
        result = EOGEvent.NONE
        for _ in range(60):  # 300ms, above min_gaze_duration of 0.15s
            r = self.det.update(eog_h, self.t)
            if r != EOGEvent.NONE:
                result = r
            self._advance(config.SAMPLE_PERIOD)
        self.assertEqual(result, EOGEvent.LOOK_RIGHT)

    def test_look_left_detected(self):
        """Sustained eog_h below LOOK_LEFT_THRESHOLD should return LOOK_LEFT."""
        eog_h = config.LOOK_LEFT_THRESHOLD - 100
        result = EOGEvent.NONE
        for _ in range(60):
            r = self.det.update(eog_h, self.t)
            if r != EOGEvent.NONE:
                result = r
            self._advance(config.SAMPLE_PERIOD)
        self.assertEqual(result, EOGEvent.LOOK_LEFT)

    def test_baseline_no_gaze(self):
        """eog_h near baseline should produce NONE."""
        eog_h = config.EOG_BASELINE
        result = EOGEvent.NONE
        for _ in range(60):
            r = self.det.update(eog_h, self.t)
            if r != EOGEvent.NONE:
                result = r
            self._advance(config.SAMPLE_PERIOD)
        self.assertEqual(result, EOGEvent.NONE)

    def test_cooldown_prevents_retrigger(self):
        """Rapid horizontal gaze should be suppressed by cooldown."""
        eog_h = config.LOOK_RIGHT_THRESHOLD + 100
        # First detection
        triggered = False
        for _ in range(60):
            r = self.det.update(eog_h, self.t)
            if r == EOGEvent.LOOK_RIGHT:
                triggered = True
                break
            self._advance(config.SAMPLE_PERIOD)
        self.assertTrue(triggered)

        # Reset gaze briefly
        self.det.update(config.EOG_BASELINE, self.t)
        self._advance(0.05)

        # Try again within cooldown - should NOT trigger
        result = EOGEvent.NONE
        for _ in range(60):
            r = self.det.update(eog_h, self.t)
            if r != EOGEvent.NONE:
                result = r
            self._advance(config.SAMPLE_PERIOD)
        self.assertEqual(result, EOGEvent.NONE)

    def test_transient_not_detected(self):
        """Very brief horizontal gaze should not be detected."""
        for _ in range(2):  # 10ms, below 150ms threshold
            self.det.update(config.LOOK_RIGHT_THRESHOLD + 100, self.t)
            self._advance(config.SAMPLE_PERIOD)
        result = self.det.update(config.EOG_BASELINE, self.t)
        self.assertEqual(result, EOGEvent.NONE)


if __name__ == "__main__":
    unittest.main()
