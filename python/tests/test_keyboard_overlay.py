"""
Tests for KeyboardOverlay — verifies that synthesized EOG values
produce the correct events through independent detectors.

All tests manipulate KeyboardOverlay's internal key state directly
(no pynput dependency) and use synthetic time for determinism.
"""

import sys
import os
import unittest

sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

from eog_cursor.keyboard_overlay import KeyboardOverlay
from eog_cursor.event_detector import EOGEvent
from eog_cursor import config


class TestKeyboardOverlay(unittest.TestCase):
    """Test KeyboardOverlay.poll() with direct key-state manipulation."""

    def setUp(self):
        self.kb = KeyboardOverlay()
        self.t = 0.0

    def _advance(self, seconds):
        self.t += seconds

    def _poll_n(self, n, dt=0.005):
        """Poll n times, advancing dt each time. Return last result."""
        result = None
        for _ in range(n):
            self._advance(dt)
            result = self.kb.poll(self.t)
        return result

    # ------------------------------------------------------------------
    # Double blink via space bar
    # ------------------------------------------------------------------

    def test_double_blink_from_space(self):
        """Two quick space taps → DOUBLE_BLINK (left click)."""
        # First blink: press 100ms, release
        self.kb._space_pressed = True
        self._poll_n(20)  # 100ms at 5ms steps
        self.kb._space_pressed = False
        self._poll_n(10)  # 50ms gap

        # Second blink: press 100ms, release
        self.kb._space_pressed = True
        self._poll_n(20)
        self.kb._space_pressed = False

        # Wait for WAIT_THIRD timeout (>600ms)
        events = []
        for _ in range(200):
            self._advance(0.005)
            b, g, h, frozen = self.kb.poll(self.t)
            if b != EOGEvent.NONE:
                events.append(b)

        self.assertIn(EOGEvent.DOUBLE_BLINK, events)

    # ------------------------------------------------------------------
    # Triple blink via space bar
    # ------------------------------------------------------------------

    def test_triple_blink_from_space(self):
        """Three quick space taps → TRIPLE_BLINK (double click)."""
        events = []

        for _ in range(3):
            self.kb._space_pressed = True
            for _ in range(20):  # 100ms blink
                self._advance(0.005)
                b, g, h, frozen = self.kb.poll(self.t)
                if b != EOGEvent.NONE:
                    events.append(b)
            self.kb._space_pressed = False
            for _ in range(10):  # 50ms gap
                self._advance(0.005)
                b, g, h, frozen = self.kb.poll(self.t)
                if b != EOGEvent.NONE:
                    events.append(b)

        # Also wait for any timeout
        for _ in range(50):
            self._advance(0.005)
            b, g, h, frozen = self.kb.poll(self.t)
            if b != EOGEvent.NONE:
                events.append(b)

        self.assertIn(EOGEvent.TRIPLE_BLINK, events)

    # ------------------------------------------------------------------
    # Long blink via space bar
    # ------------------------------------------------------------------

    def test_long_blink_from_space(self):
        """Hold space > 400ms → LONG_BLINK (right click)."""
        self.kb._space_pressed = True
        self._poll_n(120)  # 600ms hold
        self.kb._space_pressed = False

        events = []
        for _ in range(20):
            self._advance(0.005)
            b, g, h, frozen = self.kb.poll(self.t)
            if b != EOGEvent.NONE:
                events.append(b)

        self.assertIn(EOGEvent.LONG_BLINK, events)

    # ------------------------------------------------------------------
    # Gaze events from U/D keys
    # ------------------------------------------------------------------

    def test_look_up_from_u_key(self):
        """Holding U → LOOK_UP after min gaze duration."""
        self.kb._look_up = True
        events = []
        for _ in range(60):  # 300ms
            self._advance(0.005)
            b, g, h, frozen = self.kb.poll(self.t)
            if g != EOGEvent.NONE:
                events.append(g)

        self.assertIn(EOGEvent.LOOK_UP, events)

    def test_look_down_from_d_key(self):
        """Holding D → LOOK_DOWN after min gaze duration."""
        self.kb._look_down = True
        events = []
        for _ in range(60):
            self._advance(0.005)
            b, g, h, frozen = self.kb.poll(self.t)
            if g != EOGEvent.NONE:
                events.append(g)

        self.assertIn(EOGEvent.LOOK_DOWN, events)

    # ------------------------------------------------------------------
    # Horizontal gaze / cursor freeze from L/R keys
    # ------------------------------------------------------------------

    def test_cursor_frozen_from_l_key(self):
        """Pressing L → cursor_frozen=True immediately."""
        self.kb._look_left = True
        _, _, _, frozen = self.kb.poll(self.t)
        self.assertTrue(frozen)

    def test_cursor_frozen_from_r_key(self):
        """Pressing R → cursor_frozen=True immediately."""
        self.kb._look_right = True
        _, _, _, frozen = self.kb.poll(self.t)
        self.assertTrue(frozen)

    def test_cursor_not_frozen_idle(self):
        """No keys → cursor_frozen=False."""
        _, _, _, frozen = self.kb.poll(self.t)
        self.assertFalse(frozen)

    def test_look_left_horiz_event(self):
        """Holding L long enough → LOOK_LEFT horizontal event."""
        self.kb._look_left = True
        events = []
        for _ in range(80):  # 400ms
            self._advance(0.005)
            b, g, h, frozen = self.kb.poll(self.t)
            if h != EOGEvent.NONE:
                events.append(h)

        self.assertIn(EOGEvent.LOOK_LEFT, events)

    def test_look_right_horiz_event(self):
        """Holding R long enough → LOOK_RIGHT horizontal event."""
        self.kb._look_right = True
        events = []
        for _ in range(80):
            self._advance(0.005)
            b, g, h, frozen = self.kb.poll(self.t)
            if h != EOGEvent.NONE:
                events.append(h)

        self.assertIn(EOGEvent.LOOK_RIGHT, events)

    # ------------------------------------------------------------------
    # No events when idle
    # ------------------------------------------------------------------

    def test_idle_no_events(self):
        """No keys pressed → all events are NONE."""
        for _ in range(100):
            self._advance(0.005)
            b, g, h, frozen = self.kb.poll(self.t)
            self.assertEqual(b, EOGEvent.NONE)
            self.assertEqual(g, EOGEvent.NONE)
            self.assertEqual(h, EOGEvent.NONE)
            self.assertFalse(frozen)

    # ------------------------------------------------------------------
    # Independence: blink and gaze don't interfere
    # ------------------------------------------------------------------

    def test_space_does_not_produce_gaze(self):
        """Space bar produces blink-level signal, not gaze events."""
        self.kb._space_pressed = True
        gaze_events = []
        for _ in range(40):
            self._advance(0.005)
            b, g, h, frozen = self.kb.poll(self.t)
            if g != EOGEvent.NONE:
                gaze_events.append(g)

        # GazeDetector should reject blink-level signals
        self.assertEqual(len(gaze_events), 0)


if __name__ == "__main__":
    unittest.main()
