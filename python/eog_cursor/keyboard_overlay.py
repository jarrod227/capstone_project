"""
Keyboard overlay for hardware mode.

Allows injecting discrete EOG events via keyboard while real hardware
sensor data continues to flow unmodified.  The keyboard events are
processed through independent event detectors so they do not interfere
with hardware-sourced EOG signals.

Key mapping (EOG events only — IMU still comes from hardware):
  Space (tap x2):   Double blink → left click
  Space (hold):     Long blink → right click
  Space (tap x3):   Triple blink → double click
  U:                Look up  (for scroll fusion with hardware IMU)
  D:                Look down (for scroll fusion with hardware IMU)
  L:                Look left  (freezes cursor, enables nod/roll from hardware IMU)
  R:                Look right (freezes cursor, enables nod/roll from hardware IMU)
"""

import logging

from . import config
from .event_detector import BlinkDetector, GazeDetector, HorizontalGazeDetector, EOGEvent

logger = logging.getLogger(__name__)


class KeyboardOverlay:
    """Injects discrete EOG events from keyboard alongside hardware input.

    Each ``poll(now)`` call feeds synthesized EOG values (derived from the
    current key-press state) through its own set of event detectors.  The
    caller merges these results with hardware-sourced events using simple
    OR logic (keyboard fills in when hardware produces ``NONE``).
    """

    def __init__(self):
        self._listener = None

        # Independent detectors — separate from the hardware-fed ones
        self._blink_detector = BlinkDetector()
        self._gaze_detector = GazeDetector()
        self._horiz_gaze_detector = HorizontalGazeDetector()

        # Key state (updated by pynput listener thread)
        self._space_pressed = False
        self._look_up = False
        self._look_down = False
        self._look_left = False
        self._look_right = False

    # ------------------------------------------------------------------
    # Lifecycle
    # ------------------------------------------------------------------

    def start(self):
        """Start background keyboard listener."""
        from pynput.keyboard import Listener

        self._listener = Listener(
            on_press=self._on_press,
            on_release=self._on_release,
        )
        self._listener.daemon = True
        self._listener.start()
        logger.info("Keyboard overlay active: Space=blink, U/D/L/R=gaze")

    def stop(self):
        """Stop keyboard listener."""
        if self._listener:
            self._listener.stop()
        logger.info("Keyboard overlay stopped.")

    # ------------------------------------------------------------------
    # pynput callbacks
    # ------------------------------------------------------------------

    def _on_press(self, key):
        try:
            from pynput.keyboard import Key
            if key == Key.space:
                self._space_pressed = True
        except AttributeError:
            pass

        if hasattr(key, 'char') and key.char:
            c = key.char
            if c == 'u':
                self._look_up = True
            elif c == 'd':
                self._look_down = True
            elif c == 'l':
                self._look_left = True
            elif c == 'r':
                self._look_right = True

    def _on_release(self, key):
        try:
            from pynput.keyboard import Key
            if key == Key.space:
                self._space_pressed = False
        except AttributeError:
            pass

        if hasattr(key, 'char') and key.char:
            c = key.char
            if c == 'u':
                self._look_up = False
            elif c == 'd':
                self._look_down = False
            elif c == 'l':
                self._look_left = False
            elif c == 'r':
                self._look_right = False

    # ------------------------------------------------------------------
    # Polling
    # ------------------------------------------------------------------

    def poll(self, now: float):
        """Feed keyboard state through independent detectors.

        Returns:
            (blink_event, gaze_event, horiz_event, cursor_frozen)
            where *cursor_frozen* is True when L or R is held.
        """
        # --- Synthesize eog_v ---
        if self._space_pressed:
            synth_eog_v = config.BLINK_THRESHOLD + 500  # ~3500
        elif self._look_up:
            synth_eog_v = config.LOOK_UP_THRESHOLD + 100  # ~2900
        elif self._look_down:
            synth_eog_v = config.LOOK_DOWN_THRESHOLD - 200  # ~1000
        else:
            synth_eog_v = config.EOG_BASELINE

        blink_event = self._blink_detector.update(synth_eog_v, now)
        gaze_event = self._gaze_detector.update(synth_eog_v, now)

        # --- Synthesize eog_h ---
        if self._look_right:
            synth_eog_h = config.LOOK_RIGHT_THRESHOLD + 100  # ~2900
        elif self._look_left:
            synth_eog_h = config.LOOK_LEFT_THRESHOLD - 200  # ~1000
        else:
            synth_eog_h = config.EOG_BASELINE

        horiz_event = self._horiz_gaze_detector.update(synth_eog_h, now)
        cursor_frozen = self._look_left or self._look_right

        return blink_event, gaze_event, horiz_event, cursor_frozen
