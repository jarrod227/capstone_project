"""
Cursor control implementations.

Action mapping (per hardware capability table):
  - Left Click:     Double blink (two rapid blinks)
  - Right Click:    Long blink (eyes closed >=0.4s)
  - Scroll Up:      Eye Up + Head Up (eye-head sync, eog_v + gx)
  - Scroll Down:    Eye Down + Head Down (eye-head sync, eog_v + gx)
  - Back:           Eye Left + Head Left (eye-head sync, eog_h + gy)
  - Forward:        Eye Right + Head Right (eye-head sync, eog_h + gy)
  - Window Switch:  Head Roll Flick (lateral head tilt, gyro_z)
  - Double Click:   Double Head Nod (two quick nods, gyro_x)
  - Cursor Move:    IMU Gyro X/Y (angular velocity)

Two controller variants:
  1. ThresholdController - Direct event detection
  2. StateSpaceController - Physics-based cursor motion with inertia
"""

import time
import logging

import numpy as np

from . import config
from .event_detector import (
    BlinkDetector, GazeDetector, HorizontalGazeDetector, HeadRollDetector,
    DoubleNodDetector, EOGEvent
)

logger = logging.getLogger(__name__)

# Lazy-load pyautogui to allow testing without a display
_pyautogui = None


def _get_pyautogui():
    """Import and configure pyautogui on first use."""
    global _pyautogui
    if _pyautogui is None:
        import pyautogui
        pyautogui.FAILSAFE = False
        pyautogui.PAUSE = 0
        _pyautogui = pyautogui
    return _pyautogui


class _BaseController:
    """
    Shared event detection and action dispatch for all cursor controllers.

    Subclasses implement _compute_cursor_move() to define how IMU gyro
    data maps to pixel displacement.

    "Cursor frozen" means the user is looking left or right (horizontal
    EOG beyond threshold).  Head roll and double nod are only recognised
    in this state, which prevents accidental triggers during normal head
    movement and eliminates cursor drift during nods/rolls.
    """

    def __init__(self):
        self.deadzone = config.GYRO_DEADZONE

        # Event detectors
        self.blink_detector = BlinkDetector()
        self.gaze_detector = GazeDetector()
        self.horizontal_gaze_detector = HorizontalGazeDetector()
        self.roll_detector = HeadRollDetector()
        self.nod_detector = DoubleNodDetector()

        # Fusion cooldown state
        self.last_scroll_time = 0.0
        self.last_nav_time = 0.0
        self.last_roll_time = 0.0
        self.last_nod_time = 0.0

    def _compute_cursor_move(self, gx, gy, any_action, gui):
        """Compute and apply cursor movement. Subclasses must override."""
        raise NotImplementedError

    def update(self, eog_v: int, eog_h: int, gx: int, gy: int, gz: int,
               cursor_frozen_override: bool = False):
        """
        Process one sensor sample and execute actions.

        Coordinate mapping (head motion to screen):
          - Head turn right (gy > 0) → cursor moves right (+dx)
          - Head tilt down  (gx > 0) → cursor moves down  (+dy)

        Args:
            cursor_frozen_override: When True, force cursor-frozen state
                even if eog_h is at baseline.  Used by ML mode to signal
                that the classifier detected horizontal gaze.
        """
        now = time.time()
        gui = _get_pyautogui()

        # --- 1. Cursor movement from IMU gyro ---
        # Suppress ALL cursor movement when any action is active:
        #   - Eye gaze (vertical/horizontal) = scroll/nav intent
        #   - Post-roll/nod grace window absorbs residual coupled motion
        gaze_vertical = (eog_v > config.LOOK_UP_THRESHOLD or
                         eog_v < config.LOOK_DOWN_THRESHOLD)
        gaze_horizontal = (eog_h > config.LOOK_RIGHT_THRESHOLD or
                           eog_h < config.LOOK_LEFT_THRESHOLD)
        cursor_frozen = gaze_horizontal or cursor_frozen_override

        any_action = (gaze_vertical or cursor_frozen or
                      now - self.last_roll_time < config.HEAD_ROLL_SUPPRESS_DURATION or
                      now - self.last_nod_time < config.HEAD_ROLL_SUPPRESS_DURATION)

        self._compute_cursor_move(gx, gy, any_action, gui)

        # --- 2. Blink events (double blink → left, long blink → right) ---
        blink_event = self.blink_detector.update(eog_v, now)

        if blink_event == EOGEvent.DOUBLE_BLINK:
            gui.click(_pause=False)
            logger.info("Double blink → left click")
        elif blink_event == EOGEvent.LONG_BLINK:
            gui.click(button='right', _pause=False)
            logger.info("Long blink → right click")

        # --- 3. Scroll: eye gaze + head tilt fusion ---
        gaze_event = self.gaze_detector.update(eog_v, now)

        if gaze_event == EOGEvent.LOOK_UP and gx < -self.deadzone:
            if now - self.last_scroll_time > config.SCROLL_COOLDOWN:
                amount = max(1, int(abs(gx) / self.deadzone * config.SCROLL_AMOUNT))
                gui.scroll(amount, _pause=False)
                self.last_scroll_time = now
                logger.info(f"Scroll up {amount} lines (eye up + head up)")
        elif gaze_event == EOGEvent.LOOK_DOWN and gx > self.deadzone:
            if now - self.last_scroll_time > config.SCROLL_COOLDOWN:
                amount = max(1, int(abs(gx) / self.deadzone * config.SCROLL_AMOUNT))
                gui.scroll(-amount, _pause=False)
                self.last_scroll_time = now
                logger.info(f"Scroll down {amount} lines (eye down + head down)")

        # --- 4. Window switch: head roll flick (only while cursor frozen) ---
        roll_event = self.roll_detector.update(gz, now, cursor_frozen=cursor_frozen)
        if roll_event == "switch_window":
            self.last_roll_time = now
            gui.hotkey('alt', 'tab', _pause=False)
            logger.info("Head roll → window switch (Alt+Tab)")

        # --- 5. Double click: double head nod (only while cursor frozen) ---
        nod_event = self.nod_detector.update(gx, now, cursor_frozen=cursor_frozen)
        if nod_event == "double_click":
            self.last_nod_time = now
            gui.doubleClick(_pause=False)
            logger.info("Double nod → double click")

        # --- 6. Browser back/forward: horizontal gaze + head turn fusion ---
        horiz_event = self.horizontal_gaze_detector.update(eog_h, now)

        if horiz_event == EOGEvent.LOOK_LEFT and gy < -self.deadzone:
            if now - self.last_nav_time > config.HORIZONTAL_GAZE_COOLDOWN:
                gui.hotkey('alt', 'left', _pause=False)
                self.last_nav_time = now
                logger.info("Back (eye left + head left)")
        elif horiz_event == EOGEvent.LOOK_RIGHT and gy > self.deadzone:
            if now - self.last_nav_time > config.HORIZONTAL_GAZE_COOLDOWN:
                gui.hotkey('alt', 'right', _pause=False)
                self.last_nav_time = now
                logger.info("Forward (eye right + head right)")

    def reset(self):
        """Reset all detector state."""
        self.blink_detector.reset()
        self.gaze_detector.reset()
        self.horizontal_gaze_detector.reset()
        self.roll_detector.reset()
        self.nod_detector.reset()
        self.last_scroll_time = 0.0
        self.last_nav_time = 0.0
        self.last_roll_time = 0.0
        self.last_nod_time = 0.0


class ThresholdController(_BaseController):
    """
    Threshold-based cursor control with full action mapping.

    Direct proportional cursor movement: pixel displacement is
    proportional to gyro angular velocity (no inertia).
    """

    def __init__(self):
        super().__init__()
        self.sensitivity = config.CURSOR_SENSITIVITY

    def _compute_cursor_move(self, gx, gy, any_action, gui):
        dx = 0.0
        dy = 0.0

        if not any_action:
            if abs(gy) > self.deadzone:
                dx = gy * self.sensitivity
            if abs(gx) > self.deadzone:
                dy = gx * self.sensitivity

        if dx != 0 or dy != 0:
            gui.moveRel(dx, dy, _pause=False)


class StateSpaceController(_BaseController):
    """
    Physics-based cursor control using a state-space model.

    State vector: [pos_x, vel_x, pos_y, vel_y]
    Adds inertia so the cursor "glides" after head motion stops.

    State equation: x[k+1] = A * x[k] + B * u[k]
    - A contains velocity retention (velocity decays exponentially)
    - B maps gyro input to velocity changes

    Head roll and double nod only activate when the cursor is frozen
    (user is looking left or right).
    """

    def __init__(self):
        super().__init__()
        self.velocity_retain = config.SS_VELOCITY_RETAIN
        self.sensitivity = config.SS_SENSITIVITY
        self.dt = config.SS_DT

        # State: [pos_x, vel_x, pos_y, vel_y]
        self.state = np.zeros(4)

        # State transition matrix
        self.A = np.array([
            [1, self.dt,      0, 0],
            [0, self.velocity_retain, 0, 0],
            [0, 0,            1, self.dt],
            [0, 0,            0, self.velocity_retain]
        ])

        # Input matrix (gyro → velocity)
        self.B = np.array([
            [0,               0],
            [self.sensitivity, 0],
            [0,               0],
            [0,               self.sensitivity]
        ])

    def _compute_cursor_move(self, gx, gy, any_action, gui):
        if any_action:
            u = np.array([0, 0])
            # Zero velocity to freeze cursor immediately
            self.state[1] = 0
            self.state[3] = 0
        else:
            ux = gy if abs(gy) > self.deadzone else 0
            uy = gx if abs(gx) > self.deadzone else 0
            u = np.array([ux, uy])

        self.state = self.A @ self.state + self.B @ u

        dx = self.state[0]
        dy = self.state[2]

        if abs(dx) > 0.1 or abs(dy) > 0.1:
            gui.moveRel(dx, dy, _pause=False)

        # Reset position accumulator, keep velocity
        self.state[0] = 0
        self.state[2] = 0

    def reset(self):
        """Reset all internal state."""
        super().reset()
        self.state = np.zeros(4)
