"""
EOG event detector for blink patterns and gaze direction.

Two EOG channels: vertical (eog_v) for blinks/up/down, horizontal (eog_h) for left/right

Implements a state machine that distinguishes:
  - Double blink  → left click
  - Triple blink  → double click
  - Long blink    → right click
  - Look up/down  → scroll (requires head tilt fusion in controller)
  - Look left/right → cursor freeze (enables nod/roll while frozen)
  - Look left + head turn left  → browser back (requires fusion in controller)
  - Look right + head turn right → browser forward (requires fusion in controller)
  - Head roll     → window switch  (requires cursor_frozen / looking left or right)
  - Double nod    → center cursor  (requires cursor_frozen / looking left or right)

Head roll and double nod only activate when the cursor is frozen (user is
looking left or right).  This prevents accidental triggers during normal
head movement and eliminates the cursor-drift problem during nods/rolls.

EOG Signal Model (12-bit ADC, vertical channel):
  Baseline ~2048 | Blink/Up > 2048 | Down < 2048
"""

import time
import logging
from enum import Enum, auto

from . import config

logger = logging.getLogger(__name__)


class BlinkState(Enum):
    """Blink detection state machine states."""
    IDLE = auto()           # Waiting for EOG to rise above threshold
    IN_BLINK = auto()       # EOG is above threshold (blink in progress)
    WAIT_SECOND = auto()    # One blink ended, waiting for potential second blink
    WAIT_THIRD = auto()     # Two blinks ended, waiting for potential third blink


class EOGEvent(Enum):
    """Discrete events detected from EOG + IMU."""
    NONE = auto()
    DOUBLE_BLINK = auto()   # → left click
    TRIPLE_BLINK = auto()   # → double click
    LONG_BLINK = auto()     # → right click
    LOOK_UP = auto()        # → scroll fusion component
    LOOK_DOWN = auto()      # → scroll fusion component
    LOOK_LEFT = auto()      # → browser back
    LOOK_RIGHT = auto()     # → browser forward


class BlinkDetector:
    """
    Detects double-blink, triple-blink, and long-blink patterns from raw EOG values.

    State machine:
      IDLE → (EOG > threshold) → IN_BLINK
      IN_BLINK → (EOG < threshold, 50-250ms) → WAIT_SECOND
      IN_BLINK → (EOG < threshold, 0.4-2.5s) → emit LONG_BLINK → IDLE
      IN_BLINK → (EOG < threshold, >2.5s) → too long, discard → IDLE
      WAIT_SECOND → (EOG > threshold within window) → IN_BLINK (2nd)
      WAIT_SECOND → (timeout) → ignore single blink → IDLE
      IN_BLINK (2nd) → (EOG < threshold, valid) → WAIT_THIRD
      WAIT_THIRD → (EOG > threshold within window) → IN_BLINK (3rd)
      WAIT_THIRD → (timeout) → emit DOUBLE_BLINK → IDLE
      IN_BLINK (3rd) → (EOG < threshold) → emit TRIPLE_BLINK → IDLE
    """

    def __init__(self):
        self.state = BlinkState.IDLE
        self.blink_start_time = 0.0
        self.blink_end_time = 0.0
        self.blink_count = 0
        self.last_event_time = -100.0  # Allow first event immediately

    def update(self, eog: int, now: float = None) -> EOGEvent:
        """
        Feed one EOG sample and return detected event.

        Args:
            eog: Raw 12-bit ADC value
            now: Current time (default: time.time())

        Returns:
            EOGEvent.DOUBLE_BLINK, TRIPLE_BLINK, LONG_BLINK, or NONE
        """
        if now is None:
            now = time.time()

        is_high = eog > config.BLINK_THRESHOLD

        if self.state == BlinkState.IDLE:
            if is_high:
                self.state = BlinkState.IN_BLINK
                self.blink_start_time = now
                self.blink_count = 1

        elif self.state == BlinkState.IN_BLINK:
            duration = now - self.blink_start_time

            if is_high:
                # Still in blink — just wait for release
                pass
            else:
                # EOG dropped below threshold — blink ended, check duration
                if duration < config.BLINK_MIN_DURATION:
                    # Too short, probably noise
                    self.state = BlinkState.IDLE
                elif self.blink_count >= 3:
                    # Third blink ended → triple blink!
                    if duration <= config.BLINK_MAX_DURATION:
                        if now - self.last_event_time > config.TRIPLE_BLINK_COOLDOWN:
                            self.state = BlinkState.IDLE
                            self.last_event_time = now
                            logger.debug("Triple blink detected")
                            return EOGEvent.TRIPLE_BLINK
                    self.state = BlinkState.IDLE
                elif self.blink_count >= 2:
                    # Second blink ended → wait for potential third
                    if duration <= config.BLINK_MAX_DURATION:
                        self.blink_end_time = now
                        self.state = BlinkState.WAIT_THIRD
                    else:
                        self.state = BlinkState.IDLE
                elif duration >= config.LONG_BLINK_MIN_DURATION:
                    # Long blink (right click) — fires on release
                    if duration <= config.LONG_BLINK_MAX_DURATION:
                        if now - self.last_event_time > config.LONG_BLINK_COOLDOWN:
                            self.state = BlinkState.IDLE
                            self.last_event_time = now
                            logger.debug(f"Long blink detected ({duration:.2f}s)")
                            return EOGEvent.LONG_BLINK
                    # duration > MAX or cooldown blocked — discard
                    self.state = BlinkState.IDLE
                elif duration <= config.BLINK_MAX_DURATION:
                    # Normal blink (50-250ms), wait for second
                    self.blink_end_time = now
                    self.state = BlinkState.WAIT_SECOND
                else:
                    # Ambiguous (250ms-400ms gap), discard
                    self.state = BlinkState.IDLE

        elif self.state == BlinkState.WAIT_SECOND:
            elapsed = now - self.blink_end_time

            if is_high and elapsed < config.DOUBLE_BLINK_WINDOW:
                # Second blink started!
                self.state = BlinkState.IN_BLINK
                self.blink_start_time = now
                self.blink_count = 2
            elif elapsed >= config.DOUBLE_BLINK_WINDOW:
                # Timeout - was just a single blink, ignore it
                self.state = BlinkState.IDLE

        elif self.state == BlinkState.WAIT_THIRD:
            elapsed = now - self.blink_end_time

            if is_high and elapsed < config.TRIPLE_BLINK_WINDOW:
                # Third blink started!
                self.state = BlinkState.IN_BLINK
                self.blink_start_time = now
                self.blink_count = 3
            elif elapsed >= config.TRIPLE_BLINK_WINDOW:
                # Timeout - was a double blink (no third blink came)
                if now - self.last_event_time > config.DOUBLE_BLINK_COOLDOWN:
                    self.last_event_time = now
                    self.state = BlinkState.IDLE
                    logger.debug("Double blink detected")
                    return EOGEvent.DOUBLE_BLINK
                self.state = BlinkState.IDLE

        return EOGEvent.NONE

    def reset(self):
        """Reset state machine."""
        self.state = BlinkState.IDLE
        self.blink_count = 0
        self.last_event_time = -100.0


class GazeDetector:
    """
    Detects sustained gaze direction from EOG values.

    Distinguishes gaze shifts from blinks by requiring the signal
    to stay in the threshold region for a minimum duration.
    """

    def __init__(self):
        self.gaze_start_time = 0.0
        self.current_gaze = EOGEvent.NONE
        self._min_gaze_duration = 0.1  # seconds to confirm gaze

    def update(self, eog: int, now: float = None) -> EOGEvent:
        """
        Feed one EOG sample, return gaze direction if sustained.

        Only returns LOOK_UP or LOOK_DOWN for sub-blink-threshold
        sustained deviations from baseline.
        """
        if now is None:
            now = time.time()

        # Don't detect gaze during blink-level signals
        if eog > config.BLINK_THRESHOLD:
            self.current_gaze = EOGEvent.NONE
            return EOGEvent.NONE

        if eog > config.LOOK_UP_THRESHOLD:
            new_gaze = EOGEvent.LOOK_UP
        elif eog < config.LOOK_DOWN_THRESHOLD:
            new_gaze = EOGEvent.LOOK_DOWN
        else:
            self.current_gaze = EOGEvent.NONE
            return EOGEvent.NONE

        if new_gaze != self.current_gaze:
            self.current_gaze = new_gaze
            self.gaze_start_time = now
            return EOGEvent.NONE

        # Sustained gaze detected
        if now - self.gaze_start_time >= self._min_gaze_duration:
            return self.current_gaze

        return EOGEvent.NONE

    def reset(self):
        self.current_gaze = EOGEvent.NONE


class HorizontalGazeDetector:
    """Detects sustained horizontal gaze from eog_h values."""

    def __init__(self):
        self.gaze_start_time = 0.0
        self.current_gaze = EOGEvent.NONE
        self._min_gaze_duration = 0.15  # slightly longer to avoid false triggers
        self.last_trigger_time = -100.0

    def update(self, eog_h: int, now: float = None) -> EOGEvent:
        """Feed one horizontal EOG sample, return LOOK_LEFT/LOOK_RIGHT if sustained."""
        if now is None:
            now = time.time()

        if eog_h > config.LOOK_RIGHT_THRESHOLD:
            new_gaze = EOGEvent.LOOK_RIGHT
        elif eog_h < config.LOOK_LEFT_THRESHOLD:
            new_gaze = EOGEvent.LOOK_LEFT
        else:
            self.current_gaze = EOGEvent.NONE
            return EOGEvent.NONE

        if new_gaze != self.current_gaze:
            self.current_gaze = new_gaze
            self.gaze_start_time = now
            return EOGEvent.NONE

        if now - self.gaze_start_time >= self._min_gaze_duration:
            if now - self.last_trigger_time > config.HORIZONTAL_GAZE_COOLDOWN:
                self.last_trigger_time = now
                return self.current_gaze

        return EOGEvent.NONE

    def reset(self):
        self.current_gaze = EOGEvent.NONE
        self.last_trigger_time = -100.0


class HeadRollDetector:
    """
    Detects head roll flick from IMU gyro_z for window switching.

    A roll flick is a rapid head tilt that returns to neutral quickly.
    If gz stays above threshold longer than HEAD_ROLL_MAX_DURATION,
    the event is discarded (not an intentional flick).
    
    Only active when cursor_frozen=True (user is looking left or right).
    When cursor_frozen=False, internal state is reset so that stale
    spikes from normal head motion are not carried over.
    """

    def __init__(self):
        self.last_trigger_time = -100.0  # Allow first event immediately
        self._spike_start = None  # When gz first exceeded threshold
        self._spike_direction = None
        self._suppressed = False  # True after held-too-long, until gz drops

    def update(self, gz: int, now: float = None, cursor_frozen: bool = False) -> str | None:
        """
        Feed one gyro_z sample.

        Args:
            gz: Raw gyro_z value.
            now: Current time (default: time.time()).
            cursor_frozen: True when cursor is frozen (looking left/right).
                Only processes roll detection when True.

        Returns:
            "switch_window" if roll flick detected, None otherwise.
        """
        if now is None:
            now = time.time()

        if not cursor_frozen:
            self._spike_start = None
            self._spike_direction = None
            self._suppressed = False
            return None
      
        above = abs(gz) > config.HEAD_ROLL_THRESHOLD

        if above:
            if self._suppressed:
                # Still held after a held-too-long discard, ignore
                pass
            elif self._spike_start is None:
                # Spike just started
                self._spike_start = now
                self._spike_direction = "right" if gz > 0 else "left"
            elif now - self._spike_start > config.HEAD_ROLL_MAX_DURATION:
                # Held too long — not a flick, discard
                self._spike_start = None
                self._spike_direction = None
                self._suppressed = True
        else:
            if self._suppressed:
                self._suppressed = False
            elif self._spike_start is not None:
                # gz returned below threshold — check if it was a valid flick
                duration = now - self._spike_start
                direction = self._spike_direction
                self._spike_start = None
                self._spike_direction = None

                if duration <= config.HEAD_ROLL_MAX_DURATION:
                    if now - self.last_trigger_time > config.HEAD_ROLL_COOLDOWN:
                        self.last_trigger_time = now
                        logger.debug(f"Head roll flick detected ({direction})")
                        return "switch_window"

        return None

    def reset(self):
        self.last_trigger_time = -100.0
        self._spike_start = None
        self._spike_direction = None
        self._suppressed = False


class DoubleNodDetector:
    """
    Detects double head nod (two quick forward nods) from gyro_x for centering cursor.

    Each nod is a gyro_x spike that returns to neutral within MAX_DURATION.
    Two valid nods within DOUBLE_NOD_WINDOW triggers cursor centering.

    Only active when cursor_frozen=True (user is looking left or right).
    When cursor_frozen=False, internal state is reset so that stale
    spikes from normal head motion are not carried over.
    """

    def __init__(self):
        self.last_trigger_time = -100.0
        self._spike_start = None
        self._suppressed = False
        self._first_nod_time = None  # When the first valid nod completed

    def update(self, gx: int, now: float = None, cursor_frozen: bool = False) -> str | None:
        """
        Feed one gyro_x sample.

        Args:
            gx: Raw gyro_x value.
            now: Current time (default: time.time()).
            cursor_frozen: True when cursor is frozen (looking left/right).
                Only processes nod detection when True.

        Returns:
            "center_cursor" if double nod detected, None otherwise.
        """
        if now is None:
            now = time.time()

        if not cursor_frozen:
            self._spike_start = None
            self._suppressed = False
            self._first_nod_time = None
            return None

        above = abs(gx) > config.DOUBLE_NOD_THRESHOLD

        if above:
            if self._suppressed:
                pass
            elif self._spike_start is None:
                self._spike_start = now
            elif now - self._spike_start > config.DOUBLE_NOD_MAX_DURATION:
                # Held too long — not a nod
                self._spike_start = None
                self._suppressed = True
        else:
            if self._suppressed:
                self._suppressed = False
            elif self._spike_start is not None:
                # gx returned below threshold — valid nod?
                duration = now - self._spike_start
                self._spike_start = None

                if duration <= config.DOUBLE_NOD_MAX_DURATION:
                    if self._first_nod_time is not None:
                        # Second nod — check window and cooldown
                        if (now - self._first_nod_time <= config.DOUBLE_NOD_WINDOW
                                and now - self.last_trigger_time > config.DOUBLE_NOD_COOLDOWN):
                            self._first_nod_time = None
                            self.last_trigger_time = now
                            logger.debug("Double nod detected → center cursor")
                            return "center_cursor"
                        else:
                            # Window expired or in cooldown, this nod becomes the new first
                            self._first_nod_time = now
                    else:
                        # First nod
                        self._first_nod_time = now

            # Expire first nod if window passed
            if (self._first_nod_time is not None
                    and now - self._first_nod_time > config.DOUBLE_NOD_WINDOW):
                self._first_nod_time = None

        return None

    def reset(self):
        self.last_trigger_time = -100.0
        self._spike_start = None
        self._suppressed = False
        self._first_nod_time = None
