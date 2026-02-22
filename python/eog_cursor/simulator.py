"""
Hardware simulator for development and demo without physical hardware.

Generates synthetic EOG and IMU data that mimics real sensor behavior,
allowing the full pipeline to be tested and demonstrated without
an STM32 board.

Keyboard controls:
  Arrow keys:       Simulate head motion (IMU gyro X/Y)
  Space (tap x2):   Simulate double blink → left click
  Space (hold):     Simulate long blink → right click
  Space (tap x3):   Simulate triple blink → double click
  L/R + N (tap x2): Simulate look left/right + double nod → center cursor
  U + Arrow Up:     Simulate look up + head up → scroll up
  D + Arrow Down:   Simulate look down + head down → scroll down
  L + Arrow Left:   Simulate look left + head left → browser back
  R + Arrow Right:  Simulate look right + head right → browser forward
  L/R + W:          Simulate look left/right + head roll flick → window switch
  Q / Escape:       Quit

EOG signal model (12-bit ADC, dual-channel):
  Vertical (eog_v):   Baseline 2048 | Blink/Up: >2048 | Down: <2048
  Horizontal (eog_h): Baseline 2048 | Right: >2048 | Left: <2048
"""

import time
import logging
from dataclasses import dataclass

import numpy as np

from . import config
from .serial_reader import SensorPacket

logger = logging.getLogger(__name__)


@dataclass
class SimState:
    """Mutable state for the simulator."""
    gyro_x: float = 0.0
    gyro_y: float = 0.0
    gyro_z: float = 0.0
    blink: bool = False
    look_up: bool = False
    look_down: bool = False
    look_left: bool = False
    look_right: bool = False
    head_roll: bool = False
    head_nod: bool = False
    running: bool = True


class HardwareSimulator:
    """
    Generates synthetic sensor packets from keyboard input.

    Replaces SerialReader for testing without hardware.
    Uses pynput for cross-platform keyboard listening.
    """

    def __init__(self):
        self.state = SimState()
        self._start_time = time.time()
        self._listener = None
        self._gyro_magnitude = 2000  # Simulated gyro amplitude

    def _on_key_press(self, key):
        """Handle key press events."""
        try:
            from pynput.keyboard import Key
            if key == Key.left:
                self.state.gyro_y = -self._gyro_magnitude
            elif key == Key.right:
                self.state.gyro_y = self._gyro_magnitude
            elif key == Key.up:
                self.state.gyro_x = -self._gyro_magnitude
            elif key == Key.down:
                self.state.gyro_x = self._gyro_magnitude
            elif key == Key.space:
                self.state.blink = True
            elif key == Key.esc:
                self.state.running = False
        except AttributeError:
            pass

        if hasattr(key, 'char') and key.char:
            if key.char == 'u':
                self.state.look_up = True
            elif key.char == 'd':
                self.state.look_down = True
            elif key.char == 'l':
                self.state.look_left = True
            elif key.char == 'r':
                self.state.look_right = True
            elif key.char == 'w':
                self.state.head_roll = True
            elif key.char == 'n':
                self.state.head_nod = True
            elif key.char == 'q':
                self.state.running = False

    def _on_key_release(self, key):
        """Handle key release events."""
        try:
            from pynput.keyboard import Key
            if key in (Key.left, Key.right):
                self.state.gyro_y = 0.0
            elif key in (Key.up, Key.down):
                self.state.gyro_x = 0.0
            elif key == Key.space:
                self.state.blink = False
        except AttributeError:
            pass

        if hasattr(key, 'char') and key.char:
            if key.char == 'u':
                self.state.look_up = False
            elif key.char == 'd':
                self.state.look_down = False
            elif key.char == 'l':
                self.state.look_left = False
            elif key.char == 'r':
                self.state.look_right = False
            elif key.char == 'w':
                self.state.head_roll = False
            elif key.char == 'n':
                self.state.head_nod = False

    def start(self):
        """Start keyboard listener in background thread."""
        from pynput.keyboard import Listener
        self._listener = Listener(
            on_press=self._on_key_press,
            on_release=self._on_key_release
        )
        self._listener.daemon = True
        self._listener.start()
        self._start_time = time.time()
        logger.info("Hardware simulator started.")
        logger.info("Arrows=move, Space(x2)=left-click, Space(hold)=right-click, Space(x3)=double-click")
        logger.info("Space(x3)=double-click, L/R+N(x2)=center-cursor, U+Up=scroll-up, D+Down=scroll-down")
        logger.info("L+Left=back, R+Right=forward, L/R+W=window-switch, Q=quit")

    def stop(self):
        """Stop keyboard listener."""
        self.state.running = False
        if self._listener:
            self._listener.stop()
        logger.info("Hardware simulator stopped.")

    def generate_packet(self) -> SensorPacket:
        """
        Generate one synthetic sensor packet based on current state.

        EOG vertical channel (12-bit ADC, baseline 2048):
          Blink:    ~3500  (spike above baseline)
          Look Up:  ~2900  (moderate positive shift)
          Look Down: ~1000 (negative shift below baseline)
          Idle:     ~2048  (baseline with noise)

        EOG horizontal channel (12-bit ADC, baseline 2048):
          Look Right: ~2900 (positive shift above baseline)
          Look Left:  ~1000 (negative shift below baseline)
          Idle:       ~2048 (baseline with noise)
        """
        now = time.time()
        elapsed_ms = int((now - self._start_time) * 1000)

        # --- Simulate EOG vertical channel (12-bit, baseline 2048) ---
        eog_baseline = config.EOG_BASELINE
        eog_v_noise = np.random.normal(0, config.SIM_NOISE_STD)
        eog_v = eog_baseline + eog_v_noise

        if self.state.blink:
            # Blink: large positive spike (~3500)
            eog_v += 1500 + np.random.normal(0, 200)
        elif self.state.look_up:
            # Look up: moderate positive shift (~2900)
            eog_v += 850 + np.random.normal(0, 50)
        elif self.state.look_down:
            # Look down: below baseline (~1000)
            eog_v -= 1050 + np.random.normal(0, 50)

        eog_v = int(np.clip(eog_v, 0, 4095))

        # --- Simulate EOG horizontal channel (12-bit, baseline 2048) ---
        eog_h_noise = np.random.normal(0, config.SIM_NOISE_STD)
        eog_h = eog_baseline + eog_h_noise

        if self.state.look_right:
            # Look right: positive shift (~2900)
            eog_h += 850 + np.random.normal(0, 50)
        elif self.state.look_left:
            # Look left: below baseline (~1000)
            eog_h -= 1050 + np.random.normal(0, 50)

        eog_h = int(np.clip(eog_h, 0, 4095))

        # --- Simulate IMU gyro ---
        # Head nod → gx spike (overrides arrow key gx)
        if self.state.head_nod:
            gx = int(4000 + np.random.normal(0, 200))
        else:
            gx = int(self.state.gyro_x + np.random.normal(0, config.SIM_GYRO_NOISE_STD))
        gy = int(self.state.gyro_y + np.random.normal(0, config.SIM_GYRO_NOISE_STD))

        # Head roll flick → gz spike
        if self.state.head_roll:
            gz = int(4000 + np.random.normal(0, 200))
        else:
            gz = int(np.random.normal(0, config.SIM_GYRO_NOISE_STD))

        return SensorPacket(
            timestamp=elapsed_ms,
            eog_v=eog_v,
            eog_h=eog_h,
            gyro_x=gx,
            gyro_y=gy,
            gyro_z=gz,
            pc_time=now
        )

    def stream(self):
        """Generator that yields simulated packets at the configured sample rate."""
        self.start()
        try:
            while self.state.running:
                yield self.generate_packet()
                time.sleep(config.SAMPLE_PERIOD)
        finally:
            self.stop()

    def __enter__(self):
        self.start()
        return self

    def __exit__(self, exc_type, exc_val, exc_tb):
        self.stop()
        return False
