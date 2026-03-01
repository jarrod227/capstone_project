# Capstone Project Interview Questions

> EOG-IMU Cursor Control System — common questions and suggested talking points.

---

## 1. Project Overview

**Q: Can you describe your capstone project in 1-2 minutes?**

- We built a hands-free cursor control system using eye movements (EOG) and head motion (IMU)
- Two AD8232 boards capture vertical and horizontal eye signals, an MPU9250 tracks head rotation
- STM32 samples all sensors at 200 Hz and streams data over UART to a PC
- Python side does signal processing (low-pass filter, Kalman filter), event detection (blink/gaze state machine), and cursor control (threshold, state-space, or ML mode)
- Practical application: accessibility for users who cannot use a traditional mouse

---

## 2. Design Decisions

**Q: Why did you choose STM32 over Arduino or Raspberry Pi?**

- STM32 has hardware timers (TIM6) for precise 200 Hz sampling — Arduino `millis()` has jitter
- DMA for non-blocking UART TX — Arduino `Serial.print()` blocks the CPU
- Dual independent ADCs — can read both EOG channels without multiplexing delay
- Real-time deterministic timing matters for signal processing downstream
- Raspberry Pi is overkill and runs Linux (non-real-time OS), not ideal for precise sensor sampling

**Q: Why 200 Hz sampling rate?**

- EOG signals are 0–30 Hz bandwidth, Nyquist requires > 60 Hz minimum
- 200 Hz gives comfortable oversampling (>6x) for clean filtering
- Fast enough to detect rapid blinks (~150–400 ms) with good temporal resolution
- Not so fast that it overwhelms UART bandwidth or PC processing

**Q: Why use DMA for UART instead of polling or interrupt?**

- Polling (`HAL_UART_Transmit`) blocks the CPU — would miss the next 5 ms sample window
- Byte-by-byte interrupts have overhead per byte
- DMA transfers the entire formatted string (~40 bytes) in the background while the CPU reads the next set of sensors
- Ping-pong double buffering: CPU fills buffer A while DMA sends buffer B, zero idle time

**Q: Why split the system into STM32 firmware + Python PC software?**

- STM32 handles time-critical sampling (deterministic 200 Hz)
- Python handles compute-heavy processing (Kalman filter, ML inference, GUI control)
- Separation of concerns: firmware is simple and reliable, Python is flexible and easy to iterate
- Allows testing with CSV replay or simulator when hardware is unavailable

---

## 3. Technical Deep Dives

**Q: How does your blink detection work?**

- 4-state state machine: IDLE → BLINK_START → BLINK_END → WAIT_NEXT
- Vertical EOG spike above threshold (3000) = blink onset
- Blink duration determines type: < 250 ms = short blink, > 400 ms = long blink (right click)
- After first blink ends, we wait up to 600 ms for a second blink
- Double blink = left click, triple blink = double click, single blink = ignored (prevents false triggers)

**Q: Explain your Kalman filter and why you need it.**

- IMU gyroscope has slowly drifting bias — raw readings drift even when the head is still
- 2-state Kalman filter per axis: tracks true angular velocity and bias simultaneously
- Prediction step: bias is modeled as a random walk (slow change)
- Update step: measurement corrects both states
- Result: clean motion signal with bias removed, cursor doesn't drift when head is stationary

**Q: What is the state-space cursor model?**

- 4-state system: [pos_x, vel_x, pos_y, vel_y]
- Gyro input drives velocity, position integrates from velocity
- Velocity decay factor (0.95): cursor glides naturally and stops, like a physical object with friction
- Feels more natural than directly mapping gyro → pixel displacement (threshold mode)

**Q: How does sensor fusion work in your system?**

- Scroll: requires BOTH eye gaze (up/down) AND head tilt to agree — prevents false triggers
- Browser navigation: requires BOTH horizontal gaze AND head turn
- Cursor freeze: looking far left/right freezes cursor movement so nod gestures can center it
- Principle: requiring two independent signals to agree dramatically reduces false positives

---

## 4. Challenges & Problem Solving

**Q: What was the hardest technical challenge?**

- **Blink vs. gaze disambiguation**: both cause vertical EOG changes — solved with amplitude threshold + duration-based state machine
- **Gyro drift**: cursor slowly moved on its own — solved with Kalman filter bias tracking
- **False click triggers**: single involuntary blinks caused clicks — solved by requiring double blink (single blink is ignored)
- **Timing jitter**: inconsistent sampling broke filter assumptions — solved with hardware timer (TIM6) instead of software delay

**Q: How did you test without hardware?**

- Three data sources: real hardware, CSV replay, keyboard simulator
- `generate_demo_data.py` creates deterministic synthetic data (seed=42) with known blink/gaze events
- CSV replay lets us regression-test against recorded sessions
- Keyboard simulator maps keys to sensor values for interactive testing
- 70 unit tests covering event detection, signal processing, ML pipeline

**Q: If you could redo this project, what would you change?**

- Use SPI instead of I2C for faster IMU reads
- Add wireless (BLE) to eliminate the USB tether
- Implement adaptive thresholds that calibrate per user
- Use a more powerful ML model (e.g., LSTM) for temporal gesture patterns
- Add visual feedback overlay showing detected gaze direction

---

## 5. Software Engineering Practices

**Q: How is your codebase organized?**

- Firmware: `main.c` + `mpu9250.c/h` — minimal, only does sampling and transmission
- Python: modular packages — `signal_processing`, `event_detector`, `cursor_control`, `ml_classifier`
- Config: all 100+ tunable parameters in one `config.py` file
- Tests: 70 tests across 4 test files, runnable with `pytest`
- Docs: technical write-ups for Kalman filter math, state-space derivation, data flow diagrams

---

## 6. Quick Technical Facts

Keep these numbers in your head for rapid-fire questions:

| Topic | Value |
|-------|-------|
| Sampling rate | 200 Hz (5 ms period) |
| EOG bandwidth | 0–30 Hz |
| ADC resolution | 12-bit (0–4095) |
| Blink threshold | 3000 (vertical EOG) |
| Short blink duration | < 250 ms |
| Long blink duration | > 400 ms |
| Double blink window | 600 ms |
| Gyro deadzone | 300 raw units |
| Kalman states | 2 per axis (velocity + bias) |
| State-space states | 4 (pos_x, vel_x, pos_y, vel_y) |
| UART baud rate | 115200 |
| I2C speed | 400 kHz (Fast Mode) |
| ML model | SVM, 9 classes, 20 features |
| Test count | 70 unit tests |
| Data format | `timestamp,eog_v,eog_h,gyro_x,gyro_y,gyro_z` |
