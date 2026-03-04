# Capstone Project Interview Questions

> EOG-IMU Cursor Control System — common questions and suggested talking points.

---

## 1. Project Overview

**Q: Can you describe your capstone project in 1-2 minutes?**

- Basically, we built a hands-free mouse — you move the cursor by turning your head, click by blinking, and scroll by looking up or down
- On the hardware side, there are two small amplifier boards (AD8232) that pick up electrical signals from eye movements, and a gyroscope (MPU9250) on the head that senses rotation
- An STM32 board reads all the sensors 200 times a second and sends the data to a PC over USB
- The Python software then does the heavy lifting — filtering out noise, figuring out what action the user is doing, and actually moving the cursor
- The whole point is to help people who can't use a regular mouse, like someone with a spinal cord injury

---

## 2. Design Decisions

**Q: Why did you choose STM32 over Arduino or Raspberry Pi?**

- Mainly because of timing precision — we need exactly 200 samples per second, and STM32 has hardware timers that can do that dead-on. Arduino's `millis()` has jitter, which messes up our filters
- STM32 also has DMA, so it can send data over UART in the background without blocking the CPU. Arduino's `Serial.print()` stalls everything until it's done
- It has dual ADCs too, so we can read both eye channels at the same time without switching back and forth
- Raspberry Pi would work for the processing side, but it runs Linux which isn't real-time — not great for precise sensor sampling

**Q: Why 200 Hz sampling rate?**

- Eye signals go up to about 30 Hz, so by Nyquist you need at least 60 Hz. 200 gives us plenty of headroom — over 6x oversampling
- It's also fast enough to catch quick blinks, which can be as short as 150 ms
- But not so fast that it floods the serial port or overwhelms the PC. It's a sweet spot

**Q: Why use DMA for UART instead of polling or interrupt?**

- Polling means the CPU writes each byte to the UART register and waits for the "transmit done" flag before sending the next one — it's stuck in a busy loop for the entire transfer. At 115200 baud, a 40-byte packet ties up the CPU for about 3.5 ms, which is most of our 5 ms sample window
- Interrupt-per-byte is better — the CPU kicks off a byte and goes back to work, then gets interrupted when it's time to load the next byte. But you still get an interrupt for every single byte, so there's overhead
- DMA is different because the DMA controller is a separate piece of hardware that moves data on its own. You just tell it "here's a buffer, here's how many bytes, go" — and then the CPU is completely free. It's not waiting, not getting interrupted, it just goes and reads the next set of sensors while the DMA controller feeds bytes to UART independently
- On top of that, we use a ping-pong setup: the CPU fills one buffer while DMA sends the other, so there's zero downtime

**Q: How is DMA implemented in your firmware specifically?**

- There are two 80-byte buffers that alternate — while DMA is sending one, the CPU writes into the other
- A simple `tx_idx ^= 1` flips which buffer is active each cycle
- `HAL_UART_Transmit_DMA()` kicks off a non-blocking send, and a `dma_busy` flag prevents us from starting another send before the current one finishes
- The key thing people miss: you need the `TxCpltCallback` to clear that flag when DMA is done — without it, `dma_busy` stays stuck at 1 and only the first frame ever gets sent
- There's also a watchdog: if DMA appears stuck for more than 2 ticks (~10 ms), we force-abort and reset
- Error callback handles UART errors like overrun, so the system recovers instead of locking up permanently
- The math works out nicely: 40 bytes at 115200 baud is about 3.5 ms, well within our 5 ms window

**Q: Why split the system into STM32 firmware + Python PC software?**

- Each part does what it's best at — STM32 is great at precise, repetitive sampling; Python is great at complex processing and has all the libraries we need for Kalman filters, ML, mouse control
- It also means we can test the Python side without hardware — just replay a CSV file or use a keyboard simulator
- And the firmware stays dead simple — just read sensors, format a line, send it. Less code means fewer bugs in the hard-to-debug part

---

## 3. Technical Deep Dives

**Q: How does your blink detection work?**

- It's a state machine with four states: IDLE, IN_BLINK, WAIT_SECOND, and WAIT_THIRD
- When the vertical eye signal spikes above 2600, that's a blink starting
- How long the eye stays closed tells us the type: under 250 ms is a normal blink, over 400 ms is a long blink which maps to right-click
- After the first blink ends, we wait up to 600 ms to see if another blink comes
- Two blinks in that window = left click, three = double click. A single blink by itself is just ignored — this prevents accidental clicks from involuntary blinks

**Q: Explain your Kalman filter and why you need it.**

- The gyroscope has this annoying problem: even when your head is perfectly still, the reading slowly drifts. Without fixing this, the cursor would creep across the screen on its own
- So we run a Kalman filter that tracks two things per axis: the actual angular velocity, and the bias (that drifting offset)
- Each cycle, it predicts where those values should be, then corrects based on the new reading
- The bias is modeled as changing very slowly, and the angular velocity can change fast — the filter figures out the right balance
- End result: when you stop moving your head, the cursor actually stops

**Q: What is the state-space cursor model?**

- Instead of directly converting gyro readings to pixel movement, we treat the cursor like a physical object with position and velocity
- The gyro input drives the velocity, and position updates from velocity — just like real physics
- There's a decay factor of 0.95 on velocity, so when you stop turning your head, the cursor glides to a stop naturally instead of freezing instantly
- It feels way more like a real mouse than the naive approach of "gyro value = pixel jump"

**Q: How does sensor fusion work in your system?**

- Scroll: requires BOTH eye gaze (up/down) AND head tilt to agree — prevents false triggers
- Browser navigation: requires BOTH horizontal gaze AND head turn
- Cursor freeze: looking far left/right freezes cursor movement so nod gestures can center it
- Principle: requiring two independent signals to agree dramatically reduces false positives

---

## 4. Challenges & Problem Solving

**Q: What was the hardest technical challenge?**

- **Blink vs. looking up** — both push the vertical eye signal upward, so they look similar. We solved it by checking duration: blinks are quick spikes (under 250 ms), looking up is sustained. Amplitude matters too — blinks tend to be sharper
- **Gyro drift** — the cursor kept slowly moving on its own. This was the most frustrating one. Fixed it with the Kalman filter that tracks and removes the bias in real-time
- **Accidental clicks** — people blink involuntarily all the time, so single blinks were causing random clicks. Solution: only double blink triggers a click. Single blinks are just ignored
- **Sampling jitter** — at first we used software delays, which gave inconsistent timing and made the filters behave unpredictably. Switching to a hardware timer (TIM6) fixed it completely

**Q: How did you test without hardware?**

- We set up three ways to feed data into the system: real hardware, CSV replay, and a keyboard simulator
- There's a script that generates synthetic data with known events — like "blink at t=2.0s, look left at t=5.0s" — so we can verify the detection code catches everything
- CSV replay lets us record a real session once and re-test against it whenever we change the code
- The keyboard simulator maps keys to sensor values so you can interactively test without electrodes
- On top of that, we have about 70 unit tests covering event detection, signal processing, and the ML pipeline

**Q: If you could redo this project, what would you change?**

- Switch from I2C to SPI for the IMU — it's faster and we don't really need I2C's multi-device addressing
- Go wireless with BLE so there's no USB cable tethering the user to the computer
- Make the thresholds auto-calibrate per user — right now the values are tuned for me, someone else might need different settings
- Try a temporal ML model like an LSTM that can learn gesture patterns over time, not just static windows
- Add a visual overlay showing what the system is detecting — useful for both debugging and user feedback

---

## 5. Software Engineering Practices

**Q: How is your codebase organized?**

- The firmware is minimal on purpose — just `main.c` for sampling logic and `mpu9250.c` for the IMU driver. That's it
- Python side is split into modules: signal processing, event detection, cursor control, and ML classification. Each one is independent and testable
- Every tunable parameter (over 100 of them) lives in one `config.py` file — so adjusting thresholds or filter settings is just changing a number, not hunting through code
- 70 unit tests across 4 test files, all runnable with `pytest`
- Documentation covers the math behind the Kalman filter, the state-space model derivation, and data flow diagrams

---

## 6. Quick Technical Facts

Handy numbers for rapid-fire questions:

| Topic | Value |
|-------|-------|
| Sampling rate | 200 Hz (5 ms period) |
| EOG bandwidth | 0–30 Hz |
| ADC resolution | 12-bit (0–4095) |
| Blink threshold | 2600 (vertical EOG) |
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
