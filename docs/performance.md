# Performance & Robustness

Quantitative evaluation of the EOG cursor control system — **ML mode with real hardware**.

> **Status:** Template — fill in after hardware testing with real EOG data.

---

## 1. ML Classification Accuracy

Trained on real EOG sessions, evaluated with 5-fold cross-validation (not training set).

| Class | Precision | Recall | F1 | Support |
|-------|-----------|--------|----|---------|
| idle | — | — | — | — |
| blink | — | — | — | — |
| double_blink | — | — | — | — |
| long_blink | — | — | — | — |
| look_up | — | — | — | — |
| look_down | — | — | — | — |
| look_left | — | — | — | — |
| look_right | — | — | — | — |
| **Weighted avg** | — | — | — | — |

**How to measure:**

1. Collect real data (multiple sessions, varied lighting/posture):
   ```bash
   cd python
   python -m scripts.collect_data --port COM4
   ```
2. Train with cross-validation:
   ```bash
   python -m scripts.train_model --data ../data/raw --cv-folds 5
   ```
3. Copy the cross-validation classification report (not the training set report).

---

## 2. Confusion Matrix

```
(paste confusion matrix here)
```

Same command as above.

---

## 3. End-to-End Latency

Time from serial sample arrival to pyautogui action execution.

| Component | Latency | Notes |
|-----------|---------|-------|
| Serial read + parse | — ms | `readline()` + `split(",")` |
| EOG low-pass filter | — ms | Per-sample IIR |
| Feature extraction | — ms | 20 features from 100-sample window |
| SVM prediction | — ms | `model.predict()` on 1 feature vector |
| State-space cursor update | — ms | 4x4 matrix multiply |
| pyautogui action | — ms | OS mouse/keyboard API |
| **Total pipeline** | — ms | Serial → filter → SVM → action + cursor |

**How to measure:**

Add timing to `run_ml_mode()` in `main.py`:

```python
import time
sample_count = 0

for packet in source.stream():
    t0 = time.perf_counter()

    # ML mode uses raw EOG (no filtering — must match training data)
    prediction = classifier.predict(float(packet.eog_v), float(packet.eog_h))
    # ... (fusion + action logic) ...
    controller.update(config.EOG_BASELINE, config.EOG_BASELINE,
                      packet.gyro_x, packet.gyro_y, packet.gyro_z)

    dt_ms = (time.perf_counter() - t0) * 1000
    sample_count += 1
    if sample_count % 200 == 0:
        print(f"Pipeline latency: {dt_ms:.2f} ms")
```

Target: < 5 ms per sample (one period at 200 Hz).

---

## 4. Action Accuracy

Test each action with real EOG signals. Perform N intentional gestures, count successes and false triggers.

| Action | Attempts | Successes | False Positives | Accuracy |
|--------|----------|-----------|-----------------|----------|
| Double blink → left click | — | — | — | — |
| Triple blink → double click | — | — | — | — |
| Long blink → right click | — | — | — | — |
| Center cursor (look L/R + double nod) | — | — | — | — |
| Scroll up (eye up + head up) | — | — | — | — |
| Scroll down (eye down + head down) | — | — | — | — |
| Browser back (eye left + head left) | — | — | — | — |
| Browser forward (eye right + head right) | — | — | — | — |
| Window switch (look L/R + head roll) | — | — | — | — |
| Cursor move (head motion) | — | — | — | — |

**How to measure:**

1. Run ML mode with hardware: `python main.py --port COM4 --mode ml`
2. Perform each gesture 20 times with ~3s gaps
3. Record: successful triggers, missed triggers, false triggers during idle

---

## 5. Robustness

| Test | Result | Notes |
|------|--------|-------|
| False positive rate (idle, 5 min) | — / min | Unintended actions while sitting still |
| Baseline drift (10+ min session) | — | Accuracy in first 2 min vs. last 2 min |
| Head-only rejection | — | Head motion without eye gaze should NOT trigger scroll/nav |
| Eye-only rejection | — | Eye gaze without head motion should NOT trigger scroll/nav |
| Continuous runtime | — min | Max duration before needing restart |

**How to measure:**

- False positive: run ML mode, sit idle for 5 minutes, count unintended actions
- Baseline drift: record a 15-min session, compare event detection accuracy at start vs. end
- Fusion rejection: intentionally move only head (no eye gaze) or only eyes (no head) — should produce no scroll/nav actions

---

## Summary

| Metric | Target | Measured | Status |
|--------|--------|----------|--------|
| ML accuracy (weighted F1) | > 0.90 | — | — |
| End-to-end latency | < 5 ms | — | — |
| Action accuracy (avg) | > 0.85 | — | — |
| False positive rate (idle) | < 1 / min | — | — |
| Continuous runtime | > 30 min | — | — |
