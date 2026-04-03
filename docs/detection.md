# EOG Event Detection — Technical Details

This document describes how EOG events (blinks, gaze shifts) are detected from raw ADC signals. For a high-level overview, see the [README](../README.md).

Source code: [`python/eog_cursor/event_detector.py`](../python/eog_cursor/event_detector.py)

## EOG Signal Characteristics

The vertical EOG channel (eog_v) captures two fundamentally different signal types:

```
Blink waveform:                    Gaze shift (look up):

eog_v                              eog_v
  ^                                  ^
  │     ╭──╮                         │  ╭──────────────╮
  │    ╱    ╲    Peak > 2600         │ ╱                ╲   ~2400, sustained
  │───╱──────╲── Baseline 2048       │╱──────────────────╲─ Baseline 2048
  │                                  │
  └──────────> time                  └──────────────────> time
     50-250ms                            seconds
```

**Key difference**: blinks are fast transient spikes (50–250ms) with peak > 2600; gaze shifts are slower, sustained deviations (~2400) that stay below the blink threshold.

## Blink Detection State Machine

The `BlinkDetector` discriminates between double blinks (→ left click), triple blinks (→ double click), and long blinks (→ right click) using a 4-state machine that analyzes the **temporal pattern** of the signal, not just its amplitude.

### State Diagram

```
                    ┌──────────────────────────────────────────────────────┐
                    │                                                      │
                    ▼                                                      │
                 ┌──────┐                                                  │
          ┌─────>│ IDLE │<──────────────────────────────────┐              │
          │      └──┬───┘                                   │              │
          │         │ eog > BLINK_THRESHOLD (2600)          │              │
          │         │ record blink_start_time               │              │
          │         ▼                                       │              │
          │   ┌──────────┐                                  │              │
          │   │ IN_BLINK │───────────────────────────┐      │              │
          │   │ count=1  │                           │      │              │
          │   └──┬───┬───┘                           │      │              │
          │      │   │                               │      │              │
          │      │   │ eog < threshold               │      │              │
          │      │   │ (signal dropped back)         │      │              │
          │      │   │                               │      │              │
          │      │   ├─ duration < 50ms              │      │              │
          │      │   │  → noise, reset ──────────────┼──────┘              │
          │      │   │                               │                     │
          │      │   ├─ 50ms ≤ duration ≤ 250ms      │                     │
          │      │   │  → valid blink                │                     │
          │      │   │                               │                     │
          │      │   ▼                               │                     │
          │   ┌─────────────┐                        │ signal drops after  │
          │   │ WAIT_SECOND │                        │ 0.4s–2.5s held     │
          │   │(blink ended,│                        │                     │
          │   │ wait 0.6s)  │                        ▼                     │
          │   └──┬──────┬───┘              ┌──────────────────┐            │
          │      │      │                  │  EMIT LONG_BLINK │            │
          │      │      │                  │  (→ right click)  │           │
          │      │      │                  │  fires on RELEASE │           │
          │      │      │                  └──────────────────┘            │
          │      │      │                                                  │
          │      │      │ timeout > 0.6s (no 2nd blink)                   │
          │      │      └─ single blink, ignored ─────────────────────────┘
          │      │
          │      │ eog > threshold again (2nd spike within 0.6s)
          │      ▼
          │   ┌──────────┐
          │   │ IN_BLINK │
          │   │ count=2  │
          │   └────┬─────┘
          │        │ eog < threshold (2nd blink ended, valid duration)
          │        ▼
          │   ┌─────────────┐
          │   │ WAIT_THIRD  │
          │   │(2 blinks,   │
          │   │ wait 0.6s)  │
          │   └──┬──────┬───┘
          │      │      │
          │      │      │ timeout > 0.6s (no 3rd blink)
          │      │      ▼
          │      │  ┌───────────────────┐
          │      │  │ EMIT DOUBLE_BLINK │
          │      │  │  (→ left click)   │
          │      │  └───────┬───────────┘
          │      │          │
          │      │      ┌───┘
          │      │      │
          │      │ eog > threshold again (3rd spike within 0.6s)
          │      ▼
          │   ┌──────────┐
          │   │ IN_BLINK │
          │   │ count=3  │
          │   └────┬─────┘
          │        │ eog < threshold (3rd blink ended)
          │        ▼
          │  ┌───────────────────┐
          │  │ EMIT TRIPLE_BLINK │
          │  │  (→ double click) │
          │  └───────────────────┘
          │        │
          └────────┘
```

### Parameters (from `config.py`)

| Parameter | Value | Purpose |
|-----------|-------|---------|
| `BLINK_THRESHOLD` | 2600 | ADC value for rising/falling edge detection |
| `BLINK_MIN_DURATION` | 50ms | Debounce — reject noise spikes |
| `BLINK_MAX_DURATION` | 250ms | Maximum duration for a normal blink |
| `DOUBLE_BLINK_WINDOW` | 600ms | Maximum gap between two blinks to count as double |
| `LONG_BLINK_MIN_DURATION` | 400ms | Minimum hold time for long blink |
| `LONG_BLINK_MAX_DURATION` | 2.5s | Maximum hold — beyond this is likely sustained gaze or noise |
| `DOUBLE_BLINK_COOLDOWN` | 800ms | Prevent re-trigger after double blink |
| `TRIPLE_BLINK_WINDOW` | 600ms | Maximum gap after 2nd blink for 3rd to count as triple |
| `TRIPLE_BLINK_COOLDOWN` | 1.0s | Prevent re-trigger after triple blink |
| `LONG_BLINK_COOLDOWN` | 1.0s | Prevent re-trigger after long blink |

### Why Not Just Peak Detection?

A simple "eog > 2600" check cannot distinguish:
- **Blink** (50–250ms spike) vs **long blink** (>=400ms sustained) vs **sustained look up** (seconds-long shift at ~2400)
- **Single blink** (ignored) vs **double blink** (left click) vs **triple blink** (double click)
- **Noise spike** (<50ms) vs **real blink**

The state machine solves this by tracking the **rising edge, duration, and falling edge** of each positive deflection.

## Gaze Detection (Vertical)

The `GazeDetector` handles sustained vertical gaze shifts for scroll fusion.

### Key Design: Excluding Blink-Level Signals

```python
# event_detector.py — GazeDetector.update()
if eog > config.BLINK_THRESHOLD:  # > 2600
    self.current_gaze = EOGEvent.NONE
    return EOGEvent.NONE
```

Any signal above 2600 is handed off to the `BlinkDetector`. The `GazeDetector` only processes signals in the "gaze zone":
- **Look Up**: 2400 < eog_v < 2600 (sustained for >100ms)
- **Look Down**: eog_v < 1600 (sustained for >100ms)

This prevents a blink spike from being misclassified as a gaze event.

### Signal Zones

```
ADC value
  3000 ─ ─ ─ ─ ─ ─ ─ ─ ─ ─ ─ ─ ─ ─ ─ ─ ─ ─ ─
         BLINK ZONE (BlinkDetector only)
  2600 ━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━  ← BLINK_THRESHOLD
         LOOK UP ZONE (GazeDetector)
  2400 ━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━  ← LOOK_UP_THRESHOLD
         NEUTRAL ZONE (idle)
  2048 - - - - - - - - - - - - - - - - - - - - -  ← Baseline
         NEUTRAL ZONE (idle)
  1600 ━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━  ← LOOK_DOWN_THRESHOLD
         LOOK DOWN ZONE (GazeDetector)
  1200 ─ ─ ─ ─ ─ ─ ─ ─ ─ ─ ─ ─ ─ ─ ─ ─ ─ ─ ─
```

## Horizontal Gaze Detection

The `HorizontalGazeDetector` operates on the eog_h channel with similar logic but a longer minimum duration (150ms vs 100ms) to reduce false triggers from eye saccades.

**Important**: Horizontal gaze events alone do not trigger browser navigation. They lock the system into a nav-ready state; head turn then confirms the action (two-step, same pattern as scroll).

### Nav Ready State Machine

| State | Condition | Cursor |
|-------|-----------|--------|
| `IDLE` | Eyes in neutral zone | Moves normally |
| `NAV_LEFT_READY` | eog_h in look-left zone (sustained >150ms) | Frozen |
| `NAV_RIGHT_READY` | eog_h in look-right zone (sustained >150ms) | Frozen |

### Transitions

```
Eyes neutral
    │
    ▼
  IDLE ──── look left (eog_h < 1600) ────► NAV_LEFT_READY
    │                                           │
    │                                           │ head turns left (gy < -300)
    │                                           │     → browser back fires
    │                                           │ eyes return to neutral
    │                                           ▼
    └──── look right (eog_h > 2400) ──► NAV_RIGHT_READY
                                                │
                                                │ head turns right (gy > 300)
                                                │     → browser forward fires
                                                │ eyes return to neutral
                                                ▼
                                              IDLE
```

| Action | Eye Condition | Head Condition |
|--------|--------------|----------------|
| Browser Back | eog_h < 1600 for >150ms (NAV_LEFT_READY) | gy < -300 (head turning left) |
| Browser Forward | eog_h > 2400 for >150ms (NAV_RIGHT_READY) | gy > 300 (head turning right) |

Implemented via `nav_state` in `_BaseController` (threshold/statespace modes) and `ml_nav_state` in `run_ml_mode()` (ML mode).

## Vertical Gaze — Scroll Ready State Machine

Scroll uses a **two-step state machine** rather than a simultaneous eye+head check. This makes scroll reliable when eye and head movements are not perfectly synchronised.

### States

| State | Condition | Cursor |
|-------|-----------|--------|
| `IDLE` | Eyes in neutral zone | Moves normally |
| `SCROLL_UP_READY` | eog_v in look-up zone (sustained >100ms) | Frozen |
| `SCROLL_DOWN_READY` | eog_v in look-down zone (sustained >100ms) | Frozen |

### Transitions

```
Eyes neutral
    │
    ▼
  IDLE ──── look up (eog_v > 2400) ────► SCROLL_UP_READY
    │                                          │
    │                                          │ head tilts up (gx < -300)
    │                                          │     → scroll up fires
    │                                          │ eyes return to neutral
    │                                          ▼
    └──── look down (eog_v < 1600) ──► SCROLL_DOWN_READY
                                               │
                                               │ head tilts down (gx > 300)
                                               │     → scroll down fires
                                               │ eyes return to neutral
                                               ▼
                                             IDLE
```

**Workflow:** Look up/down (cursor freezes, enters ready state) → tilt head (scroll fires, repeats while in state) → return eyes to neutral (exits state, cursor unfreezes).

While in `SCROLL_UP_READY` or `SCROLL_DOWN_READY`, no other cursor actions are possible — blink detection still runs, but cursor movement and navigation are suppressed. This is implemented via `scroll_state` in `_BaseController` (threshold/statespace modes) and `ml_scroll_state` in `run_ml_mode()` (ML mode).

## ML-Based Detection (Alternative)

In `--mode ml`, the SVM classifier replaces the threshold-based state machine. Instead of analyzing edges and durations in real time, it extracts 20 features from a 1.0s sliding window of both EOG channels and classifies the waveform pattern:

| Feature | What It Captures |
|---------|-----------------|
| `peak_amplitude` | Maximum deflection from baseline (blink vs gaze magnitude) |
| `max_derivative` | Fastest rate of change (blinks have steep edges, gaze is gradual) |
| `skewness` | Waveform asymmetry (blinks are positively skewed spikes) |
| `kurtosis` | Peak sharpness (blinks have high kurtosis, gaze is flat) |
| `zero_crossings` | Signal oscillation frequency |
| `slope` | Overall trend direction (up vs down vs neutral) |
| `mean`, `std` | Signal level and variability |
| `rms` | Signal energy |
| `derivative_variance` | Roughness/regularity of change |

These 10 features are extracted from **both** eog_v and eog_h channels (20 total), allowing the classifier to distinguish all 9 event classes (including triple blink and horizontal gaze directions that are invisible on eog_v alone).

## Double Nod Detection

The `DoubleNodDetector` detects two quick forward head nods for cursor centering. Each nod is a gyro_x spike that returns to neutral within `DOUBLE_NOD_MAX_DURATION` (0.3s). Two valid nods within `DOUBLE_NOD_WINDOW` (0.8s) triggers the event, moving the cursor to the center of the screen.

**Cursor freeze prerequisite:** Double nod is **only active when the cursor is frozen** (user is looking left or right, i.e. eog_h beyond horizontal gaze thresholds). During normal head movement, gyro_x nods are ignored by the detector. This eliminates the cursor drift problem: since the cursor is already frozen before the nod starts, no gx motion is translated into cursor movement. When the cursor is not frozen, the detector's internal state is reset so stale nods are not carried over.

**Workflow:** Look left/right (cursor freezes) → nod twice (cursor centers on screen) → look back to center (cursor unfreezes after a short grace period).

This is a **gyro gesture**, not an EOG event — the ML model does not need to classify it.

| Parameter | Value | Purpose |
|-----------|-------|---------|
| `DOUBLE_NOD_THRESHOLD` | 3000 | Minimum \|gx\| for nod detection |
| `DOUBLE_NOD_MAX_DURATION` | 0.3s | Single nod must be shorter than this |
| `DOUBLE_NOD_WINDOW` | 0.8s | Two nods within this window = center cursor |
| `DOUBLE_NOD_COOLDOWN` | 1.0s | Prevent re-trigger |
