# Gyroscope Kalman Filter — Derivation and Analysis

A 2-state Kalman filter runs independently on each gyroscope axis to separate true angular velocity from slowly-drifting sensor bias, enabling continuous bias correction without requiring an accelerometer or explicit stillness detection.

## The Problem

MEMS gyroscopes output a reading that is the sum of three components:

```
z[k] = ω[k] + b[k] + v[k]
```

| Symbol | Meaning | Character |
|--------|---------|-----------|
| `z[k]` | Raw gyro reading (what we measure) | Observable |
| `ω[k]` | True angular velocity (what we want) | Unobservable |
| `b[k]` | Sensor bias (slowly drifting offset) | Unobservable |
| `v[k]` | White measurement noise | Unobservable |

The fundamental challenge: we have **one measurement** (`z`) but **two unknowns** (`ω` and `b`). Simple subtraction of a fixed calibration bias fails because `b` drifts over time due to temperature changes and sensor aging.

## Why Not Just Re-Calibrate?

Startup calibration (averaging readings while stationary) gives a good initial bias estimate, but:

1. **Bias drifts** — temperature changes shift the offset during a session
2. **Re-calibration requires stillness** — cannot interrupt the user mid-session
3. **Drift is gradual** — a filter can track it continuously if modeled correctly

The Kalman filter exploits the key asymmetry: **bias changes slowly, angular velocity changes quickly**. This difference in timescale makes the two states separable from a single measurement.

## State-Space Model

### State Vector

```
        ┌      ┐
x[k] =  │ ω[k] │
        │ b[k] │
        └      ┘
```

### Process Model (Prediction)

The state evolves as:

```
x[k+1] = F · x[k] + w[k]
```

where `w[k] ~ N(0, Q)` is process noise.

#### F Matrix (State Transition)

```
    ┌        ┐
F = │  0   0 │
    │  0   1 │
    └        ┘
```

**Row-by-row interpretation:**

| Row | Equation | Meaning |
|-----|----------|---------|
| 1 | `ω[k+1] = 0` | Angular velocity has **no temporal persistence** — the predicted value before incorporating the measurement is zero. This encodes the prior assumption that the head is at rest. |
| 2 | `b[k+1] = b[k]` | Bias **persists** — the best prediction of next bias is the current bias (random walk model). |

**Why F₁₁ = 0 (not 1)?** Setting `F₁₁ = 0` means the filter predicts `ω = 0` at each step before seeing the measurement. This is correct because:

- We have no dynamic model for head motion (it's driven by human intent, not physics)
- The measurement will override this prediction — the Kalman gain for `ω` will be high because `Q_ω` is large (see below)
- If we set `F₁₁ = 1`, the filter would predict that the current angular velocity continues, which would delay the response to direction changes and blur the separation between `ω` and `b`

#### Q Matrix (Process Noise Covariance)

```
    ┌              ┐
Q = │  Q_ω    0    │
    │  0      Q_b  │
    └              ┘
```

Default values: `Q_ω = 1000`, `Q_b = 0.001`.

**Physical interpretation:**

- `Q_ω = 1000` (large): Angular velocity is **highly uncertain** in prediction. The filter expects `ω` to jump unpredictably between steps, so it trusts the measurement heavily for estimating `ω`.
- `Q_b = 0.001` (small): Bias changes **very slowly**. The filter resists changing its bias estimate unless sustained evidence accumulates over many samples.

The ratio `Q_ω / Q_b = 10⁶` is the core design parameter — it determines how aggressively the filter separates fast signals (attributed to `ω`) from slow signals (attributed to `b`).

### Measurement Model

```
z[k] = H · x[k] + v[k]
```

where `v[k] ~ N(0, R)` is measurement noise.

#### H Matrix (Measurement)

```
H = [ 1   1 ]
```

This encodes `z = ω + b`: the gyro measures the **sum** of angular velocity and bias.

#### R Matrix (Measurement Noise Covariance)

```
R = [ 500 ]
```

This is the variance of the sensor noise `v[k]`. It should approximate the actual noise power of the gyroscope. Higher `R` makes the filter trust measurements less, producing smoother but more delayed output.

## Kalman Filter Equations

Each time step runs predict → update:

### Predict

```
x̂⁻[k] = F · x̂[k-1]         (predicted state)
P⁻[k]  = F · P[k-1] · Fᵀ + Q  (predicted covariance)
```

### Update

```
ỹ[k]   = z[k] - H · x̂⁻[k]              (innovation)
S[k]   = H · P⁻[k] · Hᵀ + R            (innovation covariance)
K[k]   = P⁻[k] · Hᵀ · S[k]⁻¹           (Kalman gain)
x̂[k]  = x̂⁻[k] + K[k] · ỹ[k]           (updated state)
P[k]   = (I - K[k] · H) · P⁻[k]        (updated covariance)
```

The filter output is `x̂[k][0]` = estimated true angular velocity `ω̂[k]`.

## Steady-State Analysis

After several iterations, the covariance `P` and Kalman gain `K` converge to fixed values. We can derive the steady-state analytically.

### Predicted Covariance

Substituting our specific F and Q:

```
P⁻ = F · P · Fᵀ + Q
```

Let the steady-state updated covariance be:

```
        ┌            ┐
P∞   =  │ p₁₁  p₁₂  │
        │ p₁₂  p₂₂  │
        └            ┘
```

Then the predicted covariance is:

```
              ┌ 0  0 ┐   ┌ p₁₁  p₁₂ ┐   ┌ 0  0 ┐   ┌ Q_ω    0   ┐
P⁻ = F·P·Fᵀ + Q  =  │     │ · │          │ · │     │ + │            │
              └ 0  1 ┘   └ p₁₂  p₂₂ ┘   └ 0  1 ┘   └ 0      Q_b  ┘

     ┌ Q_ω          0        ┐
   = │                        │
     │ 0        p₂₂ + Q_b    │
     └                        ┘
```

Key observation: **the predicted uncertainty of `ω` is always `Q_ω` regardless of previous state**, because `F₁₁ = 0` erases all memory of `ω`. The bias uncertainty accumulates: `p₂₂ + Q_b`.

### Innovation Covariance

```
S = H · P⁻ · Hᵀ + R = Q_ω + (p₂₂ + Q_b) + R
```

### Kalman Gain

```
         P⁻ · Hᵀ     1     ┌   Q_ω        ┐
K  =  ─────────── = ───── · │               │
            S         S     └  p₂₂ + Q_b    ┘
```

The gain for `ω` is:

```
K_ω = Q_ω / S = Q_ω / (Q_ω + p₂₂ + Q_b + R)
```

The gain for `b` is:

```
K_b = (p₂₂ + Q_b) / S
```

### Steady-State Gain (Approximate)

With default values `Q_ω = 1000`, `Q_b = 0.001`, `R = 500`, and noting that `p₂₂` converges to a small value:

```
S ≈ Q_ω + R = 1000 + 500 = 1500

K_ω ≈ 1000 / 1500 ≈ 0.667
K_b ≈ (p₂₂ + 0.001) / 1500 ≈ very small
```

**Interpretation:** Each measurement contributes ~67% to the `ω` estimate (high responsiveness) but almost nothing to the `b` estimate (slow adaptation). This is exactly the desired behavior — fast angular velocity tracking, slow bias tracking.

### Steady-State Bias Covariance

The bias covariance converges by solving the Riccati equation. For the bias component:

```
p₂₂[k+1] = (p₂₂[k] + Q_b) · (1 - K_b)
          = (p₂₂[k] + Q_b) · (1 - (p₂₂[k] + Q_b) / S)
```

At steady state `p₂₂[k+1] = p₂₂[k] = p∞`:

```
p∞ = (p∞ + Q_b) · (1 - (p∞ + Q_b) / S)
```

Let `a = p∞ + Q_b`. Then:

```
a - Q_b = a · (1 - a/S)
a - Q_b = a - a²/S
Q_b = a²/S
a = √(Q_b · S) ≈ √(0.001 · 1500) ≈ 1.22
p∞ = a - Q_b ≈ 1.22
```

So the steady-state bias uncertainty is ~1.22 (in raw gyro units²). The steady-state bias Kalman gain is:

```
K_b∞ = a / S ≈ 1.22 / 1500 ≈ 0.00081
```

This means the bias estimate moves by only 0.08% of each innovation — equivalent to an exponential moving average with a time constant of ~1/0.00081 ≈ 1230 samples = **6.15 seconds** at 200 Hz. This matches intuition: bias drift should be tracked on a timescale of seconds, not milliseconds.

## Parameter Sensitivity

### Effect of Q_ω / Q_b Ratio

| Q_ω | Q_b | Ratio | Bias Time Constant | ω Responsiveness | Character |
|-----|-----|-------|-------------------|-----------------|-----------|
| 100 | 0.001 | 10⁵ | ~3.9s | K_ω ≈ 0.17 | Sluggish ω, moderate bias tracking |
| 1000 | 0.001 | 10⁶ | ~6.2s | K_ω ≈ 0.67 | **Default — good balance** |
| 10000 | 0.001 | 10⁷ | ~10.9s | K_ω ≈ 0.95 | Very responsive ω, very slow bias |
| 1000 | 0.01 | 10⁵ | ~2.0s | K_ω ≈ 0.66 | Faster bias adaptation |
| 1000 | 0.1 | 10⁴ | ~0.7s | K_ω ≈ 0.63 | Aggressive bias tracking (may eat real motion) |

**The danger of high Q_b:** If bias adapts too quickly, sustained head motion (e.g., holding a turn) gets partially absorbed into the bias estimate, attenuating the real signal. The default `Q_b = 0.001` keeps the bias time constant at ~6s, safely above typical head motion durations.

### Effect of R (Measurement Noise)

| R | K_ω (approx) | Effect |
|---|--------------|--------|
| 100 | 0.91 | Noisy output, very responsive |
| 500 | 0.67 | **Default — balanced** |
| 2000 | 0.33 | Smooth but delayed |

Higher `R` smooths the output at the cost of latency.

## Startup Initialization

### From Calibration

The startup calibration provides a bias estimate `b₀` by averaging 400 stationary samples. We seed the filter:

```
x₀ = [0, b₀]ᵀ

        ┌ 1000    0  ┐
P₀   =  │            │
        └  0     100  ┘
```

- `P₀[0,0] = 1000`: No information about initial angular velocity
- `P₀[1,1] = 100`: Moderate confidence in calibrated bias (not the default 1000)

This gives the filter a head start. Without it, the bias would start at 0 and take ~6s to converge, causing cursor drift during startup.

### Without Calibration

If no calibration is available, the filter starts with:

```
x₀ = [0, 0]ᵀ,  P₀ = diag(1000, 1000)
```

The filter will converge on the true bias, but the first ~6s will show noticeable drift.

## Comparison with Static Calibration

| Property | Static Calibration | Kalman Filter |
|----------|-------------------|---------------|
| Bias model | Fixed at startup | Tracked continuously |
| Handles drift | No | Yes |
| Requires stillness | Yes (at startup) | No (after initialization) |
| Latency | None (subtraction) | Minimal (K_ω ≈ 0.67) |
| Computational cost | Negligible | 2×2 matrix ops per axis per sample |

In our system, both are used together: static calibration provides the initial estimate, the Kalman filter tracks drift from there.

## Implementation Reference

See [`python/eog_cursor/signal_processing.py`](../python/eog_cursor/signal_processing.py), classes `GyroKalmanFilter` (single axis) and `GyroKalmanFilter3Axis` (convenience wrapper). Parameters in [`python/eog_cursor/config.py`](../python/eog_cursor/config.py).
