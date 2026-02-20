# State-Space Cursor Model — Derivation and Analysis

The cursor motion uses a discrete-time linear state-space model that converts raw gyroscope angular velocity into smooth, physically-intuitive cursor movement with inertia and velocity decay.

## Why State-Space?

A naive approach maps gyro directly to cursor displacement:

```
dx = sensitivity * gyro_y
```

This works but feels "robotic" — the cursor stops instantly when the head stops moving. Real physical objects have inertia: they glide and decelerate. The state-space model adds this by tracking **velocity** as internal state that decays exponentially.

## State Vector

The system tracks position and velocity independently for X and Y axes. Since the two axes are decoupled (no cross-coupling), the full 4-state system is equivalent to two identical 2-state systems:

```
x[k] = [pos_x, vel_x, pos_y, vel_y]^T
```

| State | Meaning | Unit |
|-------|---------|------|
| `pos_x` | Accumulated horizontal displacement since last frame | pixels |
| `vel_x` | Horizontal cursor velocity | pixels/step |
| `pos_y` | Accumulated vertical displacement since last frame | pixels |
| `vel_y` | Vertical cursor velocity | pixels/step |

## Input Vector

Gyroscope readings (after deadzone filtering) serve as the control input:

```
u[k] = [gy, gx]^T
```

| Input | Source | Mapping |
|-------|--------|---------|
| `gy` | Gyro Y-axis (head turn left/right) | Controls horizontal cursor (pos_x, vel_x) |
| `gx` | Gyro X-axis (head tilt up/down) | Controls vertical cursor (pos_y, vel_y) |

Note: `gy` maps to X-axis and `gx` maps to Y-axis because head turn (yaw) produces horizontal cursor motion, while head tilt (pitch) produces vertical cursor motion.

## State Equation

```
x[k+1] = A * x[k] + B * u[k]
```

### A Matrix (State Transition)

```
    ┌                          ┐
    │  1    dt     0     0     │
A = │  0    d      0     0     │
    │  0    0      1     dt    │
    │  0    0      0     d     │
    └                          ┘
```

Where:
- `dt` = sample period = 1/200 = 0.005s
- `d` = velocity retention factor (default 0.95, keeps 95% of velocity per step)

**Row-by-row interpretation:**

| Row | Equation | Meaning |
|-----|----------|---------|
| 1 | `pos_x[k+1] = pos_x[k] + dt * vel_x[k]` | Position integrates velocity (kinematics) |
| 2 | `vel_x[k+1] = d * vel_x[k]` | Velocity retains fraction `d` each step |
| 3 | `pos_y[k+1] = pos_y[k] + dt * vel_y[k]` | Same for Y axis |
| 4 | `vel_y[k+1] = d * vel_y[k]` | Same for Y axis |

### B Matrix (Input Mapping)

```
    ┌                ┐
    │  0       0     │
B = │  s       0     │
    │  0       0     │
    │  0       s     │
    └                ┘
```

Where `s` = sensitivity (default 0.05).

**Interpretation:** Gyro input directly adds to velocity (not position). This means the head motion controls **acceleration**, producing the inertia effect.

### Combined Update (Expanded)

Each time step:

```
pos_x[k+1] = pos_x[k] + dt * vel_x[k]
vel_x[k+1] = d * vel_x[k] + s * gy[k]

pos_y[k+1] = pos_y[k] + dt * vel_y[k]
vel_y[k+1] = d * vel_y[k] + s * gx[k]
```

## Position Reset Trick

After computing the state update, the controller reads `pos_x` and `pos_y` as the **displacement to apply this frame**, then resets them to zero:

```python
dx = state[0]   # pos_x = accumulated displacement
dy = state[2]   # pos_y = accumulated displacement
move_cursor(dx, dy)
state[0] = 0    # reset position accumulator
state[2] = 0    # keep velocity in state[1], state[3]
```

This avoids unbounded position growth. The position states act as **displacement accumulators** rather than absolute positions. Velocity is preserved across frames, providing the glide effect.

## Velocity Retention Behavior

The velocity retention factor `d` controls how quickly the cursor decelerates after the head stops moving. With no input (`u = 0`), velocity decays exponentially:

```
vel[k] = d^k * vel[0]
```

### Time to Stop

The velocity drops below a perceptual threshold `eps` (e.g., 0.1 px/step) after:

```
k_stop = log(eps / vel[0]) / log(d)
```

At 200 Hz, the real-time glide duration is `k_stop / 200` seconds.

### Retention Comparison

| Retain `d` | Character | Glide Duration (approx) | Use Case |
|-------------|-----------|------------------------|----------|
| 0.80 | Snappy | ~52ms | Precise targeting |
| 0.90 | Moderate | ~109ms | Balanced feel |
| 0.95 (default) | Smooth | ~225ms | General use |
| 0.99 | Floaty | ~1146ms | Large screen sweeps |

Calculated as time for velocity to decay from 1.0 to 0.1 at 200 Hz:
- `d=0.80`: `log(0.1)/log(0.80)` = 10.3 steps = 52ms
- `d=0.95`: `log(0.1)/log(0.95)` = 44.9 steps = 225ms
- `d=0.99`: `log(0.1)/log(0.99)` = 229.1 steps = 1146ms

## Deadzone Integration

Before entering the state equation, gyro values pass through a deadzone filter:

```
u_x = gy  if |gy| > GYRO_DEADZONE  else  0
u_y = gx  if |gx| > GYRO_DEADZONE  else  0
```

This prevents sensor noise from feeding into the velocity integrator. Without the deadzone, the cursor would drift randomly even when the head is stationary, because the retention factor `d < 1` only reduces velocity — it never eliminates small noise-driven velocities completely.

## Decoupled Structure

The A and B matrices are block-diagonal:

```
    ┌          ┐       ┌      ┐
    │ A2   0   │       │ B2   │
A = │          │,  B = │      │
    │ 0    A2  │       │ B2   │
    └          ┘       └      ┘
```

Where the 2x2 subsystem for each axis is:

```
       ┌         ┐          ┌   ┐
A2  =  │  1   dt │    B2 =  │ 0 │
       │  0   d  │          │ s │
       └         ┘          └   ┘
```

This means X and Y axes are completely independent — horizontal head motion has no effect on vertical cursor movement. This is physically correct and computationally efficient.

## Stability Analysis

The system eigenvalues are the diagonal elements of A: `{1, d, 1, d}`.

- The eigenvalue `1` corresponds to position integration (marginally stable). This is handled by the position reset trick — without it, position would accumulate indefinitely.
- The eigenvalue `d` (0 < d < 1) corresponds to velocity retention (asymptotically stable). Velocity always decays to zero when input stops.

The system is **BIBO stable** (bounded-input, bounded-output) because:
1. Position is reset each frame
2. Velocity eigenvalue `|d| < 1` ensures bounded velocity for bounded input

## Implementation Reference

See [`python/eog_cursor/cursor_control.py`](../python/eog_cursor/cursor_control.py), class `StateSpaceController`.
