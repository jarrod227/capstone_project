#!/usr/bin/env python3
"""
Synthetic Dual-Channel EOG Dataset Generator

Generates realistic time-series dual-channel EOG+IMU data for offline
development and demo without hardware. Produces labeled CSV files that
can be:
  1. Used to train the SVM model
  2. Replayed through the full pipeline for offline demo

Two EOG channels (12-bit ADC, baseline 2048):
  eog_v (vertical):   electrodes above/below one eye
  eog_h (horizontal): electrodes at left/right outer canthi

Labels are **EOG event labels** — they describe eye movement patterns only.
The ML classifier uses only EOG features; gyro data is not used for classification.
However, ~50% of gaze events include correlated head motion to simulate realistic
fusion scenarios (eye+head triggers scroll/navigation in cursor_control.py).

Signal model (9 classes):
  idle:          eog_v ~2048 ± noise, eog_h ~2048 ± noise, gyro ≈ noise
  blink:         eog_v baseline → spike to ~3500 → back (100-200ms)
  double_blink:  two blinks spaced ~300ms apart (eog_v)
  triple_blink:  three blinks spaced ~250ms apart (eog_v)
  long_blink:    sustained high (~3500) for 400-600ms (eog_v)
  look_up:       eog_v ~2900 sustained (gyro = noise OR gx≈-800 with_head)
  look_down:     eog_v ~1000 sustained (gyro = noise OR gx≈+800 with_head)
  look_left:     eog_h ~1000 sustained (gyro = noise OR gy≈-800 with_head)
  look_right:    eog_h ~2900 sustained (gyro = noise OR gy≈+800 with_head)

Usage:
    python -m scripts.generate_demo_data
    python -m scripts.generate_demo_data --sessions 3 --output data/raw
"""

import argparse
import os
import sys

import numpy as np
import pandas as pd

sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

from eog_cursor import config

# Sampling parameters
FS = config.SAMPLE_RATE  # 200 Hz
DT = 1.0 / FS            # 5ms


def _noise(n, std=40):
    """Gaussian noise."""
    return np.random.normal(0, std, n)


def generate_idle(duration_s: float) -> tuple[np.ndarray, np.ndarray, np.ndarray, list[str]]:
    """Generate idle baseline signal."""
    n = int(duration_s * FS)
    eog_v = np.full(n, config.EOG_BASELINE, dtype=float) + _noise(n)
    eog_h = np.full(n, config.EOG_BASELINE, dtype=float) + _noise(n)
    gyro = np.column_stack([_noise(n, 80), _noise(n, 80), _noise(n, 80)])
    labels = ['idle'] * n
    return eog_v, eog_h, gyro, labels


def generate_single_blink(duration_s: float = 0.15) -> tuple[np.ndarray, np.ndarray, np.ndarray, list[str]]:
    """
    Generate one blink waveform: baseline → spike → baseline.
    Realistic blink is ~100-200ms with a sharp rise and slower fall.
    """
    n = int(duration_s * FS)
    peak = n // 3
    eog_v = np.zeros(n)

    # Rise phase
    eog_v[:peak] = np.linspace(0, 1500, peak)
    # Peak
    eog_v[peak:peak*2] = 1500
    # Fall phase
    eog_v[peak*2:] = np.linspace(1500, 0, n - peak*2)

    eog_v += config.EOG_BASELINE + _noise(n, 30)
    eog_h = np.full(n, config.EOG_BASELINE, dtype=float) + _noise(n)
    gyro = np.column_stack([_noise(n, 80), _noise(n, 80), _noise(n, 80)])
    labels = ['blink'] * n
    return eog_v, eog_h, gyro, labels


def generate_blink_event(blink_duration_s: float = 0.15,
                         context_s: float = 1.1) -> tuple[np.ndarray, np.ndarray, np.ndarray, list[str]]:
    """
    Generate one blink embedded in baseline context, all labeled 'blink'.

    A real blink lasts ~150ms (30 samples), but the training window is
    200 samples (1.0s). We pad with baseline before and after so the
    'blink' label covers enough samples for the 70% majority threshold.
    """
    n_blink = int(blink_duration_s * FS)
    n_total = int(context_s * FS)
    n_pre = (n_total - n_blink) // 2
    n_post = n_total - n_blink - n_pre

    # Blink waveform (reuse generate_single_blink for the spike shape)
    blink_ev, _, _, _ = generate_single_blink(blink_duration_s)

    # Surrounding baseline
    eog_v_pre = np.full(n_pre, config.EOG_BASELINE, dtype=float) + _noise(n_pre, 30)
    eog_v_post = np.full(n_post, config.EOG_BASELINE, dtype=float) + _noise(n_post, 30)

    eog_v = np.concatenate([eog_v_pre, blink_ev, eog_v_post])
    eog_h = np.full(n_total, config.EOG_BASELINE, dtype=float) + _noise(n_total)
    gyro = np.column_stack([_noise(n_total, 80), _noise(n_total, 80), _noise(n_total, 80)])
    labels = ['blink'] * n_total
    return eog_v, eog_h, gyro, labels


def generate_double_blink(context_s: float = 1.1) -> tuple[np.ndarray, np.ndarray, np.ndarray, list[str]]:
    """Generate double blink: two quick blinks with ~250ms gap, embedded in baseline context."""
    blink1_ev, _, _, _ = generate_single_blink(0.12)
    blink2_ev, _, _, _ = generate_single_blink(0.12)
  
    gap_n = int(0.25 * FS)
    n_event = len(blink1_ev) + gap_n + len(blink2_ev)
    n_total = int(context_s * FS)
    n_pre = (n_total - n_event) // 2
    n_post = n_total - n_event - n_pre

    gap_v = np.full(gap_n, config.EOG_BASELINE, dtype=float) + _noise(gap_n, 30)
    pre_v = np.full(n_pre, config.EOG_BASELINE, dtype=float) + _noise(n_pre, 30)
    post_v = np.full(n_post, config.EOG_BASELINE, dtype=float) + _noise(n_post, 30)

    eog_v = np.concatenate([pre_v, blink1_ev, gap_v, blink2_ev, post_v])
    eog_h = np.full(n_total, config.EOG_BASELINE, dtype=float) + _noise(n_total)
    gyro = np.column_stack([_noise(n_total, 80), _noise(n_total, 80), _noise(n_total, 80)])
    labels = ['double_blink'] * n_total
    return eog_v, eog_h, gyro, labels


def generate_triple_blink(context_s: float = 1.1) -> tuple[np.ndarray, np.ndarray, np.ndarray, list[str]]:
    """Generate triple blink: three quick blinks with ~250ms gaps, embedded in baseline context."""
    blink1_ev, _, _, _ = generate_single_blink(0.12)
    blink2_ev, _, _, _ = generate_single_blink(0.12)
    blink3_ev, _, _, _ = generate_single_blink(0.12)

    gap_n = int(0.25 * FS)
    n_event = len(blink1_ev) + gap_n + len(blink2_ev) + gap_n + len(blink3_ev)
    n_total = int(context_s * FS)
    n_pre = (n_total - n_event) // 2
    n_post = n_total - n_event - n_pre

    gap1_v = np.full(gap_n, config.EOG_BASELINE, dtype=float) + _noise(gap_n, 30)
    gap2_v = np.full(gap_n, config.EOG_BASELINE, dtype=float) + _noise(gap_n, 30)
    pre_v = np.full(n_pre, config.EOG_BASELINE, dtype=float) + _noise(n_pre, 30)
    post_v = np.full(n_post, config.EOG_BASELINE, dtype=float) + _noise(n_post, 30)

    eog_v = np.concatenate([pre_v, blink1_ev, gap1_v, blink2_ev, gap2_v, blink3_ev, post_v])
    eog_h = np.full(n_total, config.EOG_BASELINE, dtype=float) + _noise(n_total)
    gyro = np.column_stack([_noise(n_total, 80), _noise(n_total, 80), _noise(n_total, 80)])
    labels = ['triple_blink'] * n_total
    return eog_v, eog_h, gyro, labels


def generate_long_blink(duration_s: float = 0.8) -> tuple[np.ndarray, np.ndarray, np.ndarray, list[str]]:
    """Generate long blink: sustained high for >0.4s."""
    n = int(duration_s * FS)
    rise = int(0.05 * FS)
    fall = int(0.05 * FS)
    sustain = n - rise - fall

    eog_v = np.zeros(n)
    eog_v[:rise] = np.linspace(0, 1500, rise)
    eog_v[rise:rise + sustain] = 1500
    eog_v[rise + sustain:] = np.linspace(1500, 0, fall)

    eog_v += config.EOG_BASELINE + _noise(n, 30)
    eog_h = np.full(n, config.EOG_BASELINE, dtype=float) + _noise(n)
    gyro = np.column_stack([_noise(n, 80), _noise(n, 80), _noise(n, 80)])
    labels = ['long_blink'] * n
    return eog_v, eog_h, gyro, labels


def generate_look_up(duration_s: float = 0.8,
                     with_head: bool = False) -> tuple[np.ndarray, np.ndarray, np.ndarray, list[str]]:
    """Generate look-up: eog_v rises to ~2900. Optionally includes head tilt up."""
    n = int(duration_s * FS)
    transition = int(0.1 * FS)
    sustain = n - 2 * transition

    # EOG vertical: gradual rise to ~2900 (shift of +850 above baseline)
    eog_v = np.zeros(n)
    eog_v[:transition] = np.linspace(0, 850, transition)
    eog_v[transition:transition + sustain] = 850
    eog_v[transition + sustain:] = np.linspace(850, 0, transition)
    eog_v += config.EOG_BASELINE + _noise(n, 30)

    # EOG horizontal: neutral
    eog_h = np.full(n, config.EOG_BASELINE, dtype=float) + _noise(n)

    # IMU: head tilts up (gx < 0) when with_head, otherwise noise only
    gx = np.full(n, -800.0) + _noise(n, 100) if with_head else _noise(n, 80)
    gyro = np.column_stack([gx, _noise(n, 80), _noise(n, 80)])

    labels = ['look_up'] * n
    return eog_v, eog_h, gyro, labels


def generate_look_down(duration_s: float = 0.8,
                       with_head: bool = False) -> tuple[np.ndarray, np.ndarray, np.ndarray, list[str]]:
    """Generate look-down: eog_v drops to ~1000. Optionally includes head tilt down."""
    n = int(duration_s * FS)
    transition = int(0.1 * FS)
    sustain = n - 2 * transition

    # EOG vertical: gradual drop to ~1000 (shift of -1050 below baseline)
    eog_v = np.zeros(n)
    eog_v[:transition] = np.linspace(0, -1050, transition)
    eog_v[transition:transition + sustain] = -1050
    eog_v[transition + sustain:] = np.linspace(-1050, 0, transition)
    eog_v += config.EOG_BASELINE + _noise(n, 30)

    # EOG horizontal: neutral
    eog_h = np.full(n, config.EOG_BASELINE, dtype=float) + _noise(n)

    # IMU: head tilts down (gx > 0) when with_head, otherwise noise only
    gx = np.full(n, 800.0) + _noise(n, 100) if with_head else _noise(n, 80)
    gyro = np.column_stack([gx, _noise(n, 80), _noise(n, 80)])

    labels = ['look_down'] * n
    return eog_v, eog_h, gyro, labels


def generate_head_roll() -> tuple[np.ndarray, np.ndarray, np.ndarray, list[str]]:
    """Generate head roll flick: gz spike for ~0.3s."""
    n = int(0.3 * FS)
    eog_v = np.full(n, config.EOG_BASELINE, dtype=float) + _noise(n, 40)
    eog_h = np.full(n, config.EOG_BASELINE, dtype=float) + _noise(n)

    gx = _noise(n, 80)
    gy = _noise(n, 80)
    # gz spike
    gz = np.zeros(n)
    peak = n // 2
    gz[:peak] = np.linspace(0, 4000, peak)
    gz[peak:] = np.linspace(4000, 0, n - peak)
    gz += _noise(n, 150)
    gyro = np.column_stack([gx, gy, gz])

    labels = ['idle'] * n  # Head roll is a gyro gesture, not an EOG event
    return eog_v, eog_h, gyro, labels


def generate_double_nod() -> tuple[np.ndarray, np.ndarray, np.ndarray, list[str]]:
    """Generate double head nod: two gx spikes ~0.2s apart."""
    nod_n = int(0.15 * FS)
    gap_n = int(0.2 * FS)

    # First nod: gx spike
    gx1 = np.zeros(nod_n)
    peak = nod_n // 2
    gx1[:peak] = np.linspace(0, 4000, peak)
    gx1[peak:] = np.linspace(4000, 0, nod_n - peak)
    gx1 += _noise(nod_n, 150)

    # Gap
    gx_gap = _noise(gap_n, 80)

    # Second nod
    gx2 = np.zeros(nod_n)
    gx2[:peak] = np.linspace(0, 4000, peak)
    gx2[peak:] = np.linspace(4000, 0, nod_n - peak)
    gx2 += _noise(nod_n, 150)

    n = 2 * nod_n + gap_n
    eog_v = np.full(n, config.EOG_BASELINE, dtype=float) + _noise(n, 40)
    eog_h = np.full(n, config.EOG_BASELINE, dtype=float) + _noise(n)
    gx = np.concatenate([gx1, gx_gap, gx2])
    gyro = np.column_stack([gx, _noise(n, 80), _noise(n, 80)])

    labels = ['idle'] * n  # Double nod is a gyro gesture, not an EOG event
    return eog_v, eog_h, gyro, labels


def generate_cursor_move(direction: str = 'right',
                          duration_s: float = 1.0) -> tuple[np.ndarray, np.ndarray, np.ndarray, list[str]]:
    """Generate cursor movement: sustained gyro in one direction."""
    n = int(duration_s * FS)
    eog_v = np.full(n, config.EOG_BASELINE, dtype=float) + _noise(n, 40)
    eog_h = np.full(n, config.EOG_BASELINE, dtype=float) + _noise(n)

    magnitude = 1500
    gx = _noise(n, 80)
    gy = _noise(n, 80)
    gz = _noise(n, 80)

    if direction == 'right':
        gy = np.full(n, magnitude) + _noise(n, 150)
    elif direction == 'left':
        gy = np.full(n, -magnitude) + _noise(n, 150)
    elif direction == 'up':
        gx = np.full(n, -magnitude) + _noise(n, 150)
    elif direction == 'down':
        gx = np.full(n, magnitude) + _noise(n, 150)

    gyro = np.column_stack([gx, gy, gz])
    labels = ['idle'] * n  # Cursor move is continuous, not an EOG event
    return eog_v, eog_h, gyro, labels


def generate_look_left(duration_s: float = 0.8,
                       with_head: bool = False) -> tuple[np.ndarray, np.ndarray, np.ndarray, list[str]]:
    """Generate look-left: eog_h drops to ~1000. Optionally includes head turn left."""
    n = int(duration_s * FS)
    transition = int(0.1 * FS)
    sustain = n - 2 * transition

    # EOG vertical: neutral
    eog_v = np.full(n, config.EOG_BASELINE, dtype=float) + _noise(n)

    # EOG horizontal: gradual drop to ~1000 (shift of -1050 below baseline)
    eog_h = np.zeros(n)
    eog_h[:transition] = np.linspace(0, -1050, transition)
    eog_h[transition:transition + sustain] = -1050
    eog_h[transition + sustain:] = np.linspace(-1050, 0, transition)
    eog_h += config.EOG_BASELINE + _noise(n, 30)

    # IMU: head turns left (gy < 0) when with_head, otherwise noise only
    gy = np.full(n, -800.0) + _noise(n, 100) if with_head else _noise(n, 80)
    gyro = np.column_stack([_noise(n, 80), gy, _noise(n, 80)])

    labels = ['look_left'] * n
    return eog_v, eog_h, gyro, labels


def generate_look_right(duration_s: float = 0.8,
                        with_head: bool = False) -> tuple[np.ndarray, np.ndarray, np.ndarray, list[str]]:
    """Generate look-right: eog_h rises to ~2900. Optionally includes head turn right."""
    n = int(duration_s * FS)
    transition = int(0.1 * FS)
    sustain = n - 2 * transition

    # EOG vertical: neutral
    eog_v = np.full(n, config.EOG_BASELINE, dtype=float) + _noise(n)

    # EOG horizontal: gradual rise to ~2900 (shift of +850 above baseline)
    eog_h = np.zeros(n)
    eog_h[:transition] = np.linspace(0, 850, transition)
    eog_h[transition:transition + sustain] = 850
    eog_h[transition + sustain:] = np.linspace(850, 0, transition)
    eog_h += config.EOG_BASELINE + _noise(n, 30)

    # IMU: head turns right (gy > 0) when with_head, otherwise noise only
    gy = np.full(n, 800.0) + _noise(n, 100) if with_head else _noise(n, 80)
    gyro = np.column_stack([_noise(n, 80), gy, _noise(n, 80)])

    labels = ['look_right'] * n
    return eog_v, eog_h, gyro, labels


def generate_session(session_id: int = 0,
                     events_per_class: int = 30) -> pd.DataFrame:
    """
    Generate one complete recording session with labeled events.

    Structure: idle gaps between events, randomized event order.
    """
    rng = np.random.default_rng(42 + session_id)

    all_eog_v = []
    all_eog_h = []
    all_gyro = []
    all_labels = []

    # Start with idle
    ev, eh, g, l = generate_idle(2.0)
    all_eog_v.append(ev); all_eog_h.append(eh); all_gyro.append(g); all_labels.extend(l)

    # Event generators (gaze events accept with_head parameter)
    event_generators = {
        'blink': generate_blink_event,
        'double_blink': generate_double_blink,
        'triple_blink': generate_triple_blink,
        'long_blink': generate_long_blink,
        'look_up': generate_look_up,
        'look_down': generate_look_down,
        'look_left': generate_look_left,
        'look_right': generate_look_right,
    }
    # Gaze events that support with_head (eye + head = triggers fusion action)
    gaze_events = {'look_up', 'look_down', 'look_left', 'look_right'}

    # Build event schedule
    events = []
    for event_name in event_generators:
        events.extend([event_name] * events_per_class)
    rng.shuffle(events)

    # Also add some cursor movements and head rolls interspersed
    for i, event_name in enumerate(events):
        # Generate the event — ~50% of gaze events include head motion
        gen_fn = event_generators[event_name]
        if event_name in gaze_events:
            with_head = bool(rng.random() < 0.5)
            ev, eh, g, l = gen_fn(with_head=with_head)
        else:
            ev, eh, g, l = gen_fn()
        all_eog_v.append(ev); all_eog_h.append(eh); all_gyro.append(g); all_labels.extend(l)

        # Idle gap (0.8-2.0s)
        gap = 0.8 + rng.random() * 1.2
        ev, eh, g, l = generate_idle(gap)
        all_eog_v.append(ev); all_eog_h.append(eh); all_gyro.append(g); all_labels.extend(l)

        # Occasionally add cursor movement or head roll
        if i % 10 == 5:
            direction = rng.choice(['right', 'left', 'up', 'down'])
            ev, eh, g, l = generate_cursor_move(direction, 0.5)
            all_eog_v.append(ev); all_eog_h.append(eh); all_gyro.append(g); all_labels.extend(l)

            ev, eh, g, l = generate_idle(0.5)
            all_eog_v.append(ev); all_eog_h.append(eh); all_gyro.append(g); all_labels.extend(l)

        if i % 15 == 10:
            ev, eh, g, l = generate_head_roll()
            all_eog_v.append(ev); all_eog_h.append(eh); all_gyro.append(g); all_labels.extend(l)

            ev, eh, g, l = generate_idle(0.5)
            all_eog_v.append(ev); all_eog_h.append(eh); all_gyro.append(g); all_labels.extend(l)

        if i % 12 == 7:
            ev, eh, g, l = generate_double_nod()
            all_eog_v.append(ev); all_eog_h.append(eh); all_gyro.append(g); all_labels.extend(l)

            ev, eh, g, l = generate_idle(0.5)
            all_eog_v.append(ev); all_eog_h.append(eh); all_gyro.append(g); all_labels.extend(l)

    # End with idle
    ev, eh, g, l = generate_idle(2.0)
    all_eog_v.append(ev); all_eog_h.append(eh); all_gyro.append(g); all_labels.extend(l)

    # Combine
    eog_v = np.concatenate(all_eog_v)
    eog_h = np.concatenate(all_eog_h)
    gyro = np.vstack(all_gyro)

    # Clip EOG to 12-bit range
    eog_v = np.clip(eog_v, 0, 4095).astype(int)
    eog_h = np.clip(eog_h, 0, 4095).astype(int)
    gyro = gyro.astype(int)

    # Build timestamps
    n = len(eog_v)
    timestamps = np.arange(n) * 5  # 5ms per sample

    df = pd.DataFrame({
        'timestamp': timestamps,
        'eog_v': eog_v,
        'eog_h': eog_h,
        'gyro_x': gyro[:, 0],
        'gyro_y': gyro[:, 1],
        'gyro_z': gyro[:, 2],
        'label': all_labels,
    })

    return df


def main():
    parser = argparse.ArgumentParser(
        description="Generate synthetic EOG demo data"
    )
    parser.add_argument("--sessions", type=int, default=3,
                        help="Number of sessions to generate (default: 3)")
    parser.add_argument("--events-per-class", type=int, default=30,
                        help="Events per class per session (default: 30)")
    parser.add_argument("--output", default=config.COLLECT_OUTPUT_DIR,
                        help=f"Output directory (default: {config.COLLECT_OUTPUT_DIR})")
    parser.add_argument("--seed", type=int, default=42,
                        help="Random seed (default: 42)")
    args = parser.parse_args()

    np.random.seed(args.seed)
    os.makedirs(args.output, exist_ok=True)

    print("=" * 60)
    print("  Synthetic EOG Data Generator")
    print("=" * 60)
    print(f"  Sessions:         {args.sessions}")
    print(f"  Events per class: {args.events_per_class}")
    print(f"  Output:           {args.output}")
    print(f"  EOG baseline:     {config.EOG_BASELINE} (12-bit ADC)")
    print()

    all_files = []
    total_samples = 0

    for i in range(args.sessions):
        df = generate_session(session_id=i, events_per_class=args.events_per_class)
        filename = f"demo_session_{i:02d}.csv"
        filepath = os.path.join(args.output, filename)
        df.to_csv(filepath, index=False)

        duration = len(df) / FS
        total_samples += len(df)
        all_files.append(filepath)

        print(f"  Session {i}: {len(df):>6} samples ({duration:.1f}s) -> {filename}")

        # Print label distribution for this session
        counts = df['label'].value_counts()
        for label in sorted(counts.index):
            count = counts[label]
            pct = count / len(df) * 100
            print(f"    {label:>15}: {count:>5} ({pct:.1f}%)")

    total_duration = total_samples / FS
    print(f"\n  Total: {total_samples} samples ({total_duration:.1f}s) across {args.sessions} sessions")
    print(f"  Files saved to: {args.output}/")

    # Also generate one unlabeled replay file (no 'label' column)
    # for testing the real-time pipeline
    replay_df = generate_session(session_id=99, events_per_class=15)
    replay_path = os.path.join(args.output, "demo_replay.csv")
    replay_df.to_csv(replay_path, index=False)
    print(f"\n  Replay file (with labels for reference): {replay_path}")
    print(f"  ({len(replay_df)} samples, {len(replay_df)/FS:.1f}s)")

    print("\nDone! Next steps:")
    print(f"  1. Train model:  cd python && python -m scripts.train_model --data ../{args.output}")
    print(f"  2. Offline demo: cd python && python main.py --replay ../{replay_path}")
    print(f"  3. Live demo:    cd python && python main.py --simulate")

    return all_files


if __name__ == "__main__":
    main()
