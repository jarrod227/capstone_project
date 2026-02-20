#!/usr/bin/env python3
"""
EOG SVM Model Training Script

Loads collected CSV data, extracts features using sliding windows,
trains an SVM classifier, and evaluates with cross-validation.

Usage:
    python -m scripts.train_model --data data/raw/eog_session_*.csv
    python -m scripts.train_model --data data/raw/ --generate-demo
"""

import argparse
import glob
import os
import sys

import numpy as np
import pandas as pd
from sklearn.metrics import classification_report, confusion_matrix
from sklearn.model_selection import cross_val_score, StratifiedKFold

# Add parent directory to path
sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

from eog_cursor import config
from eog_cursor.feature_extraction import extract_dual_features, DUAL_FEATURE_NAMES
from eog_cursor.ml_classifier import train_model


def load_data(data_paths: list[str]) -> pd.DataFrame:
    """Load and concatenate multiple CSV data files."""
    frames = []
    for path in data_paths:
        if os.path.isdir(path):
            csv_files = glob.glob(os.path.join(path, "*.csv"))
            for f in csv_files:
                # Skip replay file — it's held out for demo/testing
                if os.path.basename(f) == "demo_replay.csv":
                    continue
                frames.append(pd.read_csv(f))
        else:
            frames.append(pd.read_csv(path))

    if not frames:
        raise FileNotFoundError("No CSV data files found")

    data = pd.concat(frames, ignore_index=True)
    print(f"Loaded {len(data)} samples from {len(frames)} file(s)")
    return data


def extract_windowed_features(data: pd.DataFrame,
                               window_size: int = None,
                               step_size: int = None) -> tuple[np.ndarray, np.ndarray]:
    """
    Extract features using sliding windows.

    For each window, uses the majority label as the window's label.
    """
    window_size = window_size or config.ML_WINDOW_SIZE
    step_size = step_size or config.ML_WINDOW_STEP

    # Vertical EOG channel (required)
    if 'eog_v' in data.columns:
        eog_v_values = data['eog_v'].values
    elif 'eog' in data.columns:
        eog_v_values = data['eog'].values
    else:
        raise ValueError("CSV must contain 'eog_v' or 'eog' column")

    # Horizontal EOG channel (optional, defaults to baseline)
    has_eog_h = 'eog_h' in data.columns
    if has_eog_h:
        eog_h_values = data['eog_h'].values
    else:
        eog_h_values = np.full(len(eog_v_values), config.EOG_BASELINE, dtype=float)

    labels = data['label'].values

    features_list = []
    labels_list = []

    for start in range(0, len(eog_v_values) - window_size, step_size):
        window_v = eog_v_values[start:start + window_size].astype(float)
        window_h = eog_h_values[start:start + window_size].astype(float)
        window_labels = labels[start:start + window_size]

        # Use majority label for this window
        unique, counts = np.unique(window_labels, return_counts=True)
        majority_label = unique[np.argmax(counts)]

        # Skip windows with mixed labels (transition regions)
        if np.max(counts) < window_size * 0.7:
            continue

        feats = extract_dual_features(window_v, window_h)
        features_list.append(feats)
        labels_list.append(majority_label)

    X = np.array(features_list)
    y = np.array(labels_list)
    print(f"Extracted {len(X)} feature windows ({len(DUAL_FEATURE_NAMES)} dual-channel features each)")
    return X, y


def generate_demo_data(output_dir: str):
    """
    Generate synthetic training data for demonstration.

    Reuses the time-series generator from generate_demo_data.py so that
    training data has realistic temporal patterns (blink spikes, double-blink
    twin peaks, long-blink sustained plateau, gaze transitions) matching
    what the model will see during replay or real-time inference.
    """
    from scripts.generate_demo_data import generate_session

    # Generate 3 sessions with 30 events per class for sufficient training data
    frames = []
    for i in range(3):
        df = generate_session(session_id=i, events_per_class=30)
        frames.append(df)

    combined = pd.concat(frames, ignore_index=True)
    os.makedirs(output_dir, exist_ok=True)
    output_path = os.path.join(output_dir, "demo_training_data.csv")
    combined.to_csv(output_path, index=False)
    print(f"Generated {len(combined)} demo samples -> {output_path}")
    return output_path


def main():
    parser = argparse.ArgumentParser(description="Train EOG SVM classifier")
    parser.add_argument("--data", nargs="+",
                        help="CSV file(s) or directory with training data")
    parser.add_argument("--generate-demo", action="store_true",
                        help="Generate synthetic demo data and train on it")
    _default_model_dir = os.path.join(
        os.path.dirname(os.path.dirname(os.path.abspath(__file__))), "models"
    )
    parser.add_argument("--output-dir", default=_default_model_dir,
                        help="Directory to save trained model")
    parser.add_argument("--window-size", type=int, default=config.ML_WINDOW_SIZE,
                        help=f"Window size in samples (default: {config.ML_WINDOW_SIZE})")
    parser.add_argument("--cv-folds", type=int, default=5,
                        help="Number of cross-validation folds (default: 5)")
    args = parser.parse_args()

    # Load or generate data
    if args.generate_demo:
        demo_path = generate_demo_data(config.COLLECT_OUTPUT_DIR)
        args.data = [demo_path]
    elif args.data is None:
        parser.error("Either --data or --generate-demo is required")

    data = load_data(args.data)

    # Show label distribution
    print("\nLabel distribution:")
    for label, count in data['label'].value_counts().items():
        print(f"  {label:>12}: {count}")

    # Extract features
    print(f"\nExtracting features (window={args.window_size}, "
          f"step={config.ML_WINDOW_STEP})...")
    X, y = extract_windowed_features(data, window_size=args.window_size)

    if len(X) < 10:
        print("ERROR: Not enough data windows for training.")
        print("Collect more data or reduce window size.")
        sys.exit(1)

    # Cross-validation
    print(f"\n{'='*60}")
    print(f"Cross-validation ({args.cv_folds}-fold)")
    print(f"{'='*60}")

    cv = StratifiedKFold(n_splits=args.cv_folds, shuffle=True, random_state=42)
    from sklearn.preprocessing import StandardScaler
    from sklearn.pipeline import make_pipeline
    from sklearn.svm import SVC

    pipeline = make_pipeline(
        StandardScaler(),
        SVC(kernel="rbf", C=10.0, gamma="scale", class_weight="balanced")
    )
    scores = cross_val_score(pipeline, X, y, cv=cv, scoring="accuracy")
    print(f"Accuracy: {scores.mean():.3f} (+/- {scores.std():.3f})")
    print(f"Per-fold: {[f'{s:.3f}' for s in scores]}")

    # Train final model on all data
    print(f"\n{'='*60}")
    print("Training final model on all data")
    print(f"{'='*60}")

    model, scaler = train_model(X, y, save_dir=args.output_dir)

    # Evaluate on training set (for reference)
    X_scaled = scaler.transform(X)
    y_pred = model.predict(X_scaled)
    print("\nTraining set classification report:")
    print(classification_report(y, y_pred))

    print("\nConfusion matrix:")
    labels = sorted(set(y))
    cm = confusion_matrix(y, y_pred, labels=labels)
    # Print header
    print(f"{'':>12}", end="")
    for l in labels:
        print(f"{l:>10}", end="")
    print()
    for i, l in enumerate(labels):
        print(f"{l:>12}", end="")
        for j in range(len(labels)):
            print(f"{cm[i][j]:>10}", end="")
        print()

    print(f"\nModel saved to: {args.output_dir}/eog_model.pkl")
    print(f"Scaler saved to: {args.output_dir}/eog_scaler.pkl")


if __name__ == "__main__":
    main()
