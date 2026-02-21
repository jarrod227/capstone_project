"""
Feature extraction for EOG signal classification.

Extracts time-domain and statistical features from windowed EOG data
for use with SVM classifier.

Supports dual-channel EOG (eog_v + eog_h) via extract_dual_features().
"""

import numpy as np


def extract_features(window: np.ndarray) -> np.ndarray:
    """
    Extract classification features from an EOG signal window.

    Args:
        window: 1D array of EOG samples (typically 100 samples = 0.5s)

    Returns:
        Feature vector (1D numpy array)
    """
    features = []

    # Cache common statistics (avoid redundant computation)
    mean = np.mean(window)
    std = np.std(window)
    centered = window - mean
    derivative = np.diff(window)

    # --- Time-domain features ---

    # Peak-to-peak amplitude
    peak_amplitude = np.max(window) - np.min(window)
    features.append(peak_amplitude)

    # Zero-crossing rate (after mean subtraction)
    zero_crossings = np.sum(np.diff(np.sign(centered)) != 0)
    features.append(zero_crossings)

    # Linear slope (trend direction)
    x = np.arange(len(window))
    slope = np.polyfit(x, window, 1)[0]
    features.append(slope)

    # Maximum absolute derivative (speed of change)
    max_derivative = np.max(np.abs(derivative)) if len(derivative) > 0 else 0
    features.append(max_derivative)

    # --- Statistical features ---

    # Mean
    features.append(mean)

    # Standard deviation
    features.append(std)

    # Skewness (asymmetry)
    if std > 0:
        skewness = np.mean((centered / std) ** 3)
    else:
        skewness = 0.0
    features.append(skewness)

    # Kurtosis (peakedness)
    if std > 0:
        kurtosis = np.mean((centered / std) ** 4) - 3
    else:
        kurtosis = 0.0
    features.append(kurtosis)

    # --- Energy features ---

    # Root mean square
    rms = np.sqrt(np.mean(window ** 2))
    features.append(rms)

    # Variance of derivative (signal "roughness")
    if len(derivative) > 0:
        deriv_var = np.var(derivative)
    else:
        deriv_var = 0.0
    features.append(deriv_var)

    return np.array(features)


# Feature names for reference and model interpretation
FEATURE_NAMES = [
    "peak_amplitude",
    "zero_crossings",
    "slope",
    "max_derivative",
    "mean",
    "std",
    "skewness",
    "kurtosis",
    "rms",
    "derivative_variance",
]


def extract_dual_features(eog_v_window: np.ndarray,
                           eog_h_window: np.ndarray) -> np.ndarray:
    """
    Extract features from both vertical and horizontal EOG channels.

    Concatenates the 10 features from each channel into a 20-feature vector.
    This allows the classifier to distinguish horizontal gaze events
    (look_left, look_right) which only appear in the eog_h channel.

    Args:
        eog_v_window: 1D array of vertical EOG samples
        eog_h_window: 1D array of horizontal EOG samples

    Returns:
        Feature vector of length 20 (10 per channel)
    """
    feats_v = extract_features(eog_v_window)
    feats_h = extract_features(eog_h_window)
    return np.concatenate([feats_v, feats_h])


DUAL_FEATURE_NAMES = (
    [f"v_{name}" for name in FEATURE_NAMES] +
    [f"h_{name}" for name in FEATURE_NAMES]
)
