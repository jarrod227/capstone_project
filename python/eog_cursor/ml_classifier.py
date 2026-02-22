"""
Machine learning classifier for EOG event detection.

Uses SVM (Support Vector Machine) to classify EOG signal patterns
into discrete events: idle, blink, double_blink, triple_blink,
long_blink, look_up, look_down, look_left, look_right.
"""

import logging
import os

import joblib
import numpy as np
from sklearn.preprocessing import StandardScaler
from sklearn.svm import SVC

from . import config
from .feature_extraction import extract_dual_features
from .signal_processing import SlidingWindow

logger = logging.getLogger(__name__)


class EOGClassifier:
    """
    SVM-based EOG event classifier.

    Operates on sliding windows of EOG data, extracting features
    and classifying into event types.
    """

    def __init__(self, model_path=None, scaler_path=None):
        self.model_path = model_path or config.ML_MODEL_PATH
        self.scaler_path = scaler_path or config.ML_SCALER_PATH
        self.model = None
        self.scaler = None
        self.window_v = SlidingWindow(config.ML_WINDOW_SIZE)
        self.window_h = SlidingWindow(config.ML_WINDOW_SIZE)
        self._step_counter = 0

    def load(self) -> bool:
        """
        Load trained model and scaler from disk.

        Returns:
            True if model loaded successfully, False otherwise.
        """
        if not os.path.exists(self.model_path):
            logger.warning(f"Model not found: {self.model_path}")
            return False
        if not os.path.exists(self.scaler_path):
            logger.warning(f"Scaler not found: {self.scaler_path}")
            return False

        self.model = joblib.load(self.model_path)
        self.scaler = joblib.load(self.scaler_path)
        logger.info("ML model loaded successfully.")
        return True

    def predict(self, eog_v_sample: float,
                eog_h_sample: float = None) -> str | None:
        """
        Feed one dual-channel EOG sample and get classification result.

        Returns a class label every ML_WINDOW_STEP samples after
        the window is full, otherwise None.

        Args:
            eog_v_sample: Vertical EOG value
            eog_h_sample: Horizontal EOG value (defaults to baseline)
        """
        if eog_h_sample is None:
            eog_h_sample = float(config.EOG_BASELINE)

        self.window_v.push(eog_v_sample)
        self.window_h.push(eog_h_sample)
        self._step_counter += 1

        if not self.window_v.is_full():
            return None

        if self._step_counter < config.ML_WINDOW_STEP:
            return None

        self._step_counter = 0

        # Extract dual-channel features
        features = extract_dual_features(
            self.window_v.get(), self.window_h.get()
        )
        features = features.reshape(1, -1)

        # Scale features
        if self.scaler:
            features = self.scaler.transform(features)

        # Predict
        label = self.model.predict(features)[0]
        return label


def train_model(X: np.ndarray, y: np.ndarray,
                save_dir: str = "python/models") -> tuple[SVC, StandardScaler]:
    """
    Train an SVM classifier on extracted features.

    Args:
        X: Feature matrix (n_samples, n_features)
        y: Labels array (n_samples,)
        save_dir: Directory to save model and scaler

    Returns:
        Tuple of (trained model, fitted scaler)
    """
    # Scale features
    scaler = StandardScaler()
    X_scaled = scaler.fit_transform(X)

    # Train SVM with RBF kernel
    model = SVC(
        kernel="rbf",
        C=10.0,
        gamma="scale",
        class_weight="balanced",  # Handle imbalanced classes
        probability=True           # Enable confidence scores
    )
    model.fit(X_scaled, y)

    # Save model and scaler
    os.makedirs(save_dir, exist_ok=True)
    model_path = os.path.join(save_dir, "eog_model.pkl")
    scaler_path = os.path.join(save_dir, "eog_scaler.pkl")
    joblib.dump(model, model_path)
    joblib.dump(scaler, scaler_path)

    logger.info(f"Model saved to {model_path}")
    logger.info(f"Scaler saved to {scaler_path}")

    return model, scaler
