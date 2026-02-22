"""
Tests for ML training and inference pipeline.

Tests feature extraction, model training, saving/loading,
and prediction using synthetic data.
"""

import os
import sys
import tempfile
import unittest

import numpy as np

sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

from eog_cursor.feature_extraction import (
    extract_features, extract_dual_features,
    FEATURE_NAMES, DUAL_FEATURE_NAMES,
)
from eog_cursor.ml_classifier import EOGClassifier, train_model
from eog_cursor import config


def generate_synthetic_windows(n_per_class=50, window_size=200):
    """Generate synthetic dual-channel EOG windows for each class."""
    rng = np.random.default_rng(42)
    X_list = []
    y_list = []

    baseline_h = lambda: rng.normal(config.EOG_BASELINE, 50, window_size)

    patterns = {
        "idle": (
            lambda: rng.normal(1500, 50, window_size),
            baseline_h,
        ),
        "blink": (
            lambda: np.concatenate([
                rng.normal(1500, 30, 80),
                rng.normal(3500, 200, 30),
                rng.normal(1500, 30, 90)
            ]),
            baseline_h,
        ),
        "double_blink": (
            lambda: np.concatenate([
                rng.normal(1500, 30, 40),
                rng.normal(3500, 200, 25),
                rng.normal(1500, 30, 40),
                rng.normal(3400, 200, 25),
                rng.normal(1500, 30, 70)
            ]),
            baseline_h,
        ),
        "triple_blink": (
            lambda: np.concatenate([
                rng.normal(1500, 30, 20),
                rng.normal(3500, 200, 24),
                rng.normal(1500, 30, 30),
                rng.normal(3400, 200, 24),
                rng.normal(1500, 30, 30),
                rng.normal(3400, 200, 24),
                rng.normal(1500, 30, 48)
            ]),
            baseline_h,
        ),
        "long_blink": (
            lambda: np.concatenate([
                rng.normal(1500, 30, 40),
                rng.normal(3300, 150, 120),
                rng.normal(1500, 30, 40)
            ]),
            baseline_h,
        ),
        "look_up": (
            lambda: np.linspace(1500, 2800, window_size) + rng.normal(0, 30, window_size),
            baseline_h,
        ),
        "look_down": (
            lambda: np.linspace(1500, 700, window_size) + rng.normal(0, 30, window_size),
            baseline_h,
        ),
        "look_left": (
            lambda: rng.normal(config.EOG_BASELINE, 50, window_size),
            lambda: np.linspace(config.EOG_BASELINE, 900, window_size) + rng.normal(0, 30, window_size),
        ),
        "look_right": (
            lambda: rng.normal(config.EOG_BASELINE, 50, window_size),
            lambda: np.linspace(config.EOG_BASELINE, 2900, window_size) + rng.normal(0, 30, window_size),
        ),
    }

    for label, (gen_v, gen_h) in patterns.items():
        for _ in range(n_per_class):
            window_v = gen_v()
            window_h = gen_h()
            features = extract_dual_features(window_v, window_h)
            X_list.append(features)
            y_list.append(label)

    return np.array(X_list), np.array(y_list)


class TestMLPipeline(unittest.TestCase):
    """Test the full ML pipeline: features -> train -> save -> load -> predict."""

    @classmethod
    def setUpClass(cls):
        """Generate test data and train model once for all tests."""
        cls.X, cls.y = generate_synthetic_windows(n_per_class=50)
        cls.tmpdir = tempfile.mkdtemp()
        cls.model, cls.scaler = train_model(cls.X, cls.y, save_dir=cls.tmpdir)

    def test_feature_dimensions(self):
        """Feature matrix should have correct shape (20 dual-channel features)."""
        self.assertEqual(self.X.shape[1], len(DUAL_FEATURE_NAMES))

    def test_model_trains(self):
        """Model should train without errors."""
        self.assertIsNotNone(self.model)
        self.assertIsNotNone(self.scaler)

    def test_model_accuracy(self):
        """Trained model should achieve reasonable accuracy on training data."""
        X_scaled = self.scaler.transform(self.X)
        accuracy = self.model.score(X_scaled, self.y)
        # Synthetic data should be easily separable
        self.assertGreater(accuracy, 0.8)

    def test_model_save_load(self):
        """Model should be loadable from disk."""
        model_path = os.path.join(self.tmpdir, "eog_model.pkl")
        scaler_path = os.path.join(self.tmpdir, "eog_scaler.pkl")
        self.assertTrue(os.path.exists(model_path))
        self.assertTrue(os.path.exists(scaler_path))

        classifier = EOGClassifier(
            model_path=model_path,
            scaler_path=scaler_path
        )
        self.assertTrue(classifier.load())

    def test_prediction_output(self):
        """Predictions should be valid class labels."""
        X_scaled = self.scaler.transform(self.X)
        predictions = self.model.predict(X_scaled)
        valid_labels = {"idle", "blink", "double_blink", "triple_blink",
                        "long_blink", "look_up", "look_down", "look_left",
                        "look_right"}
        for pred in predictions:
            self.assertIn(pred, valid_labels)

    def test_classifier_predict_stream(self):
        """EOGClassifier should predict from streaming dual-channel samples."""
        model_path = os.path.join(self.tmpdir, "eog_model.pkl")
        scaler_path = os.path.join(self.tmpdir, "eog_scaler.pkl")

        classifier = EOGClassifier(
            model_path=model_path,
            scaler_path=scaler_path
        )
        classifier.load()

        # Feed a window of blink-like samples (both channels)
        baseline_h = float(config.EOG_BASELINE)
        for i in range(config.ML_WINDOW_SIZE + config.ML_WINDOW_STEP):
            if 40 <= (i % config.ML_WINDOW_SIZE) < 60:
                sample_v = 3500.0
            else:
                sample_v = 1500.0
            result = classifier.predict(sample_v, baseline_h)

        # After enough samples, we should get a prediction
        # Feed more to get a valid prediction
        for _ in range(config.ML_WINDOW_STEP):
            result = classifier.predict(1500.0, baseline_h)
        # Result could be None or a label - just verify no crash
        if result is not None:
            self.assertIsInstance(result, str)

    def test_classes_are_distinguishable(self):
        """Different EOG patterns should produce different feature distributions."""
        rng = np.random.default_rng(99)

        idle_feats = extract_features(rng.normal(1500, 50, 100))
        blink_feats = extract_features(
            np.concatenate([
                rng.normal(1500, 30, 40),
                rng.normal(3500, 200, 20),
                rng.normal(1500, 30, 40)
            ])
        )

        # Blink should have higher peak amplitude (feature 0)
        self.assertGreater(blink_feats[0], idle_feats[0] * 2)

    @classmethod
    def tearDownClass(cls):
        """Clean up temporary model files."""
        import shutil
        shutil.rmtree(cls.tmpdir, ignore_errors=True)


if __name__ == "__main__":
    unittest.main()
