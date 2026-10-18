"""
Streaming Isolation Forest detector with rolling reservoir sampling and adaptive thresholds.
"""

from collections import deque
import numpy as np
from sklearn.ensemble import IsolationForest
from typing import Dict, Optional, Deque
from src.detector.base import BaseStreamingDetector, AnomalyScore


class StreamingIsolationForest(BaseStreamingDetector):
    """
    Streaming Isolation Forest implementing adaptive window re-fitting
    and online feature-contribution attribution.
    """

    def __init__(
        self,
        buffer_size: int = 2000,
        refit_interval: int = 250,
        contamination: float = 0.05,
        n_estimators: int = 100,
        random_state: int = 42,
    ):
        self.buffer_size = buffer_size
        self.refit_interval = refit_interval
        self.contamination = contamination
        self.n_estimators = n_estimators
        self.random_state = random_state

        self.buffer: Deque[np.ndarray] = deque(maxlen=buffer_size)
        self.model: Optional[IsolationForest] = None
        self.points_since_refit = 0
        self.baseline_mean = np.zeros(len(self.FEATURE_NAMES))
        self.baseline_std = np.ones(len(self.FEATURE_NAMES))
        self.threshold = 0.65

    def fit_batch(self, X: np.ndarray) -> None:
        """Warm-up detector on initial batch of historical or baseline vectors."""
        if len(X) == 0:
            return

        for row in X:
            self.buffer.append(row.astype(np.float64))

        self.baseline_mean = np.mean(X, axis=0)
        self.baseline_std = np.std(X, axis=0)
        self.baseline_std[self.baseline_std < 1e-6] = 1.0

        self._fit_model()

    def _fit_model(self) -> None:
        """Internal retrain on the current buffer contents."""
        if len(self.buffer) < 50:
            return

        data = np.array(self.buffer)
        norm_data = (data - self.baseline_mean) / self.baseline_std

        clf = IsolationForest(
            n_estimators=self.n_estimators,
            contamination=self.contamination,
            random_state=self.random_state,
            n_jobs=1,
        )
        clf.fit(norm_data)
        self.model = clf
        self.points_since_refit = 0

    def score_one(self, x: np.ndarray, flow_id: str = "", timestamp: float = 0.0) -> AnomalyScore:
        """Score a single incoming flow vector and compute feature attributions."""
        x_vec = np.asarray(x, dtype=np.float64).reshape(1, -1)
        norm_x = (x_vec - self.baseline_mean) / self.baseline_std

        if self.model is None or len(self.buffer) < 50:
            # Fallback to Mahalanobis / Z-score distance before model is trained
            z_scores = np.abs(norm_x[0])
            raw_score = float(np.mean(z_scores))
            norm_score = float(1.0 / (1.0 + np.exp(-0.5 * (raw_score - 2.5))))
            is_anomaly = norm_score >= self.threshold

            top_indices = np.argsort(z_scores)[::-1][:4]
            contributions = {
                self.FEATURE_NAMES[i]: float(z_scores[i]) for i in top_indices
            }
            return AnomalyScore(
                flow_id=flow_id,
                is_anomaly=is_anomaly,
                anomaly_score=norm_score,
                threshold=self.threshold,
                detector_name="StreamingIsolationForest_ColdStart",
                feature_contributions=contributions,
                raw_score=raw_score,
                timestamp=timestamp,
            )

        # Isolation forest decision_function: lower score = more anomalous
        # decision_function yields values in roughly [-0.5, 0.5]
        dec_score = self.model.decision_function(norm_x)[0]
        # Invert and normalize so that higher = more anomalous (0.0 normal, 1.0 anomalous)
        anomaly_score = float(1.0 / (1.0 + np.exp(10.0 * dec_score)))
        is_anomaly = anomaly_score >= self.threshold

        # Feature contributions via absolute normalized deviation
        abs_deviations = np.abs(norm_x[0])
        top_indices = np.argsort(abs_deviations)[::-1][:4]
        contributions = {
            self.FEATURE_NAMES[i]: float(abs_deviations[i]) for i in top_indices
        }

        return AnomalyScore(
            flow_id=flow_id,
            is_anomaly=is_anomaly,
            anomaly_score=anomaly_score,
            threshold=self.threshold,
            detector_name="StreamingIsolationForest",
            feature_contributions=contributions,
            raw_score=float(dec_score),
            timestamp=timestamp,
        )

    def update_one(self, x: np.ndarray) -> None:
        """Add observation to buffer and periodically retrain."""
        self.buffer.append(np.asarray(x, dtype=np.float64))
        self.points_since_refit += 1

        if self.points_since_refit >= self.refit_interval and len(self.buffer) >= 100:
            data = np.array(self.buffer)
            self.baseline_mean = np.mean(data, axis=0)
            self.baseline_std = np.std(data, axis=0)
            self.baseline_std[self.baseline_std < 1e-6] = 1.0
            self._fit_model()
