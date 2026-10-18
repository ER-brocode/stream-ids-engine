"""
Streaming Ensemble combining Isolation Forest and Autoencoder detectors.
"""

from typing import List, Optional
import numpy as np
from src.detector.base import BaseStreamingDetector, AnomalyScore
from src.detector.isolation_forest import StreamingIsolationForest
from src.detector.autoencoder import StreamingAutoencoder


class StreamingEnsembleDetector(BaseStreamingDetector):
    """
    Combines Streaming Isolation Forest and Streaming Autoencoder predictions
    using weighted confidence aggregation for robust low-false-positive detection.
    """

    def __init__(
        self,
        weight_iforest: float = 0.5,
        weight_autoencoder: float = 0.5,
        ensemble_threshold: float = 0.60,
    ):
        self.weight_iforest = weight_iforest
        self.weight_autoencoder = weight_autoencoder
        self.ensemble_threshold = ensemble_threshold

        self.iforest = StreamingIsolationForest()
        self.autoencoder = StreamingAutoencoder()

    def fit_batch(self, X: np.ndarray) -> None:
        """Warm-up both ensemble component detectors."""
        self.iforest.fit_batch(X)
        self.autoencoder.fit_batch(X)

    def score_one(self, x: np.ndarray, flow_id: str = "", timestamp: float = 0.0) -> AnomalyScore:
        """Score with both models and generate unified anomaly attribution."""
        score_if = self.iforest.score_one(x, flow_id=flow_id, timestamp=timestamp)
        score_ae = self.autoencoder.score_one(x, flow_id=flow_id, timestamp=timestamp)

        combined_score = (
            self.weight_iforest * score_if.anomaly_score
            + self.weight_autoencoder * score_ae.anomaly_score
        )
        is_anomaly = combined_score >= self.ensemble_threshold

        # Merge feature contributions
        contributions = {}
        for k, v in score_if.feature_contributions.items():
            contributions[k] = contributions.get(k, 0.0) + self.weight_iforest * v
        for k, v in score_ae.feature_contributions.items():
            contributions[k] = contributions.get(k, 0.0) + self.weight_autoencoder * v

        # Top 4 contributors
        sorted_contrib = dict(
            sorted(contributions.items(), key=lambda item: item[1], reverse=True)[:4]
        )

        return AnomalyScore(
            flow_id=flow_id,
            is_anomaly=is_anomaly,
            anomaly_score=combined_score,
            threshold=self.ensemble_threshold,
            detector_name="StreamingEnsemble(IForest+Autoencoder)",
            feature_contributions=sorted_contrib,
            raw_score=combined_score,
            timestamp=timestamp,
        )

    def update_one(self, x: np.ndarray) -> None:
        """Stream update both models."""
        self.iforest.update_one(x)
        self.autoencoder.update_one(x)
