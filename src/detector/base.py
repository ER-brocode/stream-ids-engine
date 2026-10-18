"""
Base classes and result contracts for streaming anomaly detectors.
"""

from abc import ABC, abstractmethod
from dataclasses import dataclass
from typing import Dict, Any, List, Optional
import numpy as np


@dataclass
class AnomalyScore:
    """Standardized anomaly evaluation outcome."""
    flow_id: str
    is_anomaly: bool
    anomaly_score: float  # Normalized 0.0 to 1.0 (higher = more anomalous)
    threshold: float
    detector_name: str
    feature_contributions: Dict[str, float]
    raw_score: float
    timestamp: float

    def to_dict(self) -> Dict[str, Any]:
        return {
            "flow_id": self.flow_id,
            "is_anomaly": self.is_anomaly,
            "anomaly_score": round(float(self.anomaly_score), 4),
            "threshold": round(float(self.threshold), 4),
            "detector_name": self.detector_name,
            "feature_contributions": {
                k: round(float(v), 4) for k, v in self.feature_contributions.items()
            },
            "raw_score": round(float(self.raw_score), 4),
            "timestamp": self.timestamp,
        }


class BaseStreamingDetector(ABC):
    """Abstract interface for all streaming ML anomaly detection algorithms."""

    FEATURE_NAMES = [
        "duration",
        "packet_count",
        "byte_count",
        "packet_rate",
        "byte_rate",
        "avg_packet_size",
        "std_packet_size",
        "min_packet_size",
        "max_packet_size",
        "avg_iat",
        "std_iat",
        "syn_count",
        "ack_count",
        "fin_count",
        "rst_count",
        "psh_count",
        "avg_entropy",
    ]

    @abstractmethod
    def fit_batch(self, X: np.ndarray) -> None:
        """Warm-up or fit detector on baseline normal traffic."""
        pass

    @abstractmethod
    def score_one(self, x: np.ndarray, flow_id: str = "", timestamp: float = 0.0) -> AnomalyScore:
        """Score a single incoming flow vector in real time."""
        pass

    @abstractmethod
    def update_one(self, x: np.ndarray) -> None:
        """Online update detector state with an incoming observation."""
        pass
