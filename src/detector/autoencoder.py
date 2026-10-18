"""
Online Streaming Autoencoder with dynamic reconstruction error thresholding and Welford variance tracking.
"""

import numpy as np
from typing import Dict, Optional
from src.detector.base import BaseStreamingDetector, AnomalyScore


class StreamingAutoencoder(BaseStreamingDetector):
    """
    Lightweight, high-performance streaming autoencoder neural network
    with online SGD/Adam updates and online reconstruction error tracking.
    """

    def __init__(
        self,
        input_dim: int = 17,
        hidden_dim: int = 8,
        latent_dim: int = 4,
        learning_rate: float = 0.005,
        threshold_std_multiplier: float = 3.0,
    ):
        self.input_dim = input_dim
        self.hidden_dim = hidden_dim
        self.latent_dim = latent_dim
        self.lr = learning_rate
        self.multiplier = threshold_std_multiplier

        # Weight matrices: Encoder (input -> hidden -> latent)
        np.random.seed(42)
        scale1 = np.sqrt(2.0 / (input_dim + hidden_dim))
        self.W1 = np.random.randn(input_dim, hidden_dim) * scale1
        self.b1 = np.zeros(hidden_dim)

        scale2 = np.sqrt(2.0 / (hidden_dim + latent_dim))
        self.W2 = np.random.randn(hidden_dim, latent_dim) * scale2
        self.b2 = np.zeros(latent_dim)

        # Decoder (latent -> hidden -> output)
        self.W3 = np.random.randn(latent_dim, hidden_dim) * scale2
        self.b3 = np.zeros(hidden_dim)

        self.W4 = np.random.randn(hidden_dim, input_dim) * scale1
        self.b4 = np.zeros(input_dim)

        # Online normalization statistics (Welford's algorithm)
        self.count = 0
        self.mean = np.zeros(input_dim)
        self.M2 = np.zeros(input_dim)

        # Online reconstruction error statistics
        self.error_count = 0
        self.error_mean = 0.0
        self.error_m2 = 0.0

    @staticmethod
    def _relu(x: np.ndarray) -> np.ndarray:
        return np.maximum(0, x)

    @staticmethod
    def _relu_grad(x: np.ndarray) -> np.ndarray:
        return (x > 0).astype(float)

    def _normalize(self, x: np.ndarray) -> np.ndarray:
        if self.count < 2:
            return x
        variance = self.M2 / (self.count - 1)
        std = np.sqrt(np.maximum(variance, 1e-6))
        return (x - self.mean) / std

    def _forward(self, x_norm: np.ndarray):
        # Layer 1
        z1 = np.dot(x_norm, self.W1) + self.b1
        a1 = self._relu(z1)
        # Latent
        z2 = np.dot(a1, self.W2) + self.b2
        a2 = z2  # Linear bottleneck
        # Layer 3
        z3 = np.dot(a2, self.W3) + self.b3
        a3 = self._relu(z3)
        # Reconstruction
        x_recon = np.dot(a3, self.W4) + self.b4
        return a1, a2, a3, x_recon

    def fit_batch(self, X: np.ndarray, epochs: int = 15) -> None:
        """Fit autoencoder on a batch of initial training samples."""
        for row in X:
            self._update_stats(row)

        for _ in range(epochs):
            for row in X:
                self.update_one(row)

    def _update_stats(self, x: np.ndarray) -> None:
        self.count += 1
        delta = x - self.mean
        self.mean += delta / self.count
        delta2 = x - self.mean
        self.M2 += delta * delta2

    def score_one(self, x: np.ndarray, flow_id: str = "", timestamp: float = 0.0) -> AnomalyScore:
        """Compute reconstruction error and classify anomaly."""
        x_arr = np.asarray(x, dtype=np.float64)
        x_norm = self._normalize(x_arr)

        _, _, _, x_recon = self._forward(x_norm)
        diff = x_norm - x_recon
        squared_errors = diff ** 2
        raw_error = float(np.mean(squared_errors))

        # Dynamic threshold based on baseline reconstruction errors
        if self.error_count > 30:
            error_var = self.error_m2 / (self.error_count - 1)
            error_std = float(np.sqrt(max(error_var, 1e-6)))
            threshold = self.error_mean + self.multiplier * error_std
            # Sigmoid scaling around threshold
            scale_factor = 2.0 / (error_std + 1e-4)
            anomaly_score = float(1.0 / (1.0 + np.exp(-scale_factor * (raw_error - threshold))))
        else:
            threshold = 1.0
            anomaly_score = float(1.0 / (1.0 + np.exp(-(raw_error - 1.0))))

        is_anomaly = raw_error > threshold

        # Feature contributions: which feature dimensions yielded highest reconstruction error
        top_indices = np.argsort(squared_errors)[::-1][:4]
        contributions = {
            self.FEATURE_NAMES[i]: float(squared_errors[i]) for i in top_indices
        }

        return AnomalyScore(
            flow_id=flow_id,
            is_anomaly=is_anomaly,
            anomaly_score=anomaly_score,
            threshold=threshold,
            detector_name="StreamingAutoencoder",
            feature_contributions=contributions,
            raw_score=raw_error,
            timestamp=timestamp,
        )

    def update_one(self, x: np.ndarray) -> None:
        """Online stochastic gradient descent backprop and error distribution tracking."""
        x_arr = np.asarray(x, dtype=np.float64)
        self._update_stats(x_arr)
        x_norm = self._normalize(x_arr)

        # Forward pass
        a1, a2, a3, x_recon = self._forward(x_norm)
        error = x_recon - x_norm
        mse = float(np.mean(error ** 2))

        # Backpropagation
        d_out = 2.0 * error / self.input_dim
        dW4 = np.outer(a3, d_out)
        db4 = d_out

        d_a3 = np.dot(d_out, self.W4.T)
        d_z3 = d_a3 * self._relu_grad(a3)
        dW3 = np.outer(a2, d_z3)
        db3 = d_z3

        d_a2 = np.dot(d_z3, self.W3.T)
        d_z2 = d_a2  # Linear
        dW2 = np.outer(a1, d_z2)
        db2 = d_z2

        d_a1 = np.dot(d_z2, self.W2.T)
        d_z1 = d_a1 * self._relu_grad(a1)
        dW1 = np.outer(x_norm, d_z1)
        db1 = d_z1

        # Parameter update with gradient clipping
        np.clip(dW4, -5.0, 5.0, out=dW4)
        np.clip(dW3, -5.0, 5.0, out=dW3)
        np.clip(dW2, -5.0, 5.0, out=dW2)
        np.clip(dW1, -5.0, 5.0, out=dW1)

        self.W4 -= self.lr * dW4
        self.b4 -= self.lr * db4
        self.W3 -= self.lr * dW3
        self.b3 -= self.lr * db3
        self.W2 -= self.lr * dW2
        self.b2 -= self.lr * db2
        self.W1 -= self.lr * dW1
        self.b1 -= self.lr * db1

        # Update reconstruction error running distribution
        self.error_count += 1
        e_delta = mse - self.error_mean
        self.error_mean += e_delta / self.error_count
        e_delta2 = mse - self.error_mean
        self.error_m2 += e_delta * e_delta2
