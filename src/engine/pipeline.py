"""
End-to-end Streaming IDS pipeline coordinator connecting decoder, aggregator, detector, and dispatcher.
"""

import time
import uuid
import logging
from typing import Dict, Any, Optional, List
import numpy as np

from src.stream.events import NetworkPacket, FlowMetrics
from src.stream.decoder import PacketStreamDecoder
from src.stream.aggregator import StreamFlowAggregator
from src.detector.ensemble import StreamingEnsembleDetector
from src.alerts.threat_intel import ThreatClassifier
from src.alerts.dispatcher import SiemWebhookDispatcher, AlertEvent

logger = logging.getLogger("stream_ids.engine")


class StreamIDSEngine:
    """Full production pipeline executing real-time anomaly detection on network event streams."""

    def __init__(
        self,
        window_seconds: float = 10.0,
        webhook_urls: Optional[List[str]] = None,
        warmup_samples: int = 50,
    ):
        self.decoder = PacketStreamDecoder()
        self.aggregator = StreamFlowAggregator(window_seconds=window_seconds)
        self.detector = StreamingEnsembleDetector()
        self.dispatcher = SiemWebhookDispatcher(webhook_urls=webhook_urls)
        self.warmup_samples = warmup_samples
        self.is_warmed_up = False
        self.warmup_buffer: List[np.ndarray] = []

    def warm_up(self, baseline_packets: List[Dict[str, Any]]) -> None:
        """Warm-up engine using normal baseline traffic packets."""
        feature_vectors = []
        for raw in baseline_packets:
            packet = self.decoder.decode_dict_event(raw)
            metrics = self.aggregator.ingest_packet(packet)
            if metrics:
                vec = metrics.to_feature_vector()
                feature_vectors.append(vec)

        if feature_vectors:
            mat = np.array(feature_vectors)
            self.detector.fit_batch(mat)
            self.is_warmed_up = True

    def process_raw_packet(self, packet_bytes: bytes, timestamp: Optional[float] = None) -> Optional[AlertEvent]:
        """Ingest raw Ethernet/IP packet bytes and inspect for intrusions."""
        packet = self.decoder.decode_raw_ethernet(packet_bytes, timestamp=timestamp)
        if not packet:
            return None
        return self._process_packet_internal(packet)

    def process_event(self, event_dict: Dict[str, Any]) -> Optional[AlertEvent]:
        """Ingest event dictionary and inspect for intrusions."""
        packet = self.decoder.decode_dict_event(event_dict)
        return self._process_packet_internal(packet)

    def _process_packet_internal(self, packet: NetworkPacket) -> Optional[AlertEvent]:
        metrics = self.aggregator.ingest_packet(packet)
        if not metrics:
            return None

        feature_vec = metrics.to_feature_vector()

        if not self.is_warmed_up:
            self.warmup_buffer.append(feature_vec)
            if len(self.warmup_buffer) >= self.warmup_samples:
                self.detector.fit_batch(np.array(self.warmup_buffer))
                self.is_warmed_up = True
            return None

        # Real-time scoring
        score = self.detector.score_one(feature_vec, flow_id=metrics.flow_id, timestamp=packet.timestamp)

        # Online model update
        if not score.is_anomaly:
            self.detector.update_one(feature_vec)

        if score.is_anomaly:
            threat = ThreatClassifier.classify(metrics, score)
            alert = AlertEvent(
                alert_id=f"alt-{uuid.uuid4().hex[:8]}",
                timestamp=packet.timestamp,
                threat=threat,
                anomaly=score,
                flow=metrics,
            )
            self.dispatcher.dispatch(alert)
            return alert

        return None


def main():
    """Command-line entrypoint."""
    logging.basicConfig(level=logging.INFO, format="%(asctime)s [%(levelname)s] %(message)s")
    engine = StreamIDSEngine()
    print("Stream IDS Engine initialized and ready.")


if __name__ == "__main__":
    main()
