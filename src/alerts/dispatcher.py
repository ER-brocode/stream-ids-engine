"""
SIEM webhook and notification dispatcher supporting Splunk, Elastic, Slack, and generic HTTP endpoints.
"""

import json
import logging
import time
from dataclasses import dataclass, field
from typing import Dict, Any, List, Optional
import urllib.request
import urllib.error

from src.alerts.threat_intel import ThreatReport
from src.detector.base import AnomalyScore
from src.stream.events import FlowMetrics

logger = logging.getLogger("stream_ids.dispatcher")


@dataclass
class AlertEvent:
    """Consolidated security alert model ready for SIEM / SOC transmission."""
    alert_id: str
    timestamp: float
    threat: ThreatReport
    anomaly: AnomalyScore
    flow: FlowMetrics
    metadata: Dict[str, Any] = field(default_factory=dict)

    def to_dict(self) -> Dict[str, Any]:
        return {
            "alert_id": self.alert_id,
            "timestamp": self.timestamp,
            "threat": self.threat.to_dict(),
            "anomaly": self.anomaly.to_dict(),
            "flow": self.flow.to_dict(),
            "metadata": self.metadata,
        }

    def to_json(self) -> str:
        return json.dumps(self.to_dict(), indent=2)


class SiemWebhookDispatcher:
    """Dispatches real-time alerts to SIEM, webhooks, or local audit logs."""

    def __init__(
        self,
        webhook_urls: Optional[List[str]] = None,
        timeout_seconds: float = 3.0,
        enable_audit_log: bool = True,
    ):
        self.webhook_urls = webhook_urls or []
        self.timeout = timeout_seconds
        self.enable_audit_log = enable_audit_log
        self.dispatched_alerts: List[AlertEvent] = []

    def dispatch(self, alert: AlertEvent) -> Dict[str, Any]:
        """Dispatch alert event to all registered webhooks and audit storage."""
        self.dispatched_alerts.append(alert)
        results = {}

        if self.enable_audit_log:
            logger.warning(
                f"[ALERT] {alert.threat.severity.value} - {alert.threat.category.value} | "
                f"Src: {alert.flow.src_ip}:{alert.flow.src_port} -> "
                f"Dst: {alert.flow.dst_ip}:{alert.flow.dst_port} | "
                f"Score: {alert.anomaly.anomaly_score:.2f} | "
                f"Action: {alert.threat.recommended_action}"
            )

        payload = json.dumps(alert.to_dict()).encode("utf-8")

        for url in self.webhook_urls:
            try:
                req = urllib.request.Request(
                    url,
                    data=payload,
                    headers={"Content-Type": "application/json", "User-Agent": "StreamIDS/1.0"},
                )
                with urllib.request.urlopen(req, timeout=self.timeout) as resp:
                    results[url] = {"status": resp.status, "success": True}
            except Exception as e:
                logger.error(f"Failed to post alert to {url}: {e}")
                results[url] = {"status": 0, "error": str(e), "success": False}

        return results
