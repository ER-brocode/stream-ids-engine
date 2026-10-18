"""
Alerts module initialization for threat intelligence classification and SIEM dispatching.
"""

from src.alerts.threat_intel import ThreatClassifier, ThreatReport, ThreatSeverity, AttackCategory
from src.alerts.dispatcher import SiemWebhookDispatcher, AlertEvent

__all__ = [
    "ThreatClassifier",
    "ThreatReport",
    "ThreatSeverity",
    "AttackCategory",
    "SiemWebhookDispatcher",
    "AlertEvent",
]
