"""
Threat Intelligence heuristic classification and MITRE ATT&CK taxonomy mapping.
"""

from enum import Enum
from dataclasses import dataclass
from typing import Dict, Any, List, Optional
from src.stream.events import FlowMetrics
from src.detector.base import AnomalyScore


class ThreatSeverity(str, Enum):
    LOW = "LOW"
    MEDIUM = "MEDIUM"
    HIGH = "HIGH"
    CRITICAL = "CRITICAL"


class AttackCategory(str, Enum):
    PORT_SCAN = "Reconnaissance: Port Scan"
    SYN_FLOOD = "Denial of Service: SYN Flood"
    UDP_FLOOD = "Denial of Service: UDP Flood"
    SLOWLORIS = "Denial of Service: Slowloris"
    DATA_EXFILTRATION = "Exfiltration: Encrypted Tunnel"
    BRUTE_FORCE = "Credential Access: Brute Force"
    UNKNOWN_ANOMALY = "Behavioral Anomaly: Unclassified"


@dataclass
class ThreatReport:
    """Classified threat assessment."""
    category: AttackCategory
    severity: ThreatSeverity
    mitre_technique_id: str
    description: str
    indicators: List[str]
    confidence: float
    recommended_action: str

    def to_dict(self) -> Dict[str, Any]:
        return {
            "category": self.category.value,
            "severity": self.severity.value,
            "mitre_technique_id": self.mitre_technique_id,
            "description": self.description,
            "indicators": self.indicators,
            "confidence": round(self.confidence, 4),
            "recommended_action": self.recommended_action,
        }


class ThreatClassifier:
    """Classifies anomalous flow patterns into actionable threat intelligence."""

    KNOWN_MALICIOUS_PORTS = {
        22: "SSH",
        23: "Telnet",
        3389: "RDP",
        445: "SMB",
        1433: "MSSQL",
        3306: "MySQL",
        5432: "PostgreSQL",
    }

    @classmethod
    def classify(cls, metrics: FlowMetrics, score: AnomalyScore) -> ThreatReport:
        """Classify flow metrics and anomaly indicators into MITRE attack categories."""
        indicators = []
        p_count = metrics.packet_count
        p_rate = metrics.packet_rate
        b_rate = metrics.byte_rate
        syn_ratio = metrics.syn_count / max(1, p_count)
        avg_entropy = metrics.avg_entropy

        # 1. SYN Flood Detection
        if metrics.protocol == "TCP" and syn_ratio > 0.8 and p_rate > 50:
            indicators.append(f"High SYN ratio ({syn_ratio:.1%}) with elevated packet rate ({p_rate:.1f} pps)")
            return ThreatReport(
                category=AttackCategory.SYN_FLOOD,
                severity=ThreatSeverity.CRITICAL if p_rate > 200 else ThreatSeverity.HIGH,
                mitre_technique_id="T1498.001",
                description="High-velocity TCP SYN flood detected targeting host service.",
                indicators=indicators,
                confidence=min(1.0, 0.7 + (p_rate / 500)),
                recommended_action=f"Apply rate limiting and SYN cookies on firewall for target {metrics.dst_ip}:{metrics.dst_port}.",
            )

        # 2. UDP Flood Detection
        if metrics.protocol == "UDP" and p_rate > 100:
            indicators.append(f"Volumetric UDP traffic stream at {p_rate:.1f} pps, {b_rate / 1024:.1f} KB/s")
            return ThreatReport(
                category=AttackCategory.UDP_FLOOD,
                severity=ThreatSeverity.HIGH,
                mitre_technique_id="T1498.002",
                description="Volumetric UDP packet flood detected.",
                indicators=indicators,
                confidence=min(1.0, 0.75 + (p_rate / 400)),
                recommended_action=f"Drop UDP bursts or blacklist source IP {metrics.src_ip}.",
            )

        # 3. Port Scanning / Reconnaissance
        if p_count <= 4 and metrics.duration < 2.0 and metrics.syn_count >= 1 and metrics.ack_count == 0:
            indicators.append("Brief probe sequence with unanswered SYN request")
            return ThreatReport(
                category=AttackCategory.PORT_SCAN,
                severity=ThreatSeverity.LOW if metrics.dst_port not in cls.KNOWN_MALICIOUS_PORTS else ThreatSeverity.MEDIUM,
                mitre_technique_id="T1046",
                description=f"Network service probing / port scan against port {metrics.dst_port}.",
                indicators=indicators,
                confidence=0.85,
                recommended_action=f"Monitor source IP {metrics.src_ip} for sequential sweep across subnet.",
            )

        # 4. Data Exfiltration via High Entropy Tunnel
        if avg_entropy > 7.2 and metrics.byte_count > 50000 and metrics.dst_port not in [443, 8443]:
            indicators.append(f"Abnormally high payload entropy ({avg_entropy:.2f}/8.0) on non-standard port {metrics.dst_port}")
            return ThreatReport(
                category=AttackCategory.DATA_EXFILTRATION,
                severity=ThreatSeverity.CRITICAL,
                mitre_technique_id="T1048",
                description="Potential encrypted covert tunnel or data exfiltration channel.",
                indicators=indicators,
                confidence=0.92,
                recommended_action=f"Isolate host {metrics.src_ip} and inspect outbound connections to {metrics.dst_ip}.",
            )

        # 5. Brute Force against administrative ports
        if metrics.dst_port in cls.KNOWN_MALICIOUS_PORTS and metrics.packet_count > 30 and metrics.avg_iat < 0.2:
            svc = cls.KNOWN_MALICIOUS_PORTS[metrics.dst_port]
            indicators.append(f"Rapid repetitive connection attempts to {svc} (port {metrics.dst_port})")
            return ThreatReport(
                category=AttackCategory.BRUTE_FORCE,
                severity=ThreatSeverity.HIGH,
                mitre_technique_id="T1110",
                description=f"Potential brute-force authentication attack on {svc}.",
                indicators=indicators,
                confidence=0.88,
                recommended_action=f"Enforce fail2ban or multi-factor authentication on port {metrics.dst_port}.",
            )

        # 6. Fallback Unclassified Behavioral Anomaly
        top_contrib = [f"{k}={v:.2f}" for k, v in score.feature_contributions.items()]
        indicators.append(f"Deviations detected in features: {', '.join(top_contrib)}")
        severity = ThreatSeverity.MEDIUM if score.anomaly_score < 0.8 else ThreatSeverity.HIGH
        return ThreatReport(
            category=AttackCategory.UNKNOWN_ANOMALY,
            severity=severity,
            mitre_technique_id="T1000",
            description="Statistical baseline anomaly exceeding streaming ML threshold.",
            indicators=indicators,
            confidence=score.anomaly_score,
            recommended_action=f"Inspect session telemetry between {metrics.src_ip} and {metrics.dst_ip}.",
        )
