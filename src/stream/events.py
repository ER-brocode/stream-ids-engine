"""
Packet and network event schemas for stream decoding and analysis.
"""

from dataclasses import dataclass, field
from datetime import datetime
from typing import Dict, Any, Optional
import numpy as np


@dataclass(frozen=True)
class FlowKey:
    """5-tuple network flow key identifier."""
    src_ip: str
    dst_ip: str
    src_port: int
    dst_port: int
    protocol: str

    def reverse(self) -> "FlowKey":
        """Generate reverse directional key for bidirectional flow tracking."""
        return FlowKey(
            src_ip=self.dst_ip,
            dst_ip=self.src_ip,
            src_port=self.dst_port,
            dst_port=self.src_port,
            protocol=self.protocol,
        )


@dataclass
class NetworkPacket:
    """Decoded raw packet representation."""
    timestamp: float
    src_ip: str
    dst_ip: str
    src_port: int
    dst_port: int
    protocol: str  # TCP, UDP, ICMP, etc.
    packet_size: int
    tcp_flags: Dict[str, bool] = field(default_factory=dict)
    payload_entropy: float = 0.0
    payload: bytes = b""

    @property
    def flow_key(self) -> FlowKey:
        return FlowKey(
            src_ip=self.src_ip,
            dst_ip=self.dst_ip,
            src_port=self.src_port,
            dst_port=self.dst_port,
            protocol=self.protocol,
        )


@dataclass
class FlowMetrics:
    """Extracted statistical features for ML anomaly detection."""
    flow_id: str
    src_ip: str
    dst_ip: str
    src_port: int
    dst_port: int
    protocol: str
    start_time: float
    duration: float
    packet_count: int
    byte_count: int
    packet_rate: float
    byte_rate: float
    avg_packet_size: float
    std_packet_size: float
    min_packet_size: float
    max_packet_size: float
    avg_iat: float  # Inter-arrival time
    std_iat: float
    syn_count: int
    ack_count: int
    fin_count: int
    rst_count: int
    psh_count: int
    avg_entropy: float

    def to_feature_vector(self) -> np.ndarray:
        """Convert metrics to normalized numpy feature array for model input."""
        return np.array([
            float(self.duration),
            float(self.packet_count),
            float(self.byte_count),
            float(self.packet_rate),
            float(self.byte_rate),
            float(self.avg_packet_size),
            float(self.std_packet_size),
            float(self.min_packet_size),
            float(self.max_packet_size),
            float(self.avg_iat),
            float(self.std_iat),
            float(self.syn_count),
            float(self.ack_count),
            float(self.fin_count),
            float(self.rst_count),
            float(self.psh_count),
            float(self.avg_entropy),
        ], dtype=np.float32)

    def to_dict(self) -> Dict[str, Any]:
        """Convert flow metrics to serializable dictionary."""
        return {
            "flow_id": self.flow_id,
            "src_ip": self.src_ip,
            "dst_ip": self.dst_ip,
            "src_port": self.src_port,
            "dst_port": self.dst_port,
            "protocol": self.protocol,
            "start_time": self.start_time,
            "duration": round(self.duration, 4),
            "packet_count": self.packet_count,
            "byte_count": self.byte_count,
            "packet_rate": round(self.packet_rate, 2),
            "byte_rate": round(self.byte_rate, 2),
            "avg_packet_size": round(self.avg_packet_size, 2),
            "std_packet_size": round(self.std_packet_size, 2),
            "min_packet_size": self.min_packet_size,
            "max_packet_size": self.max_packet_size,
            "avg_iat": round(self.avg_iat, 6),
            "std_iat": round(self.std_iat, 6),
            "tcp_flags": {
                "syn": self.syn_count,
                "ack": self.ack_count,
                "fin": self.fin_count,
                "rst": self.rst_count,
                "psh": self.psh_count,
            },
            "avg_entropy": round(self.avg_entropy, 4),
        }
