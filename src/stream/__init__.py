"""
Stream module initialization for packet parsing and feature extraction.
"""

from src.stream.events import NetworkPacket, FlowMetrics, FlowKey
from src.stream.decoder import PacketStreamDecoder
from src.stream.aggregator import StreamFlowAggregator

__all__ = [
    "NetworkPacket",
    "FlowMetrics",
    "FlowKey",
    "PacketStreamDecoder",
    "StreamFlowAggregator",
]
