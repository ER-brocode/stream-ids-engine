//! Packet parsing, Shannon entropy calculation, and sliding-window flow aggregation.
use serde::{Deserialize, Serialize};
use std::collections::{HashMap, VecDeque};

#[derive(Debug, Clone, PartialEq, Eq, Hash, Serialize, Deserialize)]
pub struct FlowKey {
    pub src_ip: String,
    pub dst_ip: String,
    pub src_port: u16,
    pub dst_port: u16,
    pub protocol: String,
}

#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct NetworkPacket {
    pub timestamp: f64,
    pub src_ip: String,
    pub dst_ip: String,
    pub src_port: u16,
    pub dst_port: u16,
    pub protocol: String,
    pub packet_size: usize,
    pub is_syn: bool,
    pub is_ack: bool,
    pub is_fin: bool,
    pub is_rst: bool,
    pub payload_entropy: f64,
}

impl NetworkPacket {
    pub fn flow_key(&self) -> FlowKey {
        FlowKey {
            src_ip: self.src_ip.clone(),
            dst_ip: self.dst_ip.clone(),
            src_port: self.src_port,
            dst_port: self.dst_port,
            protocol: self.protocol.clone(),
        }
    }
}

pub struct PacketDecoder;

impl PacketDecoder {
    pub fn calculate_entropy(data: &[u8]) -> f64 {
        if data.is_empty() {
            return 0.0;
        }
        let mut freq = [0usize; 256];
        for &byte in data {
            freq[byte as usize] += 1;
        }
        let len_f = data.len() as f64;
        let mut entropy = 0.0;
        for &count in &freq {
            if count > 0 {
                let p = count as f64 / len_f;
                entropy -= p * p.log2();
            }
        }
        entropy
    }
}

#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct FlowMetrics {
    pub flow_id: String,
    pub duration: f64,
    pub packet_count: usize,
    pub byte_count: usize,
    pub packet_rate: f64,
    pub byte_rate: f64,
    pub avg_packet_size: f64,
    pub syn_count: usize,
    pub avg_entropy: f64,
}

impl FlowMetrics {
    pub fn to_features(&self) -> Vec<f64> {
        vec![
            self.duration,
            self.packet_count as f64,
            self.byte_count as f64,
            self.packet_rate,
            self.byte_rate,
            self.avg_packet_size,
            self.syn_count as f64,
            self.avg_entropy,
        ]
    }
}

pub struct FlowTracker {
    window_secs: f64,
    flows: HashMap<FlowKey, VecDeque<NetworkPacket>>,
}

impl FlowTracker {
    pub fn new(window_secs: f64) -> Self {
        Self {
            window_secs,
            flows: HashMap::new(),
        }
    }

    pub fn ingest(&mut self, packet: NetworkPacket) -> Option<FlowMetrics> {
        let key = packet.flow_key();
        let queue = self.flows.entry(key.clone()).or_insert_with(VecDeque::new);
        queue.push_back(packet.clone());

        let cutoff = packet.timestamp - self.window_secs;
        while let Some(front) = queue.front() {
            if front.timestamp < cutoff {
                queue.pop_front();
            } else {
                break;
            }
        }

        if queue.is_empty() {
            return None;
        }

        let first_ts = queue.front().unwrap().timestamp;
        let last_ts = queue.back().unwrap().timestamp;
        let duration = (last_ts - first_ts).max(1e-6);
        let count = queue.len();
        let byte_count: usize = queue.iter().map(|p| p.packet_size).sum();
        let syn_count: usize = queue.iter().filter(|p| p.is_syn).count();
        let avg_entropy: f64 = queue.iter().map(|p| p.payload_entropy).sum::<f64>() / count as f64;

        Some(FlowMetrics {
            flow_id: format!("{}:{}->{}:{}", key.src_ip, key.src_port, key.dst_ip, key.dst_port),
            duration,
            packet_count: count,
            byte_count,
            packet_rate: count as f64 / duration,
            byte_rate: byte_count as f64 / duration,
            avg_packet_size: byte_count as f64 / count as f64,
            syn_count,
            avg_entropy,
        })
    }
}
