//! Binary entrypoint for stream-ids Rust engine.
use stream_ids::{FlowTracker, NetworkPacket, StreamingDetector, ThreatReport};

fn main() {
    println!("=== Stream IDS Engine (Rust Fast-Path) ===");
    let mut tracker = FlowTracker::new(10.0);
    let mut detector = StreamingDetector::new(8, 3.0);

    // Warm up detector with synthetic baseline packets
    for i in 0..50 {
        let pkt = NetworkPacket {
            timestamp: i as f64 * 0.1,
            src_ip: "192.168.1.100".to_string(),
            dst_ip: "10.0.0.1".to_string(),
            src_port: 54321,
            dst_port: 443,
            protocol: "TCP".to_string(),
            packet_size: 512,
            is_syn: false,
            is_ack: true,
            is_fin: false,
            is_rst: false,
            payload_entropy: 3.5,
        };
        if let Some(metrics) = tracker.ingest(pkt) {
            detector.update(&metrics.to_features());
        }
    }

    // Ingest anomaly packet
    let anomaly_pkt = NetworkPacket {
        timestamp: 15.0,
        src_ip: "192.168.1.200".to_string(),
        dst_ip: "10.0.0.1".to_string(),
        src_port: 60000,
        dst_port: 80,
        protocol: "TCP".to_string(),
        packet_size: 4096,
        is_syn: true,
        is_ack: false,
        is_fin: false,
        is_rst: false,
        payload_entropy: 7.9,
    };

    if let Some(metrics) = tracker.ingest(anomaly_pkt) {
        let features = metrics.to_features();
        let result = detector.score(&features, &metrics.flow_id);
        println!("Anomaly evaluation: {:?}", result);
        if result.is_anomaly {
            let threat = ThreatReport::classify(metrics.packet_rate, metrics.syn_count, metrics.avg_entropy);
            println!("Threat classified: {:?}", threat);
        }
    }
}
