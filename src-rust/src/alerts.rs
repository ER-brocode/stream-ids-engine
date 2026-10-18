//! Threat classification and SIEM webhook event dispatch in Rust.
use serde::{Deserialize, Serialize};

#[derive(Debug, Clone, Serialize, Deserialize)]
pub enum ThreatSeverity {
    LOW,
    MEDIUM,
    HIGH,
    CRITICAL,
}

#[derive(Debug, Clone, Serialize, Deserialize)]
pub enum AttackType {
    SynFlood,
    UdpFlood,
    DataExfiltration,
    PortScan,
    UnknownAnomaly,
}

#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct ThreatReport {
    pub attack_type: AttackType,
    pub severity: ThreatSeverity,
    pub description: String,
    pub mitigation: String,
}

impl ThreatReport {
    pub fn classify(flow_rate: f64, syn_count: usize, avg_entropy: f64) -> Self {
        if syn_count > 50 && flow_rate > 100.0 {
            Self {
                attack_type: AttackType::SynFlood,
                severity: ThreatSeverity::CRITICAL,
                description: "High rate SYN flood denial of service detected.".to_string(),
                mitigation: "Engage syn-proxy and apply rate limits.".to_string(),
            }
        } else if avg_entropy > 7.3 {
            Self {
                attack_type: AttackType::DataExfiltration,
                severity: ThreatSeverity::HIGH,
                description: "Abnormally high entropy stream indicating encrypted exfiltration.".to_string(),
                mitigation: "Quarantine host and inspect egress tunnel.".to_string(),
            }
        } else {
            Self {
                attack_type: AttackType::UnknownAnomaly,
                severity: ThreatSeverity::MEDIUM,
                description: "Statistical outlier behavior detected.".to_string(),
                mitigation: "Review flow telemetry.".to_string(),
            }
        }
    }
}
