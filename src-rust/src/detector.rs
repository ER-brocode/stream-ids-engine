//! Online streaming detector implementing incremental Welford statistics and outlier scoring.
use serde::{Deserialize, Serialize};

#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct AnomalyResult {
    pub flow_id: String,
    pub is_anomaly: bool,
    pub score: f64,
    pub threshold: f64,
    pub dominant_feature: String,
}

pub struct StreamingDetector {
    num_features: usize,
    count: usize,
    mean: Vec<f64>,
    m2: Vec<f64>,
    threshold_sigma: f64,
}

impl StreamingDetector {
    pub fn new(num_features: usize, threshold_sigma: f64) -> Self {
        Self {
            num_features,
            count: 0,
            mean: vec![0.0; num_features],
            m2: vec![0.0; num_features],
            threshold_sigma,
        }
    }

    pub fn update(&mut self, features: &[f64]) {
        self.count += 1;
        let c = self.count as f64;
        for i in 0..self.num_features {
            let delta = features[i] - self.mean[i];
            self.mean[i] += delta / c;
            let delta2 = features[i] - self.mean[i];
            self.m2[i] += delta * delta2;
        }
    }

    pub fn score(&self, features: &[f64], flow_id: &str) -> AnomalyResult {
        if self.count < 10 {
            return AnomalyResult {
                flow_id: flow_id.to_string(),
                is_anomaly: false,
                score: 0.0,
                threshold: self.threshold_sigma,
                dominant_feature: "warmup".to_string(),
            };
        }

        let mut max_z = 0.0;
        let mut dominant_idx = 0;
        let feature_names = [
            "duration", "packet_count", "byte_count", "packet_rate",
            "byte_rate", "avg_packet_size", "syn_count", "avg_entropy"
        ];

        for i in 0..self.num_features {
            let var = (self.m2[i] / (self.count - 1) as f64).max(1e-6);
            let std = var.sqrt();
            let z = (features[i] - self.mean[i]).abs() / std;
            if z > max_z {
                max_z = z;
                dominant_idx = i;
            }
        }

        let normalized_score = 1.0 / (1.0 + (-0.8 * (max_z - self.threshold_sigma)).exp());
        let is_anomaly = max_z > self.threshold_sigma;

        AnomalyResult {
            flow_id: flow_id.to_string(),
            is_anomaly,
            score: normalized_score,
            threshold: self.threshold_sigma,
            dominant_feature: feature_names.get(dominant_idx).unwrap_or(&"unknown").to_string(),
        }
    }
}
