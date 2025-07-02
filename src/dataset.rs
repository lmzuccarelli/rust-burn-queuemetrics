use burn::{
    data::{dataloader::batcher::Batcher, dataset::Dataset, dataset::InMemDataset},
    prelude::*,
};

pub const NUM_FEATURES: usize = 4;
pub const NUM_CLASSES: usize = 3;

// Pre-computed statistics for the vitalsign dataset features
// Inputs are num_processors, avg_batch_size, queue_length, process_time
const FEATURES_MIN: [f32; NUM_FEATURES] = [0.0, 0.0, 1.0, 1.0];
const FEATURES_MAX: [f32; NUM_FEATURES] = [4.0, 1000.10, 100.0, 100.0];

#[derive(Clone, Debug, serde::Serialize, serde::Deserialize)]
pub struct QueueMetrics {
    /// processors
    #[serde(rename = "processors")]
    pub processors: f32,

    /// batch size
    #[serde(rename = "avg_batch_size")]
    pub avg_batch_size: f32,

    /// queue_length
    #[serde(rename = "queue_length")]
    pub queue_length: f32,

    /// process_time
    #[serde(rename = "processing_time")]
    pub processing_time: f32,

    /// status - the label ;)
    #[serde(rename = "status")]
    pub status: f32,
}

pub struct QueueMetricsDataset {
    dataset: InMemDataset<QueueMetrics>,
}

impl QueueMetricsDataset {
    pub fn train() -> Self {
        Self::new("train")
    }

    pub fn validation() -> Self {
        Self::new("validation")
    }

    pub fn test() -> Self {
        Self::new("test")
    }

    pub fn new(split: &str) -> Self {
        let mut rdr = csv::ReaderBuilder::new();
        let rdr = rdr.delimiter(b',');
        let file_name = match split.as_ref() {
            "train" => "data/queuemetrics-100000.csv",
            "validation" => "data/queuemetrics-20000.csv",
            "test" => "data/queuemetrics-1000.csv",
            _ => "data/queuemetrics-100000.csv",
        };
        let dataset = InMemDataset::from_csv(file_name, &rdr).unwrap();
        Self { dataset }
    }
}

// must implement get and len
impl Dataset<QueueMetrics> for QueueMetricsDataset {
    fn get(&self, index: usize) -> Option<QueueMetrics> {
        self.dataset.get(index)
    }

    fn len(&self) -> usize {
        self.dataset.len()
    }
}

/// Normalizer for the metrics dataset.
#[derive(Clone, Debug)]
pub struct Normalizer<B: Backend> {
    pub min: Tensor<B, 2>,
    pub max: Tensor<B, 2>,
}

impl<B: Backend> Normalizer<B> {
    /// Creates a new normalizer.
    pub fn new(device: &B::Device, min: &[f32], max: &[f32]) -> Self {
        let min = Tensor::<B, 1>::from_floats(min, device).unsqueeze();
        let max = Tensor::<B, 1>::from_floats(max, device).unsqueeze();
        Self { min, max }
    }

    /// Normalizes the input according to the vital signs data set min/max.
    pub fn normalize(&self, input: Tensor<B, 2>) -> Tensor<B, 2> {
        (input - self.min.clone()) / (self.max.clone() - self.min.clone())
    }
}

#[derive(Clone, Debug)]
pub struct QueueMetricsBatcher<B: Backend> {
    device: B::Device,
    normalizer: Normalizer<B>,
}

#[derive(Clone, Debug)]
pub struct QueueMetricsBatch<B: Backend> {
    pub inputs: Tensor<B, 2>,
    pub targets: Tensor<B, 1, Int>,
}

impl<B: Backend> QueueMetricsBatcher<B> {
    pub fn new(device: B::Device) -> Self {
        Self {
            device: device.clone(),
            normalizer: Normalizer::new(&device, &FEATURES_MIN, &FEATURES_MAX),
        }
    }
}

impl<B: Backend> Batcher<QueueMetrics, QueueMetricsBatch<B>> for QueueMetricsBatcher<B> {
    fn batch(&self, items: Vec<QueueMetrics>) -> QueueMetricsBatch<B> {
        let mut inputs: Vec<Tensor<B, 2>> = Vec::new();
        for item in items.iter() {
            let input_tensor = Tensor::<B, 1>::from_floats(
                [
                    item.processors,
                    item.avg_batch_size,
                    item.queue_length,
                    item.processing_time,
                ],
                &self.device,
            );

            inputs.push(input_tensor.unsqueeze());
        }

        let inputs = Tensor::cat(inputs, 0);
        let inputs = self.normalizer.normalize(inputs);
        let mut targets: Vec<Tensor<B, 1, Int>> = Vec::new();
        for item in items.iter() {
            let target_tensor = Tensor::<B, 1, Int>::from_data([item.status], &self.device);
            targets.push(target_tensor.unsqueeze());
        }

        let targets = Tensor::cat(targets, 0);
        QueueMetricsBatch { inputs, targets }
    }
}
