use burn::{
    data::{dataloader::batcher::Batcher, dataset::Dataset},
    module::Module,
    record::{CompactRecorder, Recorder},
    tensor::backend::Backend,
};

use crate::{
    dataset::{QueueMetrics, QueueMetricsBatcher, QueueMetricsDataset, NUM_CLASSES, NUM_FEATURES},
    model::{ModelConfig, ModelRecord},
};

use custom_logger as log;

pub fn infer<B: Backend>(artifacts_dir: &str, device: B::Device) {
    // parameters are features,classes,hidden size
    let record: ModelRecord<B> = CompactRecorder::new()
        .load(format!("{artifacts_dir}/model").into(), &device)
        .expect("Trained model should exist; run train first");
    let model = ModelConfig::new(NUM_FEATURES, NUM_CLASSES, 256)
        .init(&device)
        .load_record(record);

    // Use a sample of 10 items from the test split
    let dataset = QueueMetricsDataset::test();
    let items: Vec<QueueMetrics> = dataset.iter().take(1000).collect();

    let batcher = QueueMetricsBatcher::new(device.clone());
    let batch = batcher.batch(items.clone(), &device);
    let predicted = model.forward(batch.inputs.clone());
    let targets = batch.targets;

    let expected = targets.into_data().iter::<f32>().collect::<Vec<_>>();
    let predicted = predicted
        .iter_dim(0)
        .map(|item| item.into_data().into_vec::<f32>())
        .collect::<Vec<_>>();

    let mut count = 0;
    let mut correct = 0;
    for item in predicted.iter() {
        let (index, _value) = find_max_index(item.as_ref().unwrap());
        let fmt_expected: String = format!("{}", expected[count] as usize);
        let fmt_predicted: String = format!("{}", index);
        if fmt_expected.eq(&fmt_predicted) {
            correct += 1;
        }
        log::debug!(
            "count {:0>4} : predicted {} : expected {} : {:?} ",
            count,
            fmt_predicted,
            fmt_expected,
            predicted[count]
        );
        count += 1;
    }
    log::info!(
        "summary total tests {} : correct {}%",
        count,
        (correct as f32 / count as f32) * 100.0
    );
}

fn find_max_index(input: &Vec<f32>) -> (usize, f32) {
    let mut max_index = 0;
    let mut index = 0;
    let mut max = 0.0f32;
    for item in input.iter() {
        if item > &max {
            max_index = index;
            max = *item;
        }
        index += 1;
    }
    (max_index, max)
}
