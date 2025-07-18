use burn::{
    data::dataloader::batcher::Batcher,
    module::Module,
    record::{CompactRecorder, Recorder},
};

use burn_cuda::{Cuda, CudaDevice};

use crate::{
    dataset::{QueueMetrics, QueueMetricsBatcher, NUM_CLASSES, NUM_FEATURES},
    model::{ModelConfig, ModelRecord},
    MAP_LOOKUP,
};

use custom_logger as log;
use http::{Method, Request, Response, StatusCode};
use http_body_util::{BodyExt, Full};
use hyper::body::{Bytes, Incoming};

#[derive(Clone, Debug, serde::Serialize, serde::Deserialize)]
pub struct InferenceResponse {
    #[serde(rename = "expected")]
    pub expected: usize,
    #[serde(rename = "predicted")]
    pub predicted: usize,
}

// inference endpoint
pub async fn inference_service(
    req: Request<Incoming>,
) -> Result<Response<Full<Bytes>>, hyper::Error> {
    let mut response = Response::new(Full::default());
    let hm = MAP_LOOKUP.lock().unwrap().clone();
    let artifact_dir = hm.unwrap().get("artifact_dir").unwrap().clone();
    let device = CudaDevice::default();
    type MyBackend = Cuda<f32, i32>;
    let record: ModelRecord<MyBackend> = CompactRecorder::new()
        .load(format!("{artifact_dir}/model").into(), &device)
        .expect("trained model should exist; run train first");
    // parameters are features,classes,hidden size
    let model = ModelConfig::new(NUM_FEATURES, NUM_CLASSES, 256)
        .init(&device)
        .load_record(record);
    let batcher = QueueMetricsBatcher::new(device.clone());

    match (req.method(), req.uri().path()) {
        // inference.
        (&Method::POST, "/inference") => {
            let data = req.into_body().collect().await?.to_bytes();
            let qm: QueueMetrics = serde_json::from_slice(&data).unwrap();
            log::debug!("queuemetrics {:?}", qm);
            let items = vec![qm];
            let batch = batcher.batch(items.clone(), &device);
            let predicted = model.forward(batch.inputs.clone());
            let targets = batch.targets;
            let expected = targets.into_data().iter::<f32>().collect::<Vec<_>>();
            let predicted = predicted
                .iter_dim(0)
                .map(|item| item.into_data().into_vec::<f32>())
                .collect::<Vec<_>>();
            let (predicted_max_index, _) = find_max_index(predicted[0].as_ref().unwrap());
            log::info!(
                "expected {} : predicted {}",
                expected[0],
                predicted_max_index
            );
            let ir = InferenceResponse {
                expected: expected[0] as usize,
                predicted: predicted_max_index,
            };
            *response.body_mut() = Full::from(serde_json::to_string(&ir).unwrap());
        }
        _ => {
            *response.status_mut() = StatusCode::NOT_FOUND;
        }
    };
    Ok(response)
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
