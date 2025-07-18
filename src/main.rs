use crate::certhandler::{error, CertificateInterface, ImplCertificateInterface};
use crate::serverconfig::{ConfigInterface, ImplConfigInterface, Parameters};
use burn_autodiff::Autodiff;
use burn_cuda::{Cuda, CudaDevice};
use clap::{Parser, Subcommand};
use custom_logger as log;
use hyper::service::service_fn;
use hyper_util::rt::{TokioExecutor, TokioIo};
use hyper_util::server::conn::auto::Builder;
use rustls::ServerConfig;
use server::inference_service;
use std::collections::HashMap;
use std::env;
use std::net::{Ipv4Addr, SocketAddr};
use std::sync::Arc;
use std::sync::Mutex;
use tokio::net::TcpListener;
use tokio_rustls::TlsAcceptor;

mod certhandler;
mod dataset;
mod inference;
mod model;
mod server;
mod serverconfig;
mod training;

/// cli struct
#[derive(Parser, Debug)]
#[command(name = env!("CARGO_PKG_NAME"))]
#[command(author = env!("CARGO_PKG_AUTHORS"))]
#[command(version = env!("CARGO_PKG_VERSION"))]
#[command(about = "Simple neural network to train, infer and serve queuemetrics", long_about = None)]
#[command(author, version, about, long_about = None)]
pub struct Cli {
    #[command(subcommand)]
    pub command: Option<Commands>,
    /// config file to use
    #[arg(short, long, value_name = "config")]
    pub config: String,
}

#[derive(Subcommand, Debug)]
pub enum Commands {
    /// Train subcommand
    Train {},
    /// Inference subcommand
    Inference {},
    /// Serve inference subcommand (launches json web service)
    Serve {},
}

// used for lookup in read mode only
static MAP_LOOKUP: Mutex<Option<HashMap<String, String>>> = Mutex::new(None);

fn main() {
    let args = Cli::parse();
    let config = args.config;
    let impl_config = ImplConfigInterface {};
    let params = impl_config.read(config).unwrap();
    let device = CudaDevice::default();
    type MyBackend = Cuda<f32, i32>;

    match &args.command {
        Some(Commands::Train {}) => {
            type MyAutodiffBackend = Autodiff<MyBackend>;
            training::run::<MyAutodiffBackend>(&params.artifacts_dir, device.clone());
        }
        Some(Commands::Inference {}) => {
            // use logging only for the inference and web service
            // setup logging
            let level = match params.log_level.as_str() {
                "info" => log::LevelFilter::Info,
                "debug" => log::LevelFilter::Debug,
                "trace" => log::LevelFilter::Trace,
                &_ => log::LevelFilter::Info,
            };

            log::Logging::new()
                .with_level(level)
                .init()
                .expect("should initialize");

            inference::infer::<MyBackend>(&params.artifacts_dir, device);
        }
        Some(Commands::Serve {}) => {
            // use logging only for the inference and web service
            // setup logging
            let level = match params.log_level.as_str() {
                "info" => log::LevelFilter::Info,
                "debug" => log::LevelFilter::Debug,
                "trace" => log::LevelFilter::Trace,
                &_ => log::LevelFilter::Info,
            };

            log::Logging::new()
                .with_level(level)
                .init()
                .expect("should initialize");

            let mut hm: HashMap<String, String> = HashMap::new();
            hm.insert("artifact_dir".to_string(), params.artifacts_dir.clone());
            *MAP_LOOKUP.lock().unwrap() = Some(hm.clone());

            if let Err(e) = run_server(params) {
                log::error!("{}", e);
                std::process::exit(1);
            }
        }
        None => {
            log::error!("unknown command");
            std::process::exit(1);
        }
    }
}

#[tokio::main]
async fn run_server(params: Parameters) -> Result<(), Box<dyn std::error::Error + Send + Sync>> {
    let addr = SocketAddr::new(
        Ipv4Addr::new(0, 0, 0, 0).into(),
        params.port.parse().unwrap(),
    );

    rustls::crypto::ring::default_provider()
        .install_default()
        .expect("Failed to install rustls crypto provider");
    let certs_dir = params.certs_dir.unwrap_or("".to_string()).to_string();
    let impl_certs = ImplCertificateInterface::new(params.cert_mode, Some(certs_dir));
    // Load public certificate.
    let certs = impl_certs.get_public_cert().await.unwrap();
    // Load private key.
    let key = impl_certs.get_private_cert().await.unwrap();
    log::info!("starting {} on https://{}", params.name, addr);
    // Create a TCP listener via tokio.
    let incoming = TcpListener::bind(&addr).await?;
    // Build TLS configuration.
    let mut server_config = ServerConfig::builder()
        .with_no_client_auth()
        .with_single_cert(certs, key)
        .map_err(|e| error(e.to_string()))?;
    server_config.alpn_protocols = vec![b"h2".to_vec(), b"http/1.1".to_vec(), b"http/1.0".to_vec()];
    let tls_acceptor = TlsAcceptor::from(Arc::new(server_config));
    let service = service_fn(inference_service);

    loop {
        let (tcp_stream, _remote_addr) = incoming.accept().await?;
        let tls_acceptor = tls_acceptor.clone();
        tokio::spawn(async move {
            let tls_stream = match tls_acceptor.accept(tcp_stream).await {
                Ok(tls_stream) => tls_stream,
                Err(err) => {
                    log::error!("failed to perform tls handshake: {err:#}");
                    return;
                }
            };
            if let Err(err) = Builder::new(TokioExecutor::new())
                .serve_connection(TokioIo::new(tls_stream), service)
                .await
            {
                log::error!("failed to serve connection: {err:#}");
            }
        });
    }
}
