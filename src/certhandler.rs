use async_trait::async_trait;
use rustls_pki_types::{CertificateDer, PrivateKeyDer};
use std::fs;
use std::io;

#[async_trait]
pub trait CertificateInterface {
    fn new(mode: String, cert_dir: Option<String>) -> Self;
    async fn get_public_cert(&self) -> io::Result<Vec<CertificateDer<'static>>>;
    async fn get_private_cert(&self) -> io::Result<PrivateKeyDer<'static>>;
}

pub struct ImplCertificateInterface {
    mode: String,
    certs_dir: Option<String>,
}

#[async_trait]
impl CertificateInterface for ImplCertificateInterface {
    fn new(mode: String, certs_dir: Option<String>) -> Self {
        return ImplCertificateInterface { mode, certs_dir };
    }

    async fn get_public_cert(&self) -> io::Result<Vec<CertificateDer<'static>>> {
        match self.mode.as_str() {
            "aws" => load_public_key(format!("{}/ssl.cert", self.certs_dir.as_ref().unwrap())),
            "file" => load_public_key(format!("{}/ssl.cert", self.certs_dir.as_ref().unwrap())),
            &_ => return Err(error(format!("mode {} not available", self.mode))),
        }
    }

    async fn get_private_cert(&self) -> io::Result<PrivateKeyDer<'static>> {
        match self.mode.as_str() {
            "aws" => load_private_key(format!("{}/ssl.key", self.certs_dir.as_ref().unwrap())),
            "file" => load_private_key(format!("{}/ssl.key", self.certs_dir.as_ref().unwrap())),
            &_ => return Err(error(format!("mode {} not available", self.mode))),
        }
    }
}

// Load public certificate from file.
fn load_public_key(dir: String) -> io::Result<Vec<CertificateDer<'static>>> {
    // Open certificate file.
    let certfile = fs::File::open(dir)
        .map_err(|e| error(format!("failed to open {}: {}", "./certs/ssl.cert", e)))?;
    let mut reader = io::BufReader::new(certfile);

    // Load and return certificate.
    rustls_pemfile::certs(&mut reader).collect()
}

// Load private key from file.
fn load_private_key(dir: String) -> io::Result<PrivateKeyDer<'static>> {
    // Open keyfile.
    let keyfile = fs::File::open(dir)
        .map_err(|e| error(format!("failed to open {}: {}", "./certs/ssl.key", e)))?;
    let mut reader = io::BufReader::new(keyfile);

    // Load and return a single private key.
    rustls_pemfile::private_key(&mut reader).map(|key| key.unwrap())
}

pub fn error(err: String) -> io::Error {
    io::Error::new(io::ErrorKind::Other, err)
}

