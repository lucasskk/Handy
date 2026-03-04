use anyhow::{anyhow, Context, Result};
use log::{info, warn};
use parakeet_rs::{ExecutionConfig, ExecutionProvider, ParakeetTDT, TimestampMode, Transcriber};
use std::path::Path;

#[derive(Clone, Copy, Debug)]
enum ProviderKind {
    Cuda,
    DirectML,
    Cpu,
}

impl ProviderKind {
    fn as_execution_provider(self) -> ExecutionProvider {
        match self {
            ProviderKind::Cuda => ExecutionProvider::Cuda,
            ProviderKind::DirectML => ExecutionProvider::DirectML,
            ProviderKind::Cpu => ExecutionProvider::Cpu,
        }
    }

    fn as_str(self) -> &'static str {
        match self {
            ProviderKind::Cuda => "cuda",
            ProviderKind::DirectML => "directml",
            ProviderKind::Cpu => "cpu",
        }
    }
}

pub struct ParakeetWindowsEngine {
    model: ParakeetTDT,
    provider: ProviderKind,
}

impl ParakeetWindowsEngine {
    pub fn load_from_model_dir(model_dir: &Path) -> Result<Self> {
        if !model_dir.is_dir() {
            return Err(anyhow!(
                "Parakeet model path is not a directory: {}",
                model_dir.display()
            ));
        }

        let providers = provider_attempt_order()?;
        let intra_threads = default_intra_threads();
        let mut load_errors = Vec::new();

        for provider in providers {
            let config = ExecutionConfig::new()
                .with_execution_provider(provider.as_execution_provider())
                .with_intra_threads(intra_threads)
                .with_inter_threads(1);

            match ParakeetTDT::from_pretrained(model_dir, Some(config)) {
                Ok(model) => {
                    info!(
                        "Loaded Parakeet Windows backend with {} provider (intra_threads={})",
                        provider.as_str(),
                        intra_threads
                    );
                    return Ok(Self { model, provider });
                }
                Err(error) => {
                    let message = format!("{}: {}", provider.as_str(), error);
                    warn!(
                        "Failed to initialize Parakeet Windows backend with {} provider: {}",
                        provider.as_str(),
                        error
                    );
                    load_errors.push(message);
                }
            }
        }

        Err(anyhow!(
            "Failed to initialize Parakeet Windows backend. Attempts: {}",
            load_errors.join("; ")
        ))
    }

    pub fn provider_name(&self) -> &'static str {
        self.provider.as_str()
    }

    pub fn transcribe_samples(&mut self, samples: Vec<f32>) -> Result<String> {
        if samples.is_empty() {
            return Ok(String::new());
        }

        let result = self
            .model
            .transcribe_samples(samples, 16_000, 1, Some(TimestampMode::Sentences))
            .context("Parakeet Windows transcription failed")?;

        Ok(result.text)
    }
}

fn provider_attempt_order() -> Result<Vec<ProviderKind>> {
    let provider = std::env::var("HANDY_PARAKEET_WINDOWS_PROVIDER")
        .unwrap_or_else(|_| "auto".to_string())
        .trim()
        .to_ascii_lowercase();

    let gpu_required = env_flag("HANDY_PARAKEET_GPU_REQUIRED", false);

    let providers = match provider.as_str() {
        "auto" => {
            if gpu_required {
                vec![ProviderKind::Cuda, ProviderKind::DirectML]
            } else {
                vec![
                    ProviderKind::Cuda,
                    ProviderKind::DirectML,
                    ProviderKind::Cpu,
                ]
            }
        }
        "cuda" => vec![ProviderKind::Cuda],
        "directml" | "dml" => vec![ProviderKind::DirectML],
        "cpu" => {
            if gpu_required {
                return Err(anyhow!(
                    "HANDY_PARAKEET_GPU_REQUIRED=true cannot be used with HANDY_PARAKEET_WINDOWS_PROVIDER=cpu"
                ));
            }
            vec![ProviderKind::Cpu]
        }
        other => {
            return Err(anyhow!(
                "Invalid HANDY_PARAKEET_WINDOWS_PROVIDER='{}'. Expected one of: auto, cuda, directml, cpu",
                other
            ));
        }
    };

    Ok(providers)
}

fn default_intra_threads() -> usize {
    std::thread::available_parallelism()
        .map(|n| std::cmp::max(1, n.get() / 2))
        .unwrap_or(4)
}

fn env_flag(name: &str, default: bool) -> bool {
    let Some(value) = std::env::var(name).ok() else {
        return default;
    };

    match value.trim().to_ascii_lowercase().as_str() {
        "1" | "true" | "yes" | "on" => true,
        "0" | "false" | "no" | "off" => false,
        _ => default,
    }
}
