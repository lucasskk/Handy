use anyhow::{anyhow, Context, Result};
use log::{info, warn};
use parakeet_rs::{ExecutionConfig, ExecutionProvider, ParakeetTDT, TimestampMode, Transcriber};
use std::env;
use std::fs;
use std::path::{Path, PathBuf};
use tauri::{path::BaseDirectory, AppHandle, Manager};

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
    attempt_errors: Vec<String>,
}

impl ParakeetWindowsEngine {
    pub fn load_from_model_dir(app_handle: &AppHandle, model_dir: &Path) -> Result<Self> {
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
            if let Some(preflight_error) = provider_preflight_error(provider, app_handle) {
                warn!(
                    "Skipping Parakeet Windows backend {} provider: {}",
                    provider.as_str(),
                    preflight_error
                );
                load_errors.push(format!("{}: {}", provider.as_str(), preflight_error));
                continue;
            }

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
                    if !load_errors.is_empty() {
                        warn!(
                            "Parakeet Windows backend fallback path used: {}",
                            load_errors.join("; ")
                        );
                    }
                    return Ok(Self {
                        model,
                        provider,
                        attempt_errors: load_errors,
                    });
                }
                Err(error) => {
                    let error_message = error.to_string();
                    let message = format!(
                        "{} (runtime): {}",
                        provider.as_str(),
                        summarize_provider_runtime_error(provider, &error_message)
                    );
                    warn!(
                        "Failed to initialize Parakeet Windows backend with {} provider: {}",
                        provider.as_str(),
                        error_message
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

    pub fn backend_details(&self) -> Option<String> {
        if self.attempt_errors.is_empty() {
            None
        } else {
            Some(format!(
                "Provider fallback to {} after: {}",
                self.provider.as_str(),
                self.attempt_errors.join("; ")
            ))
        }
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
                vec![ProviderKind::Cuda]
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

fn provider_preflight_error(provider: ProviderKind, app_handle: &AppHandle) -> Option<String> {
    match provider {
        ProviderKind::Cuda => ensure_cuda_runtime_dlls(app_handle).err(),
        _ => None,
    }
}

fn ensure_cuda_runtime_dlls(app_handle: &AppHandle) -> std::result::Result<(), String> {
    const REQUIRED_PROVIDER_DLLS: [&str; 2] = [
        "onnxruntime_providers_shared.dll",
        "onnxruntime_providers_cuda.dll",
    ];
    const REQUIRED_CUDA_RUNTIME_DLLS: [&str; 5] = [
        "cudnn64_9.dll",
        "cublas64_12.dll",
        "cublasLt64_12.dll",
        "cudart64_12.dll",
        "cufft64_11.dll",
    ];

    let runtime_dirs = onnxruntime_runtime_dirs(app_handle);
    let Some(runtime_dir) = runtime_dirs.first() else {
        return Err("unable to resolve runtime directory for ONNX Runtime CUDA DLLs".to_string());
    };

    let source_dirs = onnxruntime_source_dirs(app_handle, runtime_dir);

    configure_cuda_dependency_paths(runtime_dir, &source_dirs);

    let mut missing = missing_dlls_in_dir(runtime_dir, &REQUIRED_PROVIDER_DLLS);
    if missing.is_empty() {
        let search_dirs = cuda_dependency_dirs(runtime_dir, &source_dirs);
        let missing_runtime_dependencies =
            missing_dlls_across_dirs(&search_dirs, &REQUIRED_CUDA_RUNTIME_DLLS);
        if missing_runtime_dependencies.is_empty() {
            return Ok(());
        }

        return Err(format!(
            "missing CUDA runtime dependencies ({}) while searching [{}]",
            missing_runtime_dependencies.join(", "),
            format_search_dirs(&search_dirs)
        ));
    }

    for source_dir in &source_dirs {
        if source_dir.as_path() == runtime_dir {
            continue;
        }

        if !missing.iter().all(|dll| source_dir.join(dll).is_file()) {
            continue;
        }

        for dll in &missing {
            let source_path = source_dir.join(dll);
            let destination_path = runtime_dir.join(dll);

            if destination_path.is_file() {
                continue;
            }

            fs::copy(&source_path, &destination_path).map_err(|error| {
                format!(
                    "failed to copy {} from {} to {}: {}",
                    dll,
                    source_dir.display(),
                    runtime_dir.display(),
                    error
                )
            })?;

            info!(
                "Copied ONNX Runtime CUDA DLL {} from {} to {}",
                dll,
                source_dir.display(),
                runtime_dir.display()
            );
        }

        missing = missing_dlls_in_dir(runtime_dir, &REQUIRED_PROVIDER_DLLS);
        if missing.is_empty() {
            let search_dirs = cuda_dependency_dirs(runtime_dir, &source_dirs);
            let missing_runtime_dependencies =
                missing_dlls_across_dirs(&search_dirs, &REQUIRED_CUDA_RUNTIME_DLLS);
            if missing_runtime_dependencies.is_empty() {
                return Ok(());
            }

            return Err(format!(
                "missing CUDA runtime dependencies ({}) while searching [{}]",
                missing_runtime_dependencies.join(", "),
                format_search_dirs(&search_dirs)
            ));
        }
    }

    let searched_provider_dirs = {
        let mut dirs = vec![runtime_dir.to_path_buf()];
        for source_dir in &source_dirs {
            push_unique_any_dir(&mut dirs, source_dir.clone());
        }
        dirs
    };

    Err(format!(
        "missing ONNX Runtime CUDA DLLs ({}) in {} while searching [{}]. Ensure they are available in {} or set HANDY_ORT_DLL_DIR/ORT_LIB_LOCATION",
        missing.join(", "),
        runtime_dir.display(),
        format_search_dirs(&searched_provider_dirs),
        runtime_dir.join("resources").join("onnxruntime").display()
    ))
}

fn configure_cuda_dependency_paths(runtime_dir: &Path, source_dirs: &[PathBuf]) {
    let candidate_dirs = cuda_dependency_dirs(runtime_dir, source_dirs);
    if candidate_dirs.is_empty() {
        return;
    }

    let mut existing_path_dirs = env::var_os("PATH")
        .map(|value| env::split_paths(&value).collect::<Vec<_>>())
        .unwrap_or_default();

    let mut added = Vec::new();

    for candidate in candidate_dirs.into_iter().rev() {
        if existing_path_dirs
            .iter()
            .any(|existing| paths_match(existing, &candidate))
        {
            continue;
        }

        added.push(candidate.clone());
        existing_path_dirs.insert(0, candidate);
    }

    if added.is_empty() {
        return;
    }

    match env::join_paths(existing_path_dirs) {
        Ok(new_path) => {
            env::set_var("PATH", &new_path);
            let added_paths = added
                .iter()
                .map(|path| path.display().to_string())
                .collect::<Vec<_>>()
                .join("; ");
            info!("Configured CUDA dependency search paths: {}", added_paths);
        }
        Err(error) => {
            warn!(
                "Failed to update PATH for CUDA dependency search paths: {}",
                error
            );
        }
    }
}

fn cuda_dependency_dirs(runtime_dir: &Path, source_dirs: &[PathBuf]) -> Vec<PathBuf> {
    let mut directories = Vec::new();

    if let Ok(custom_dirs) = env::var("HANDY_CUDA_DLL_DIR") {
        for path in env::split_paths(&custom_dirs) {
            push_unique_dir(&mut directories, path);
        }
    }

    if let Ok(cuda_path) = env::var("CUDA_PATH") {
        push_unique_dir(&mut directories, PathBuf::from(cuda_path).join("bin"));
    }

    for (key, value) in env::vars() {
        if key.starts_with("CUDA_PATH_V") {
            push_unique_dir(&mut directories, PathBuf::from(value).join("bin"));
        }
    }

    if let Some(program_files) = env::var_os("ProgramFiles") {
        collect_cudnn_bin_dirs(
            &PathBuf::from(program_files).join("NVIDIA").join("CUDNN"),
            &mut directories,
        );
    }

    for source_dir in source_dirs {
        push_unique_dir(&mut directories, source_dir.clone());
    }

    push_unique_dir(
        &mut directories,
        runtime_dir.join("resources").join("onnxruntime"),
    );
    push_unique_dir(&mut directories, runtime_dir.to_path_buf());

    if let Some(path_value) = env::var_os("PATH") {
        for path in env::split_paths(&path_value) {
            push_unique_dir(&mut directories, path);
        }
    }

    directories
}

fn collect_cudnn_bin_dirs(root: &Path, directories: &mut Vec<PathBuf>) {
    if !root.is_dir() {
        return;
    }

    let Ok(version_dirs) = fs::read_dir(root) else {
        return;
    };

    for version_entry in version_dirs.flatten() {
        let version_path = version_entry.path();
        let bin_root = version_path.join("bin");
        if !bin_root.is_dir() {
            continue;
        }

        let Ok(bin_dirs) = fs::read_dir(&bin_root) else {
            continue;
        };

        for bin_entry in bin_dirs.flatten() {
            let bin_path = bin_entry.path();
            if !bin_path.is_dir() {
                continue;
            }

            push_unique_dir(directories, bin_path.clone());
            push_unique_dir(directories, bin_path.join("x64"));
        }
    }
}

fn paths_match(a: &Path, b: &Path) -> bool {
    if cfg!(windows) {
        a.to_string_lossy()
            .eq_ignore_ascii_case(&b.to_string_lossy())
    } else {
        a == b
    }
}

fn missing_dlls_in_dir(directory: &Path, required_dlls: &[&'static str]) -> Vec<&'static str> {
    required_dlls
        .iter()
        .copied()
        .filter(|dll| !directory.join(dll).is_file())
        .collect()
}

fn missing_dlls_across_dirs(
    directories: &[PathBuf],
    required_dlls: &[&'static str],
) -> Vec<&'static str> {
    required_dlls
        .iter()
        .copied()
        .filter(|dll| !directories.iter().any(|dir| dir.join(dll).is_file()))
        .collect()
}

fn onnxruntime_source_dirs(app_handle: &AppHandle, runtime_dir: &Path) -> Vec<PathBuf> {
    let mut directories = Vec::new();

    for runtime_candidate in onnxruntime_runtime_dirs(app_handle) {
        push_unique_dir(&mut directories, runtime_candidate);
    }

    if let Ok(custom_dir) = std::env::var("HANDY_ORT_DLL_DIR") {
        push_unique_dir(&mut directories, PathBuf::from(custom_dir));
    }

    if let Ok(ort_lib_location) = std::env::var("ORT_LIB_LOCATION") {
        let base = PathBuf::from(ort_lib_location);
        push_unique_dir(&mut directories, base.clone());
        push_unique_dir(&mut directories, base.join("lib"));
    }

    push_unique_any_dir(&mut directories, runtime_dir.to_path_buf());
    push_unique_dir(&mut directories, runtime_dir.join("resources"));
    push_unique_dir(
        &mut directories,
        runtime_dir.join("resources").join("onnxruntime"),
    );
    push_unique_dir(&mut directories, runtime_dir.join("onnxruntime"));

    if let Ok(current_dir) = std::env::current_dir() {
        push_unique_dir(&mut directories, current_dir.clone());
        push_unique_dir(&mut directories, current_dir.join("resources"));
        push_unique_dir(
            &mut directories,
            current_dir.join("resources").join("onnxruntime"),
        );
        push_unique_dir(
            &mut directories,
            current_dir
                .join("src-tauri")
                .join("resources")
                .join("onnxruntime"),
        );
    }

    directories
}

fn push_unique_dir(directories: &mut Vec<PathBuf>, candidate: PathBuf) {
    if candidate.is_dir() && !directories.iter().any(|existing| existing == &candidate) {
        directories.push(candidate);
    }
}

fn push_unique_any_dir(directories: &mut Vec<PathBuf>, candidate: PathBuf) {
    if !directories
        .iter()
        .any(|existing| paths_match(existing, &candidate))
    {
        directories.push(candidate);
    }
}

fn onnxruntime_runtime_dirs(app_handle: &AppHandle) -> Vec<PathBuf> {
    let mut dirs = Vec::new();

    if let Ok(resource_dir) = app_handle
        .path()
        .resolve("resources/onnxruntime", BaseDirectory::Resource)
    {
        push_unique_dir(&mut dirs, resource_dir);
    }

    if let Ok(custom_dir) = std::env::var("HANDY_ORT_DLL_DIR") {
        push_unique_dir(&mut dirs, PathBuf::from(custom_dir));
    }

    if let Ok(ort_lib_location) = std::env::var("ORT_LIB_LOCATION") {
        let base = PathBuf::from(ort_lib_location);
        push_unique_dir(&mut dirs, base.clone());
        push_unique_dir(&mut dirs, base.join("lib"));
    }

    if let Ok(exe_path) = std::env::current_exe() {
        if let Some(parent) = exe_path.parent() {
            push_unique_dir(&mut dirs, parent.to_path_buf());
            push_unique_dir(&mut dirs, parent.join("resources").join("onnxruntime"));
            push_unique_dir(&mut dirs, parent.join("onnxruntime"));
        }
    }

    if let Ok(current_dir) = std::env::current_dir() {
        push_unique_dir(&mut dirs, current_dir.clone());
        push_unique_dir(&mut dirs, current_dir.join("resources").join("onnxruntime"));
        push_unique_dir(
            &mut dirs,
            current_dir
                .join("src-tauri")
                .join("resources")
                .join("onnxruntime"),
        );
    }

    dirs
}

fn summarize_provider_runtime_error(provider: ProviderKind, error: &str) -> String {
    if matches!(provider, ProviderKind::Cuda) {
        if let Some(missing_dependency) = extract_missing_dependency(error) {
            return format!(
                "missing CUDA runtime dependency {} (install cuDNN/CUDA runtimes and ensure DLL paths are discoverable)",
                missing_dependency
            );
        }

        return format!("CUDA provider load failure: {}", error);
    }

    error.to_string()
}

fn format_search_dirs(directories: &[PathBuf]) -> String {
    if directories.is_empty() {
        return "<none>".to_string();
    }

    directories
        .iter()
        .map(|path| path.display().to_string())
        .collect::<Vec<_>>()
        .join("; ")
}

fn extract_missing_dependency(error: &str) -> Option<String> {
    let marker = "depends on \"";
    let start = error.find(marker)? + marker.len();
    let rest = &error[start..];
    let end = rest.find('"')?;
    let dependency = rest[..end].trim();

    if dependency.is_empty() {
        None
    } else {
        Some(dependency.to_string())
    }
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
