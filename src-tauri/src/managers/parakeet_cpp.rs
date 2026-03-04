use anyhow::{anyhow, Context, Result};
use libloading::Library;
use log::{debug, warn};
use std::ffi::{c_char, c_int, c_void, CStr, CString};
use std::fs;
use std::path::{Path, PathBuf};

type ParakeetTdtTranscriber = *mut c_void;
type ParakeetResult = *mut c_void;
type ParakeetConfig = *mut c_void;
type ParakeetOptions = *mut c_void;

type TdtCreateFn = unsafe extern "C" fn(
    weights: *const c_char,
    vocab: *const c_char,
    config: ParakeetConfig,
) -> ParakeetTdtTranscriber;
type TdtFreeFn = unsafe extern "C" fn(transcriber: ParakeetTdtTranscriber);
type TdtToGpuFn = unsafe extern "C" fn(transcriber: ParakeetTdtTranscriber) -> c_int;
type TdtToHalfFn = unsafe extern "C" fn(transcriber: ParakeetTdtTranscriber) -> c_int;
type TdtTranscribePcmFn = unsafe extern "C" fn(
    transcriber: ParakeetTdtTranscriber,
    samples: *const f32,
    n: usize,
    opts: ParakeetOptions,
) -> ParakeetResult;
type ResultTextFn = unsafe extern "C" fn(result: ParakeetResult) -> *const c_char;
type ResultFreeFn = unsafe extern "C" fn(result: ParakeetResult);
type LastErrorFn = unsafe extern "C" fn() -> *const c_char;

struct ParakeetApi {
    tdt_transcriber_create: TdtCreateFn,
    tdt_transcriber_free: TdtFreeFn,
    tdt_transcriber_to_gpu: TdtToGpuFn,
    tdt_transcriber_to_half: TdtToHalfFn,
    tdt_transcriber_transcribe_pcm: TdtTranscribePcmFn,
    result_text: ResultTextFn,
    result_free: ResultFreeFn,
    last_error: LastErrorFn,
}

impl ParakeetApi {
    unsafe fn from_library(library: &Library) -> Result<Self> {
        Ok(Self {
            tdt_transcriber_create: unsafe {
                *library
                    .get::<TdtCreateFn>(b"parakeet_tdt_transcriber_create")
                    .context("Failed to load symbol: parakeet_tdt_transcriber_create")?
            },
            tdt_transcriber_free: unsafe {
                *library
                    .get::<TdtFreeFn>(b"parakeet_tdt_transcriber_free")
                    .context("Failed to load symbol: parakeet_tdt_transcriber_free")?
            },
            tdt_transcriber_to_gpu: unsafe {
                *library
                    .get::<TdtToGpuFn>(b"parakeet_tdt_transcriber_to_gpu")
                    .context("Failed to load symbol: parakeet_tdt_transcriber_to_gpu")?
            },
            tdt_transcriber_to_half: unsafe {
                *library
                    .get::<TdtToHalfFn>(b"parakeet_tdt_transcriber_to_half")
                    .context("Failed to load symbol: parakeet_tdt_transcriber_to_half")?
            },
            tdt_transcriber_transcribe_pcm: unsafe {
                *library
                    .get::<TdtTranscribePcmFn>(b"parakeet_tdt_transcriber_transcribe_pcm")
                    .context("Failed to load symbol: parakeet_tdt_transcriber_transcribe_pcm")?
            },
            result_text: unsafe {
                *library
                    .get::<ResultTextFn>(b"parakeet_result_text")
                    .context("Failed to load symbol: parakeet_result_text")?
            },
            result_free: unsafe {
                *library
                    .get::<ResultFreeFn>(b"parakeet_result_free")
                    .context("Failed to load symbol: parakeet_result_free")?
            },
            last_error: unsafe {
                *library
                    .get::<LastErrorFn>(b"parakeet_last_error")
                    .context("Failed to load symbol: parakeet_last_error")?
            },
        })
    }
}

pub struct ParakeetCppEngine {
    _library: Library,
    api: ParakeetApi,
    transcriber: ParakeetTdtTranscriber,
}

impl ParakeetCppEngine {
    pub fn load_from_model_dir(model_dir: &Path) -> Result<Self> {
        if env_flag("HANDY_PARAKEET_CPP_DISABLE", false) {
            return Err(anyhow!(
                "Parakeet.cpp backend is disabled via HANDY_PARAKEET_CPP_DISABLE"
            ));
        }

        let (weights_path, vocab_path) = resolve_model_files(model_dir)?;
        let lib_path = std::env::var("HANDY_PARAKEET_CPP_LIB")
            .ok()
            .map(|v| v.trim().to_string())
            .filter(|v| !v.is_empty())
            .map(PathBuf::from);

        let library = load_library(lib_path.as_deref())?;
        let api = unsafe { ParakeetApi::from_library(&library)? };

        let weights_cstr = path_to_cstring(&weights_path)?;
        let vocab_cstr = path_to_cstring(&vocab_path)?;

        let transcriber = unsafe {
            (api.tdt_transcriber_create)(
                weights_cstr.as_ptr(),
                vocab_cstr.as_ptr(),
                std::ptr::null_mut(),
            )
        };

        if transcriber.is_null() {
            return Err(anyhow!(
                "parakeet_tdt_transcriber_create failed: {}",
                unsafe { read_last_error(&api) }
            ));
        }

        let engine = Self {
            _library: library,
            api,
            transcriber,
        };

        if env_flag("HANDY_PARAKEET_CPP_FP16", true) {
            if let Err(error) = engine.try_enable_fp16() {
                warn!("Failed to enable Parakeet.cpp FP16: {}", error);
            }
        }

        if env_flag("HANDY_PARAKEET_CPP_GPU", true) {
            if let Err(error) = engine.try_enable_gpu() {
                warn!(
                    "Failed to enable Parakeet.cpp GPU acceleration (continuing on CPU): {}",
                    error
                );
            }
        }

        Ok(engine)
    }

    pub fn transcribe_samples(&mut self, samples: Vec<f32>) -> Result<String> {
        if samples.is_empty() {
            return Ok(String::new());
        }

        let result_handle = unsafe {
            (self.api.tdt_transcriber_transcribe_pcm)(
                self.transcriber,
                samples.as_ptr(),
                samples.len(),
                std::ptr::null_mut(),
            )
        };

        if result_handle.is_null() {
            return Err(anyhow!(
                "Parakeet.cpp transcription failed: {}",
                self.last_error()
            ));
        }

        let text_ptr = unsafe { (self.api.result_text)(result_handle) };
        let text = if text_ptr.is_null() {
            String::new()
        } else {
            unsafe { CStr::from_ptr(text_ptr) }
                .to_string_lossy()
                .into_owned()
        };

        unsafe { (self.api.result_free)(result_handle) };
        Ok(text)
    }

    fn try_enable_fp16(&self) -> Result<()> {
        let status = unsafe { (self.api.tdt_transcriber_to_half)(self.transcriber) };
        self.check_status("parakeet_tdt_transcriber_to_half", status)
    }

    fn try_enable_gpu(&self) -> Result<()> {
        let status = unsafe { (self.api.tdt_transcriber_to_gpu)(self.transcriber) };
        self.check_status("parakeet_tdt_transcriber_to_gpu", status)
    }

    fn check_status(&self, operation: &str, status: c_int) -> Result<()> {
        if status == 0 {
            return Ok(());
        }

        Err(anyhow!(
            "{} failed with code {}: {}",
            operation,
            status,
            self.last_error()
        ))
    }

    fn last_error(&self) -> String {
        unsafe { read_last_error(&self.api) }
    }
}

impl Drop for ParakeetCppEngine {
    fn drop(&mut self) {
        if !self.transcriber.is_null() {
            unsafe { (self.api.tdt_transcriber_free)(self.transcriber) };
            self.transcriber = std::ptr::null_mut();
        }
    }
}

unsafe fn read_last_error(api: &ParakeetApi) -> String {
    let message_ptr = unsafe { (api.last_error)() };
    if message_ptr.is_null() {
        "unknown error".to_string()
    } else {
        unsafe { CStr::from_ptr(message_ptr) }
            .to_string_lossy()
            .into_owned()
    }
}

fn path_to_cstring(path: &Path) -> Result<CString> {
    let path_str = path
        .to_str()
        .ok_or_else(|| anyhow!("Path contains invalid UTF-8: {}", path.display()))?;
    CString::new(path_str).context("Path contains an interior null byte")
}

fn load_library(explicit_path: Option<&Path>) -> Result<Library> {
    if let Some(path) = explicit_path {
        return unsafe { Library::new(path) }
            .with_context(|| format!("Failed to load Parakeet.cpp library: {}", path.display()));
    }

    let mut attempts = Vec::new();
    for candidate in default_library_candidates() {
        match unsafe { Library::new(&candidate) } {
            Ok(library) => {
                debug!("Loaded Parakeet.cpp library from {}", candidate.display());
                return Ok(library);
            }
            Err(error) => {
                attempts.push(format!("{} ({})", candidate.display(), error));
            }
        }
    }

    Err(anyhow!(
        "Unable to load Parakeet.cpp library. Set HANDY_PARAKEET_CPP_LIB to the library path. Attempts: {}",
        attempts.join("; ")
    ))
}

fn default_library_candidates() -> Vec<PathBuf> {
    let library_name = default_library_name();
    let mut candidates = Vec::new();

    if let Ok(exe_path) = std::env::current_exe() {
        if let Some(parent) = exe_path.parent() {
            candidates.push(parent.join(library_name));
        }
    }

    if let Ok(current_dir) = std::env::current_dir() {
        candidates.push(current_dir.join(library_name));
    }

    candidates.push(PathBuf::from(library_name));
    candidates
}

fn default_library_name() -> &'static str {
    #[cfg(target_os = "windows")]
    {
        "parakeet.dll"
    }

    #[cfg(target_os = "macos")]
    {
        "libparakeet.dylib"
    }

    #[cfg(all(not(target_os = "windows"), not(target_os = "macos")))]
    {
        "libparakeet.so"
    }
}

fn resolve_model_files(model_dir: &Path) -> Result<(PathBuf, PathBuf)> {
    if !model_dir.is_dir() {
        return Err(anyhow!(
            "Model path is not a directory: {}",
            model_dir.display()
        ));
    }

    let weights_path = find_weights_file(model_dir)?;
    let vocab_path = find_vocab_file(model_dir)?;

    Ok((weights_path, vocab_path))
}

fn find_weights_file(model_dir: &Path) -> Result<PathBuf> {
    let preferred = model_dir.join("model.safetensors");
    if preferred.is_file() {
        return Ok(preferred);
    }

    let mut safetensors = collect_files(model_dir, |path| {
        path.extension()
            .map(|ext| ext.to_string_lossy().eq_ignore_ascii_case("safetensors"))
            .unwrap_or(false)
    })?;

    safetensors.sort();
    safetensors.into_iter().next().ok_or_else(|| {
        anyhow!(
            "No .safetensors weights file found in {}",
            model_dir.display()
        )
    })
}

fn find_vocab_file(model_dir: &Path) -> Result<PathBuf> {
    for candidate in ["vocab.txt", "tokenizer.txt", "tokenizer.model"] {
        let path = model_dir.join(candidate);
        if path.is_file() {
            return Ok(path);
        }
    }

    let mut vocab_files = collect_files(model_dir, |path| {
        let Some(file_name) = path.file_name().map(|name| name.to_string_lossy()) else {
            return false;
        };

        let file_name = file_name.to_ascii_lowercase();
        if !file_name.contains("vocab") {
            return false;
        }

        matches!(
            path.extension().map(|ext| ext.to_string_lossy().to_ascii_lowercase()),
            Some(ext) if ext == "txt" || ext == "model"
        )
    })?;

    vocab_files.sort();
    vocab_files
        .into_iter()
        .next()
        .ok_or_else(|| anyhow!("No vocabulary file found in {}", model_dir.display()))
}

fn collect_files<F>(directory: &Path, predicate: F) -> Result<Vec<PathBuf>>
where
    F: Fn(&Path) -> bool,
{
    let mut files = Vec::new();
    for entry in fs::read_dir(directory)
        .with_context(|| format!("Failed to read model directory: {}", directory.display()))?
    {
        let path = entry?.path();
        if path.is_file() && predicate(&path) {
            files.push(path);
        }
    }
    Ok(files)
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

#[cfg(test)]
mod tests {
    use super::*;
    use tempfile::TempDir;

    #[test]
    fn resolve_model_files_prefers_standard_names() {
        let temp_dir = TempDir::new().unwrap();
        let model_dir = temp_dir.path();

        let expected_weights = model_dir.join("model.safetensors");
        let expected_vocab = model_dir.join("vocab.txt");
        fs::write(&expected_weights, b"weights").unwrap();
        fs::write(&expected_vocab, b"vocab").unwrap();

        let (weights, vocab) = resolve_model_files(model_dir).unwrap();
        assert_eq!(weights, expected_weights);
        assert_eq!(vocab, expected_vocab);
    }

    #[test]
    fn resolve_model_files_uses_fallback_patterns() {
        let temp_dir = TempDir::new().unwrap();
        let model_dir = temp_dir.path();

        let expected_weights = model_dir.join("parakeet-v3.safetensors");
        let expected_vocab = model_dir.join("my_vocab.model");
        fs::write(&expected_weights, b"weights").unwrap();
        fs::write(&expected_vocab, b"vocab").unwrap();

        let (weights, vocab) = resolve_model_files(model_dir).unwrap();
        assert_eq!(weights, expected_weights);
        assert_eq!(vocab, expected_vocab);
    }
}
