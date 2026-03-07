fn main() {
    #[cfg(all(target_os = "macos", target_arch = "aarch64"))]
    build_apple_intelligence_bridge();

    #[cfg(windows)]
    ensure_windows_onnxruntime_cuda_dlls();

    generate_tray_translations();

    tauri_build::build()
}

/// Generate tray menu translations from frontend locale files.
///
/// Source of truth: src/i18n/locales/*/translation.json
/// The English "tray" section defines the struct fields.
fn generate_tray_translations() {
    use std::collections::BTreeMap;
    use std::fs;
    use std::path::Path;

    let out_dir = std::env::var("OUT_DIR").unwrap();
    let locales_dir = Path::new("../src/i18n/locales");

    println!("cargo:rerun-if-changed=../src/i18n/locales");

    // Collect all locale translations
    let mut translations: BTreeMap<String, serde_json::Value> = BTreeMap::new();

    for entry in fs::read_dir(locales_dir).unwrap().flatten() {
        let path = entry.path();
        if !path.is_dir() {
            continue;
        }

        let lang = path.file_name().unwrap().to_str().unwrap().to_string();
        let json_path = path.join("translation.json");

        println!("cargo:rerun-if-changed={}", json_path.display());

        let content = fs::read_to_string(&json_path).unwrap();
        let parsed: serde_json::Value = serde_json::from_str(&content).unwrap();

        if let Some(tray) = parsed.get("tray").cloned() {
            translations.insert(lang, tray);
        }
    }

    // English defines the schema
    let english = translations.get("en").unwrap().as_object().unwrap();
    let fields: Vec<_> = english
        .keys()
        .map(|k| (camel_to_snake(k), k.clone()))
        .collect();

    // Generate code
    let mut out = String::from(
        "// Auto-generated from src/i18n/locales/*/translation.json - do not edit\n\n",
    );

    // Struct
    out.push_str("#[derive(Debug, Clone)]\npub struct TrayStrings {\n");
    for (rust_field, _) in &fields {
        out.push_str(&format!("    pub {rust_field}: String,\n"));
    }
    out.push_str("}\n\n");

    // Static map
    out.push_str(
        "pub static TRANSLATIONS: Lazy<HashMap<&'static str, TrayStrings>> = Lazy::new(|| {\n",
    );
    out.push_str("    let mut m = HashMap::new();\n");

    for (lang, tray) in &translations {
        out.push_str(&format!("    m.insert(\"{lang}\", TrayStrings {{\n"));
        for (rust_field, json_key) in &fields {
            let val = tray.get(json_key).and_then(|v| v.as_str()).unwrap_or("");
            out.push_str(&format!(
                "        {rust_field}: \"{}\".to_string(),\n",
                escape_string(val)
            ));
        }
        out.push_str("    });\n");
    }

    out.push_str("    m\n});\n");

    fs::write(Path::new(&out_dir).join("tray_translations.rs"), out).unwrap();

    println!(
        "cargo:warning=Generated tray translations: {} languages, {} fields",
        translations.len(),
        fields.len()
    );
}

fn camel_to_snake(s: &str) -> String {
    s.chars()
        .enumerate()
        .fold(String::new(), |mut acc, (i, c)| {
            if c.is_uppercase() && i > 0 {
                acc.push('_');
            }
            acc.push(c.to_lowercase().next().unwrap());
            acc
        })
}

fn escape_string(s: &str) -> String {
    s.replace('\\', "\\\\")
        .replace('"', "\\\"")
        .replace('\n', "\\n")
        .replace('\r', "\\r")
        .replace('\t', "\\t")
}

#[cfg(windows)]
fn ensure_windows_onnxruntime_cuda_dlls() {
    use flate2::read::GzDecoder;
    use sha2::{Digest, Sha256};
    use std::env;
    use std::fs;
    use std::io;
    use std::path::{Path, PathBuf};
    use tar::Archive;

    const REQUIRED_DLLS: [&str; 2] = [
        "onnxruntime_providers_shared.dll",
        "onnxruntime_providers_cuda.dll",
    ];
    const CUDA_BUNDLE_URL: &str =
        "https://cdn.pyke.io/0/pyke:ort-rs/ms@1.22.0/x86_64-pc-windows-msvc+cu12.tgz";
    const CUDA_BUNDLE_SHA256: &str =
        "743380B97FAC97EDB2CB0DD656C517B99C5FEDD37516DA40F696335A3EDB5E55";

    println!("cargo:rerun-if-env-changed=HANDY_ORT_DLL_DIR");
    println!("cargo:rerun-if-env-changed=ORT_LIB_LOCATION");

    let manifest_dir = match env::var("CARGO_MANIFEST_DIR") {
        Ok(value) => PathBuf::from(value),
        Err(error) => {
            println!(
                "cargo:warning=Unable to resolve CARGO_MANIFEST_DIR for ONNX Runtime CUDA DLL preparation: {}",
                error
            );
            return;
        }
    };

    let destination_dir = manifest_dir.join("resources").join("onnxruntime");
    println!("cargo:rerun-if-changed={}", destination_dir.display());

    if let Err(error) = fs::create_dir_all(&destination_dir) {
        println!(
            "cargo:warning=Failed to create ONNX Runtime resources directory {}: {}",
            destination_dir.display(),
            error
        );
        return;
    }

    if required_dlls_present(&destination_dir, &REQUIRED_DLLS) {
        return;
    }

    if copy_required_dlls_from_candidates(&destination_dir, &REQUIRED_DLLS).is_ok() {
        println!(
            "cargo:warning=Prepared ONNX Runtime CUDA provider DLLs from local cache for bundling."
        );
        return;
    }

    let out_dir = match env::var("OUT_DIR") {
        Ok(value) => PathBuf::from(value),
        Err(error) => {
            println!(
                "cargo:warning=Unable to resolve OUT_DIR for ONNX Runtime CUDA DLL download: {}",
                error
            );
            return;
        }
    };

    match download_and_extract_required_dlls(
        CUDA_BUNDLE_URL,
        CUDA_BUNDLE_SHA256,
        &out_dir,
        &destination_dir,
        &REQUIRED_DLLS,
    ) {
        Ok(()) => {
            println!("cargo:warning=Downloaded ONNX Runtime CUDA provider DLLs for Windows bundle.")
        }
        Err(error) => println!(
            "cargo:warning=Unable to prepare ONNX Runtime CUDA provider DLLs automatically: {}",
            error
        ),
    }

    fn required_dlls_present(directory: &Path, required: &[&str]) -> bool {
        required.iter().all(|name| directory.join(name).is_file())
    }

    fn copy_required_dlls_from_candidates(destination: &Path, required: &[&str]) -> io::Result<()> {
        let mut candidates = Vec::new();

        if let Ok(custom_dir) = env::var("HANDY_ORT_DLL_DIR") {
            push_unique_dir(&mut candidates, PathBuf::from(custom_dir));
        }

        if let Ok(ort_lib_location) = env::var("ORT_LIB_LOCATION") {
            let base = PathBuf::from(ort_lib_location);
            push_unique_dir(&mut candidates, base.clone());
            push_unique_dir(&mut candidates, base.join("lib"));
        }

        if let Some(local_app_data) = env::var_os("LOCALAPPDATA") {
            let cache_base = PathBuf::from(local_app_data)
                .join("cache")
                .join("dfbin")
                .join("x86_64-pc-windows-msvc");
            if cache_base.is_dir() {
                for entry in fs::read_dir(cache_base)? {
                    let path = entry?.path();
                    push_unique_dir(&mut candidates, path.join("onnxruntime").join("lib"));
                }
            }
        }

        for source in candidates {
            if !required_dlls_present(&source, required) {
                continue;
            }

            copy_required_dlls(&source, destination, required)?;
            if required_dlls_present(destination, required) {
                return Ok(());
            }
        }

        Err(io::Error::new(
            io::ErrorKind::NotFound,
            "No local ONNX Runtime CUDA provider DLL source found",
        ))
    }

    fn copy_required_dlls(source: &Path, destination: &Path, required: &[&str]) -> io::Result<()> {
        fs::create_dir_all(destination)?;
        for dll in required {
            let from = source.join(dll);
            if from.is_file() {
                let to = destination.join(dll);
                if !to.is_file() {
                    fs::copy(from, to)?;
                }
            }
        }
        Ok(())
    }

    fn download_and_extract_required_dlls(
        archive_url: &str,
        expected_sha256: &str,
        out_dir: &Path,
        destination: &Path,
        required: &[&str],
    ) -> io::Result<()> {
        let downloaded = ureq::Agent::new_with_config(
            ureq::config::Config::builder()
                .https_only(true)
                .tls_config(
                    ureq::tls::TlsConfig::builder()
                        .provider(ureq::tls::TlsProvider::NativeTls)
                        .root_certs(ureq::tls::RootCerts::PlatformVerifier)
                        .build(),
                )
                .build(),
        )
        .get(archive_url)
        .call()
        .map_err(|error| {
            io::Error::new(
                io::ErrorKind::Other,
                format!("Failed to download ONNX Runtime CUDA archive: {}", error),
            )
        })?
        .into_body()
        .into_with_config()
        .limit(1_073_741_824)
        .read_to_vec()
        .map_err(|error| {
            io::Error::new(
                io::ErrorKind::Other,
                format!(
                    "Failed to read ONNX Runtime CUDA archive response: {}",
                    error
                ),
            )
        })?;

        let actual_sha256 = {
            let digest = Sha256::digest(&downloaded);
            digest
                .iter()
                .map(|byte| format!("{:02X}", byte))
                .collect::<String>()
        };

        if actual_sha256 != expected_sha256 {
            return Err(io::Error::new(
                io::ErrorKind::InvalidData,
                format!(
                    "Unexpected ONNX Runtime CUDA archive SHA-256. Expected {}, got {}",
                    expected_sha256, actual_sha256
                ),
            ));
        }

        let extract_dir = out_dir.join("onnxruntime-cuda-bundle");
        if extract_dir.is_dir() {
            fs::remove_dir_all(&extract_dir)?;
        }
        fs::create_dir_all(&extract_dir)?;

        let decoder = GzDecoder::new(downloaded.as_slice());
        let mut archive = Archive::new(decoder);
        archive.unpack(&extract_dir)?;

        let lib_dir = extract_dir.join("onnxruntime").join("lib");
        if !required_dlls_present(&lib_dir, required) {
            return Err(io::Error::new(
                io::ErrorKind::NotFound,
                format!(
                    "Downloaded ONNX Runtime archive did not contain required CUDA provider DLLs in {}",
                    lib_dir.display()
                ),
            ));
        }

        copy_required_dlls(&lib_dir, destination, required)
    }

    fn push_unique_dir(directories: &mut Vec<PathBuf>, candidate: PathBuf) {
        if candidate.is_dir() && !directories.iter().any(|existing| existing == &candidate) {
            directories.push(candidate);
        }
    }
}

#[cfg(all(target_os = "macos", target_arch = "aarch64"))]
fn build_apple_intelligence_bridge() {
    use std::env;
    use std::path::{Path, PathBuf};
    use std::process::Command;

    const REAL_SWIFT_FILE: &str = "swift/apple_intelligence.swift";
    const STUB_SWIFT_FILE: &str = "swift/apple_intelligence_stub.swift";
    const BRIDGE_HEADER: &str = "swift/apple_intelligence_bridge.h";

    println!("cargo:rerun-if-changed={REAL_SWIFT_FILE}");
    println!("cargo:rerun-if-changed={STUB_SWIFT_FILE}");
    println!("cargo:rerun-if-changed={BRIDGE_HEADER}");

    let out_dir = PathBuf::from(env::var("OUT_DIR").expect("OUT_DIR not set"));
    let object_path = out_dir.join("apple_intelligence.o");
    let static_lib_path = out_dir.join("libapple_intelligence.a");

    let sdk_path = String::from_utf8(
        Command::new("xcrun")
            .args(["--sdk", "macosx", "--show-sdk-path"])
            .output()
            .expect("Failed to locate macOS SDK")
            .stdout,
    )
    .expect("SDK path is not valid UTF-8")
    .trim()
    .to_string();

    // Check if the SDK supports FoundationModels (required for Apple Intelligence)
    let framework_path =
        Path::new(&sdk_path).join("System/Library/Frameworks/FoundationModels.framework");
    let has_foundation_models = framework_path.exists();

    let source_file = if has_foundation_models {
        println!("cargo:warning=Building with Apple Intelligence support.");
        REAL_SWIFT_FILE
    } else {
        println!("cargo:warning=Apple Intelligence SDK not found. Building with stubs.");
        STUB_SWIFT_FILE
    };

    if !Path::new(source_file).exists() {
        panic!("Source file {} is missing!", source_file);
    }

    let swiftc_path = String::from_utf8(
        Command::new("xcrun")
            .args(["--find", "swiftc"])
            .output()
            .expect("Failed to locate swiftc")
            .stdout,
    )
    .expect("swiftc path is not valid UTF-8")
    .trim()
    .to_string();

    let toolchain_swift_lib = Path::new(&swiftc_path)
        .parent()
        .and_then(|p| p.parent())
        .map(|root| root.join("lib/swift/macosx"))
        .expect("Unable to determine Swift toolchain lib directory");
    let sdk_swift_lib = Path::new(&sdk_path).join("usr/lib/swift");

    // Use macOS 11.0 as deployment target for compatibility
    // The @available(macOS 26.0, *) checks in Swift handle runtime availability
    // Weak linking for FoundationModels is handled via cargo:rustc-link-arg below
    let status = Command::new("xcrun")
        .args([
            "swiftc",
            "-target",
            "arm64-apple-macosx11.0",
            "-sdk",
            &sdk_path,
            "-O",
            "-import-objc-header",
            BRIDGE_HEADER,
            "-c",
            source_file,
            "-o",
            object_path
                .to_str()
                .expect("Failed to convert object path to string"),
        ])
        .status()
        .expect("Failed to invoke swiftc for Apple Intelligence bridge");

    if !status.success() {
        panic!("swiftc failed to compile {source_file}");
    }

    let status = Command::new("libtool")
        .args([
            "-static",
            "-o",
            static_lib_path
                .to_str()
                .expect("Failed to convert static lib path to string"),
            object_path
                .to_str()
                .expect("Failed to convert object path to string"),
        ])
        .status()
        .expect("Failed to create static library for Apple Intelligence bridge");

    if !status.success() {
        panic!("libtool failed for Apple Intelligence bridge");
    }

    println!("cargo:rustc-link-search=native={}", out_dir.display());
    println!("cargo:rustc-link-lib=static=apple_intelligence");
    println!(
        "cargo:rustc-link-search=native={}",
        toolchain_swift_lib.display()
    );
    println!("cargo:rustc-link-search=native={}", sdk_swift_lib.display());
    println!("cargo:rustc-link-lib=framework=Foundation");

    if has_foundation_models {
        // Use weak linking so the app can launch on systems without FoundationModels
        println!("cargo:rustc-link-arg=-weak_framework");
        println!("cargo:rustc-link-arg=FoundationModels");
    }

    println!("cargo:rustc-link-arg=-Wl,-rpath,/usr/lib/swift");
}
