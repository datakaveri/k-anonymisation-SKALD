use serde::Serialize;
use serde_json::Value;
use std::collections::BTreeMap;
use std::error::Error;
use std::fmt;
use std::fs;
use std::io::{BufRead, BufReader, BufWriter, Write};
use std::path::{Path, PathBuf};

#[derive(Debug)]
pub enum PipelineError {
    Io(std::io::Error),
    Json(serde_json::Error),
    Validation { code: &'static str, message: String, details: String },
}

impl fmt::Display for PipelineError {
    fn fmt(&self, f: &mut fmt::Formatter<'_>) -> fmt::Result {
        match self {
            PipelineError::Io(e) => write!(f, "I/O error: {e}"),
            PipelineError::Json(e) => write!(f, "JSON error: {e}"),
            PipelineError::Validation { code, message, details } => {
                write!(f, "[{code}] {message}: {details}")
            }
        }
    }
}

impl Error for PipelineError {}

impl From<std::io::Error> for PipelineError {
    fn from(e: std::io::Error) -> Self {
        Self::Io(e)
    }
}

impl From<serde_json::Error> for PipelineError {
    fn from(e: serde_json::Error) -> Self {
        Self::Json(e)
    }
}

#[derive(Debug, Clone)]
pub struct RuntimeConfig {
    pub enable_k_anonymity: bool,
    pub k: i64,
    pub suppression_limit: f64,
    pub output_path: String,
    pub output_directory: String,
    pub suppress: Vec<String>,
    pub hashing_with_salt: Vec<String>,
    pub hashing_without_salt: Vec<String>,
    pub masking: Vec<Value>,
    pub encrypt: Vec<Value>,
    pub charcloak: Vec<String>,
    pub tokenization: Vec<Value>,
    pub fpe: Vec<Value>,
    pub numerical_qis: Vec<NumericalQiConfig>,
    pub categorical_qis: Vec<String>,
    pub size_factors: BTreeMap<String, i64>,
    pub source_json_config: PathBuf,
}

#[derive(Debug, Clone)]
pub struct NumericalQiConfig {
    pub column: String,
    pub scale: bool,
    pub s: i64,
    pub encode: bool,
    pub dtype: String,
}

#[derive(Debug, Serialize)]
pub struct StatusPayload {
    pub status: String,
    #[serde(skip_serializing_if = "Option::is_none")]
    pub outputs: Option<Value>,
    #[serde(skip_serializing_if = "Option::is_none")]
    pub error: Option<ErrorPayload>,
    pub log_file: String,
}

#[derive(Debug, Serialize)]
pub struct ErrorPayload {
    pub code: String,
    pub message: String,
    pub details: String,
    pub suggested_fix: String,
}

pub fn validation(code: &'static str, message: &str, details: &str) -> PipelineError {
    PipelineError::Validation {
        code,
        message: message.to_string(),
        details: details.to_string(),
    }
}

pub fn split_csv_line_basic(line: &str) -> Vec<String> {
    line.split(',').map(|s| s.to_string()).collect()
}

pub fn ensure_output_dir(path: &Path) -> Result<(), PipelineError> {
    fs::create_dir_all(path)?;
    Ok(())
}

pub fn find_first_json_config(config_dir: &Path) -> Result<PathBuf, PipelineError> {
    if !config_dir.is_dir() {
        return Err(validation(
            "CONFIG_INVALID",
            "Config directory not found",
            &config_dir.display().to_string(),
        ));
    }
    let mut files: Vec<PathBuf> = fs::read_dir(config_dir)?
        .filter_map(|e| e.ok())
        .map(|e| e.path())
        .filter(|p| p.extension().and_then(|e| e.to_str()) == Some("json"))
        .collect();
    files.sort();
    files
        .into_iter()
        .next()
        .ok_or_else(|| validation("CONFIG_INVALID", "No config JSON found", "config/ is empty"))
}

pub fn parse_runtime_config(config_path: &Path) -> Result<RuntimeConfig, PipelineError> {
    let raw = fs::read_to_string(config_path)?;
    let root: Value = serde_json::from_str(&raw)?;

    let data_type = root
        .get("data_type")
        .and_then(Value::as_str)
        .ok_or_else(|| validation("CONFIG_INVALID", "Missing 'data_type'", &config_path.display().to_string()))?;

    let section = root.get(data_type).ok_or_else(|| {
        validation(
            "CONFIG_INVALID",
            "Missing data_type section in config",
            data_type,
        )
    })?;

    let output_path = section
        .get("output_path")
        .and_then(Value::as_str)
        .unwrap_or("generalized.csv")
        .to_string();

    let output_directory = section
        .get("output_directory")
        .and_then(Value::as_str)
        .unwrap_or("output")
        .to_string();

    let suppress = section
        .get("suppress")
        .and_then(Value::as_array)
        .map(|a| {
            a.iter()
                .filter_map(Value::as_str)
                .map(str::to_string)
                .collect::<Vec<_>>()
        })
        .unwrap_or_default();

    let hashing_with_salt = section
        .get("hashing_with_salt")
        .and_then(Value::as_array)
        .map(|a| {
            a.iter()
                .filter_map(Value::as_str)
                .map(str::to_string)
                .collect::<Vec<_>>()
        })
        .unwrap_or_default();

    let hashing_without_salt = section
        .get("hashing_without_salt")
        .and_then(Value::as_array)
        .map(|a| {
            a.iter()
                .filter_map(Value::as_str)
                .map(str::to_string)
                .collect::<Vec<_>>()
        })
        .unwrap_or_default();

    let masking = section
        .get("masking")
        .and_then(Value::as_array)
        .cloned()
        .unwrap_or_default();

    let encrypt = section
        .get("encrypt")
        .and_then(Value::as_array)
        .cloned()
        .unwrap_or_default();

    let charcloak = section
        .get("charcloak")
        .and_then(Value::as_array)
        .map(|a| {
            a.iter()
                .filter_map(Value::as_str)
                .map(str::to_string)
                .collect::<Vec<_>>()
        })
        .unwrap_or_default();

    let tokenization = section
        .get("tokenization")
        .and_then(Value::as_array)
        .cloned()
        .unwrap_or_default();

    let fpe = section
        .get("fpe")
        .and_then(Value::as_array)
        .cloned()
        .unwrap_or_default();

    let suppression_limit = section
        .get("suppression_limit")
        .and_then(Value::as_f64)
        .unwrap_or(0.0);

    let k = section
        .get("k_anonymize")
        .and_then(|v| v.get("k"))
        .and_then(Value::as_i64)
        .unwrap_or(2);

    let mut numerical_qis = Vec::new();
    let mut categorical_qis = Vec::new();

    if let Some(qis) = section.get("quasi_identifiers") {
        if let Some(nums) = qis.get("numerical").and_then(Value::as_array) {
            for n in nums {
                numerical_qis.push(NumericalQiConfig {
                    column: n.get("column").and_then(Value::as_str).unwrap_or("").to_string(),
                    scale: n.get("scale").and_then(Value::as_bool).unwrap_or(false),
                    s: n.get("s").and_then(Value::as_i64).unwrap_or(0),
                    encode: n.get("encode").and_then(Value::as_bool).unwrap_or(false),
                    dtype: n.get("type").and_then(Value::as_str).unwrap_or("int").to_string(),
                });
            }
        }
        if let Some(cats) = qis.get("categorical").and_then(Value::as_array) {
            for c in cats {
                if let Some(col) = c.get("column").and_then(Value::as_str) {
                    categorical_qis.push(col.to_string());
                } else if let Some(col) = c.as_str() {
                    categorical_qis.push(col.to_string());
                }
            }
        }
    }

    let mut size_factors = BTreeMap::new();
    if let Some(obj) = section.get("size").and_then(Value::as_object) {
        for (k, v) in obj {
            if let Some(iv) = v.as_i64() {
                size_factors.insert(k.clone(), iv.max(1));
            }
        }
    }

    Ok(RuntimeConfig {
        enable_k_anonymity: true,
        k: k.max(1),
        suppression_limit,
        output_path,
        output_directory,
        suppress,
        hashing_with_salt,
        hashing_without_salt,
        masking,
        encrypt,
        charcloak,
        tokenization,
        fpe,
        numerical_qis,
        categorical_qis,
        size_factors,
        source_json_config: config_path.to_path_buf(),
    })
}

pub fn list_non_empty_csvs(data_dir: &Path) -> Result<Vec<PathBuf>, PipelineError> {
    if !data_dir.is_dir() {
        return Err(validation(
            "DATA_MISSING",
            "Data directory not found",
            &data_dir.display().to_string(),
        ));
    }
    let mut out: Vec<PathBuf> = fs::read_dir(data_dir)?
        .filter_map(|e| e.ok())
        .map(|e| e.path())
        .filter(|p| p.extension().and_then(|s| s.to_str()) == Some("csv"))
        .filter(|p| fs::metadata(p).map(|m| m.len() > 0).unwrap_or(false))
        .collect();
    out.sort();
    if out.is_empty() {
        return Err(validation("DATA_MISSING", "No non-empty CSV files found", "data/"));
    }
    Ok(out)
}

pub fn split_csv_by_ram(
    data_dir: &Path,
    chunks_dir: &Path,
    _available_ram_override: Option<u64>,
    rows_per_chunk_override: Option<usize>,
) -> Result<(Vec<PathBuf>, usize), PipelineError> {
    fs::create_dir_all(chunks_dir)?;
    let csvs = list_non_empty_csvs(data_dir)?;
    if csvs.len() != 1 {
        return Err(validation(
            "DATA_MISSING",
            "Expected exactly one CSV in data/",
            &format!("found {}", csvs.len()),
        ));
    }
    let input = &csvs[0];
    let file = fs::File::open(input)?;
    let reader = BufReader::new(file);
    let mut lines = reader.lines();
    let header = lines
        .next()
        .ok_or_else(|| validation("DATA_MISSING", "CSV is empty", &input.display().to_string()))?
        .map_err(|e| validation("DATA_MISSING", "Failed reading header", &e.to_string()))?;

    let rows_per_chunk = rows_per_chunk_override.unwrap_or(50_000).max(1000);
    let mut chunk_paths = Vec::new();
    let mut chunk_idx = 1usize;
    let mut row_count = 0usize;

    let mut out_path = chunks_dir.join(format!("chunk_{chunk_idx}.csv"));
    let mut writer = BufWriter::new(fs::File::create(&out_path)?);
    writer.write_all(header.as_bytes())?;
    writer.write_all(b"\n")?;

    for line in lines {
        let line = line.map_err(|e| validation("DATA_MISSING", "Failed reading row", &e.to_string()))?;
        if row_count > 0 && row_count % rows_per_chunk == 0 {
            writer.flush()?;
            chunk_paths.push(out_path.clone());
            chunk_idx += 1;
            out_path = chunks_dir.join(format!("chunk_{chunk_idx}.csv"));
            writer = BufWriter::new(fs::File::create(&out_path)?);
            writer.write_all(header.as_bytes())?;
            writer.write_all(b"\n")?;
        }
        writer.write_all(line.as_bytes())?;
        writer.write_all(b"\n")?;
        row_count += 1;
    }
    writer.flush()?;
    chunk_paths.push(out_path);

    Ok((chunk_paths, rows_per_chunk))
}
