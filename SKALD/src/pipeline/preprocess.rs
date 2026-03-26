use crate::pipeline::bootstrap::{split_csv_line_basic, validation, PipelineError, RuntimeConfig};
use crate::pipeline::pyffx_compat::fpe_encrypt;
use hmac::{Hmac, Mac};
use serde_json::Value;
use sha2::Sha256;
use std::collections::BTreeMap;
use std::fs;
use std::io::{BufRead, BufReader, BufWriter, Write};
use std::path::{Path, PathBuf};

fn fnv1a64(input: &str) -> u64 {
    let mut hash: u64 = 0xcbf29ce484222325;
    for b in input.as_bytes() {
        hash ^= *b as u64;
        hash = hash.wrapping_mul(0x100000001b3);
    }
    hash
}

/// Generate a random 32-hex-char (16-byte) key from /dev/urandom.
/// Falls back to zeroed bytes only if the OS RNG is unavailable — still better
/// than a deterministic derivation from the column name.
fn generate_random_key_hex() -> String {
    use std::io::Read;
    let mut buf = [0u8; 16];
    if let Ok(mut f) = std::fs::File::open("/dev/urandom") {
        let _ = f.read_exact(&mut buf);
    }
    hex::encode(buf)
}

fn hash_hex(input: &str) -> String {
    format!("{:016x}", fnv1a64(input))
}

fn should_skip_value(v: &str) -> bool {
    let t = v.trim();
    t.is_empty() || t.eq_ignore_ascii_case("nan")
}

fn read_json_map_string(path: &Path) -> Result<BTreeMap<String, String>, PipelineError> {
    if !path.exists() {
        return Ok(BTreeMap::new());
    }
    let raw = fs::read_to_string(path)?;
    let val: Value = serde_json::from_str(&raw)?;
    let mut out = BTreeMap::new();
    if let Some(obj) = val.as_object() {
        for (k, v) in obj {
            if let Some(s) = v.as_str() {
                out.insert(k.clone(), s.to_string());
            }
        }
    }
    Ok(out)
}

fn write_json_pretty(path: &Path, v: &Value) -> Result<(), PipelineError> {
    if let Some(parent) = path.parent() {
        fs::create_dir_all(parent)?;
    }
    let body = serde_json::to_string_pretty(v)?;
    let tmp = path.with_extension("tmp");
    fs::write(&tmp, body)?;
    fs::rename(tmp, path)?;
    Ok(())
}

type HmacSha256 = Hmac<Sha256>;

fn derive_key(master_key: &str, context: &str) -> [u8; 16] {
    let mut mac = HmacSha256::new_from_slice(master_key.as_bytes()).expect("HMAC init");
    mac.update(context.as_bytes());
    let digest = mac.finalize().into_bytes();
    let mut out = [0u8; 16];
    out.copy_from_slice(&digest[..16]);
    out
}

fn format_preserving_encrypt_general(value: &str, master_key: &str, column: &str) -> String {
    let text = value.to_string();
    let chars: Vec<char> = text.chars().collect();
    let mut i = 0usize;
    let mut out = String::with_capacity(chars.len());

    while i < chars.len() {
        let ch = chars[i];
        let (class_name, alphabet) = if ch.is_ascii_uppercase() {
            ("upper", "ABCDEFGHIJKLMNOPQRSTUVWXYZ")
        } else if ch.is_ascii_lowercase() {
            ("lower", "abcdefghijklmnopqrstuvwxyz")
        } else if ch.is_ascii_digit() {
            ("digit", "0123456789")
        } else {
            out.push(ch);
            i += 1;
            continue;
        };

        let mut j = i + 1;
        while j < chars.len() {
            let c = chars[j];
            let same = (class_name == "upper" && c.is_ascii_uppercase())
                || (class_name == "lower" && c.is_ascii_lowercase())
                || (class_name == "digit" && c.is_ascii_digit());
            if !same {
                break;
            }
            j += 1;
        }

        let segment: String = chars[i..j].iter().collect();
        let ctx = format!("{column}:{class_name}:{}", segment.len());
        let key = derive_key(master_key, &ctx);
        let encrypted = fpe_encrypt(&key, &segment, alphabet);
        out.push_str(&encrypted);
        i = j;
    }

    out
}

fn pseudo_encrypt(value: &str, key: &str, column: &str) -> String {
    // Build a keystream via successive HMAC-SHA256 blocks so the pattern never repeats,
    // regardless of value length. Each 16-byte block i uses context "<column>:<i>".
    let plaintext = value.as_bytes();
    let mut keystream = Vec::with_capacity(plaintext.len());
    let mut block_idx: u64 = 0;
    while keystream.len() < plaintext.len() {
        let ctx = format!("{column}:{block_idx}");
        keystream.extend_from_slice(&derive_key(key, &ctx));
        block_idx += 1;
    }
    let mut out = String::from("ENC$");
    for (p, k) in plaintext.iter().zip(keystream.iter()) {
        out.push_str(&format!("{:02x}", p ^ k));
    }
    out
}

fn fpe_pan_encrypt(value: &str, master_key: &str) -> String {
    let chars: Vec<char> = value.chars().collect();
    if chars.len() != 10
        || !chars[..5].iter().all(|c| c.is_ascii_uppercase())
        || !chars[5..9].iter().all(|c| c.is_ascii_digit())
        || !chars[9].is_ascii_uppercase()
    {
        return value.to_string();
    }
    let letters_key = derive_key(master_key, "pan_letters");
    let digits_key = derive_key(master_key, "pan_digits");
    let suffix_key = derive_key(master_key, "pan_suffix");
    let part1: String = chars[..5].iter().collect();
    let part2: String = chars[5..9].iter().collect();
    let part3: String = chars[9..10].iter().collect();

    let e1 = fpe_encrypt(&letters_key, &part1, "ABCDEFGHIJKLMNOPQRSTUVWXYZ");
    let e2 = fpe_encrypt(&digits_key, &part2, "0123456789");
    let e3 = fpe_encrypt(&suffix_key, &part3, "ABCDEFGHIJKLMNOPQRSTUVWXYZ");
    format!("{e1}{e2}{e3}")
}

fn fpe_digits_encrypt(value: &str, master_key: &str) -> String {
    if value.is_empty() || !value.chars().all(|c| c.is_ascii_digit()) {
        return value.to_string();
    }
    let key = derive_key(master_key, &format!("digits_len_{}", value.len()));
    fpe_encrypt(&key, value, "0123456789")
}

#[derive(Debug)]
struct TokenizationConfigLite {
    column: String,
    prefix: String,
    digits: usize,
}

fn parse_tokenization_config(entry: &Value) -> Result<TokenizationConfigLite, PipelineError> {
    let obj = entry
        .as_object()
        .ok_or_else(|| validation("PREPROCESS_CONFIG_INVALID", "Each tokenization entry must be an object", "tokenization"))?;
    let column = obj
        .get("column")
        .and_then(Value::as_str)
        .ok_or_else(|| validation("PREPROCESS_CONFIG_INVALID", "Tokenization config missing 'column'", "tokenization"))?
        .to_string();
    let prefix = obj.get("prefix").and_then(Value::as_str).unwrap_or("TK-").to_string();
    let digits = obj.get("digits").and_then(Value::as_i64).unwrap_or(6);
    if digits <= 0 {
        return Err(validation("PREPROCESS_CONFIG_INVALID", "tokenization 'digits' must be > 0", &column));
    }
    Ok(TokenizationConfigLite { column, prefix, digits: digits as usize })
}

#[derive(Debug)]
struct FpeConfigLite {
    column: String,
    format: String,
}

fn parse_fpe_config(entry: &Value) -> Result<FpeConfigLite, PipelineError> {
    let obj = entry
        .as_object()
        .ok_or_else(|| validation("PREPROCESS_CONFIG_INVALID", "Each FPE entry must be an object", "fpe"))?;
    let column = obj
        .get("column")
        .and_then(Value::as_str)
        .ok_or_else(|| validation("PREPROCESS_CONFIG_INVALID", "FPE config missing 'column'", "fpe"))?
        .to_string();
    let format = obj.get("format").and_then(Value::as_str).unwrap_or("pan").to_string();
    if format != "pan" && format != "digits" {
        return Err(validation("PREPROCESS_CONFIG_INVALID", "FPE format must be 'pan' or 'digits'", &column));
    }
    Ok(FpeConfigLite { column, format })
}

#[derive(Debug)]
struct EncryptConfigLite {
    column: String,
    format_preserving: bool,
}

fn parse_encrypt_config(entry: &Value) -> Result<EncryptConfigLite, PipelineError> {
    if let Some(col) = entry.as_str() {
        return Ok(EncryptConfigLite { column: col.to_string(), format_preserving: false });
    }
    let obj = entry
        .as_object()
        .ok_or_else(|| validation("PREPROCESS_CONFIG_INVALID", "Encrypt entry must be string or object", "encrypt"))?;
    if obj.len() != 1 {
        return Err(validation("PREPROCESS_CONFIG_INVALID", "Encrypt object entry must contain one column key", "encrypt"));
    }
    let (column, opts) = obj.iter().next().expect("checked len=1");
    let format_preserving = opts
        .as_object()
        .and_then(|m| m.get("format_preserving"))
        .and_then(Value::as_bool)
        .unwrap_or(false);
    Ok(EncryptConfigLite { column: column.clone(), format_preserving })
}

struct SimpleRng {
    state: u64,
}

impl SimpleRng {
    fn new(seed: u64) -> Self {
        Self { state: seed.max(1) }
    }
    fn next_u32(&mut self) -> u32 {
        let mut x = self.state;
        x ^= x >> 12;
        x ^= x << 25;
        x ^= x >> 27;
        self.state = x;
        (x.wrapping_mul(2685821657736338717) >> 32) as u32
    }
}

fn randomize_preserving_class(value: &str, seed: u64) -> String {
    let mut rng = SimpleRng::new(seed);
    let mut out = String::with_capacity(value.len());
    for c in value.chars() {
        if c.is_ascii_digit() {
            out.push((b'0' + (rng.next_u32() % 10) as u8) as char);
        } else if c.is_ascii_uppercase() {
            out.push((b'A' + (rng.next_u32() % 26) as u8) as char);
        } else if c.is_ascii_lowercase() {
            out.push((b'a' + (rng.next_u32() % 26) as u8) as char);
        } else {
            out.push(c);
        }
    }
    out
}

#[derive(Debug)]
struct MaskingConfigLite {
    column: String,
    masking_char: char,
    characters_to_mask: Vec<usize>,
    class_masking_mode: Option<String>,
    class_letter: char,
    class_digit: char,
}

fn parse_masking_config(entry: &Value) -> Result<MaskingConfigLite, PipelineError> {
    let obj = entry
        .as_object()
        .ok_or_else(|| validation("PREPROCESS_CONFIG_INVALID", "Each masking entry must be an object", "masking"))?;
    let column = obj
        .get("column")
        .and_then(Value::as_str)
        .ok_or_else(|| validation("PREPROCESS_CONFIG_INVALID", "Masking config missing 'column'", "masking"))?
        .to_string();

    let masking_char_s = obj
        .get("masking_char")
        .or_else(|| obj.get("masking_character"))
        .and_then(Value::as_str)
        .unwrap_or("*");
    let masking_char = masking_char_s
        .chars()
        .next()
        .ok_or_else(|| validation("PREPROCESS_CONFIG_INVALID", "masking_char cannot be empty", &column))?;

    let mut characters_to_mask = Vec::new();
    if let Some(arr) = obj.get("characters_to_mask").and_then(Value::as_array) {
        for v in arr {
            let pos = v.as_i64().ok_or_else(|| {
                validation("PREPROCESS_CONFIG_INVALID", "characters_to_mask must be positive integers", &column)
            })?;
            if pos > 0 {
                characters_to_mask.push(pos as usize);
            }
        }
    }

    let class_masking_mode = obj.get("class_masking_mode").and_then(Value::as_str).map(|s| s.to_string());
    let class_letter = obj
        .get("class_mask_letter")
        .and_then(Value::as_str)
        .unwrap_or("X")
        .chars()
        .next()
        .unwrap_or('X');
    let class_digit = obj
        .get("class_mask_digit")
        .and_then(Value::as_str)
        .unwrap_or("0")
        .chars()
        .next()
        .unwrap_or('0');

    Ok(MaskingConfigLite {
        column,
        masking_char,
        characters_to_mask,
        class_masking_mode,
        class_letter,
        class_digit,
    })
}

fn apply_masking_value(value: &str, cfg: &MaskingConfigLite, seed: u64) -> String {
    let mut masked = value.to_string();

    if !cfg.characters_to_mask.is_empty() {
        let mut chars: Vec<char> = masked.chars().collect();
        for pos in &cfg.characters_to_mask {
            let idx = pos.saturating_sub(1);
            if idx < chars.len() {
                chars[idx] = cfg.masking_char;
            }
        }
        masked = chars.into_iter().collect();
    }

    match cfg.class_masking_mode.as_deref() {
        Some("random_class") => {
            masked = randomize_preserving_class(&masked, seed);
        }
        Some("fixed_class") => {
            let mut out = String::with_capacity(masked.len());
            for c in masked.chars() {
                if c.is_ascii_digit() {
                    out.push(cfg.class_digit);
                } else if c.is_ascii_alphabetic() {
                    out.push(cfg.class_letter);
                } else {
                    out.push(c);
                }
            }
            masked = out;
        }
        _ => {}
    }

    masked
}

pub fn preprocess_chunks(chunks: &[PathBuf], cfg: &RuntimeConfig) -> Result<(), PipelineError> {
    let masking_cfgs: Vec<MaskingConfigLite> = cfg
        .masking
        .iter()
        .map(parse_masking_config)
        .collect::<Result<Vec<_>, _>>()?;
    let token_cfgs: Vec<TokenizationConfigLite> = cfg
        .tokenization
        .iter()
        .map(parse_tokenization_config)
        .collect::<Result<Vec<_>, _>>()?;
    let fpe_cfgs: Vec<FpeConfigLite> = cfg.fpe.iter().map(parse_fpe_config).collect::<Result<Vec<_>, _>>()?;
    let encrypt_cfgs: Vec<EncryptConfigLite> = cfg.encrypt.iter().map(parse_encrypt_config).collect::<Result<Vec<_>, _>>()?;

    let out_dir_buf = if Path::new(&cfg.output_directory).is_absolute() {
        PathBuf::from(&cfg.output_directory)
    } else if let Some(first_chunk) = chunks.first() {
        if let Some(chunks_dir) = first_chunk.parent() {
            if let Some(root_dir) = chunks_dir.parent() {
                root_dir.join(&cfg.output_directory)
            } else {
                chunks_dir.join(&cfg.output_directory)
            }
        } else {
            PathBuf::from(&cfg.output_directory)
        }
    } else {
        PathBuf::from(&cfg.output_directory)
    };
    let out_dir = out_dir_buf.as_path();
    fs::create_dir_all(out_dir)?;

    // --- In-memory token vault (replaces JSON-Value traversal per row) ---
    // Keyed by column name → (forward: value→token, reverse: token→value, next sequential id)
    struct InMemVault {
        forward: BTreeMap<String, String>,
        reverse: BTreeMap<String, String>,
        next_id: u64,
    }

    let token_vault_path = out_dir.join("token_vault.json");
    let mut in_mem_vaults: BTreeMap<String, InMemVault> = BTreeMap::new();
    if !token_cfgs.is_empty() {
        let existing: Value = if token_vault_path.exists() {
            serde_json::from_str(&fs::read_to_string(&token_vault_path)?)?
        } else {
            serde_json::json!({})
        };
        for tcfg in &token_cfgs {
            let fwd: BTreeMap<String, String> = existing
                .get(&tcfg.column)
                .and_then(|c| c.get("forward"))
                .and_then(Value::as_object)
                .map(|m| m.iter().filter_map(|(k, v)| v.as_str().map(|s| (k.clone(), s.to_string()))).collect())
                .unwrap_or_default();
            let rev: BTreeMap<String, String> = existing
                .get(&tcfg.column)
                .and_then(|c| c.get("reverse"))
                .and_then(Value::as_object)
                .map(|m| m.iter().filter_map(|(k, v)| v.as_str().map(|s| (k.clone(), s.to_string()))).collect())
                .unwrap_or_default();
            let next_id = rev.len() as u64 + 1;
            in_mem_vaults.insert(tcfg.column.clone(), InMemVault { forward: fwd, reverse: rev, next_id });
        }
    }

    let fpe_keys_path = out_dir.join("fpe_keys.json");
    let mut fpe_keys = read_json_map_string(&fpe_keys_path)?;

    let symmetric_keys_path = out_dir.join("symmetric_keys.json");
    let mut symmetric_keys = read_json_map_string(&symmetric_keys_path)?;

    let fpe_encrypt_keys_path = out_dir.join("fpe_encrypt_keys.json");
    let mut fpe_encrypt_keys = read_json_map_string(&fpe_encrypt_keys_path)?;

    for chunk_path in chunks {
        let file = fs::File::open(chunk_path)?;
        let reader = BufReader::new(file);
        let mut lines = reader.lines();
        let header_line = lines
            .next()
            .ok_or_else(|| validation("DATA_EMPTY", "Chunk is empty", &chunk_path.display().to_string()))?
            .map_err(|e| validation("IO_READ_FAILED", "Failed reading header", &e.to_string()))?;
        let mut headers = split_csv_line_basic(&header_line);
        let mut rows: Vec<Vec<String>> = Vec::new();
        for line in lines {
            let line = line.map_err(|e| validation("IO_READ_FAILED", "Failed reading row", &e.to_string()))?;
            if line.trim().is_empty() {
                continue;
            }
            rows.push(split_csv_line_basic(&line));
        }

        if !cfg.suppress.is_empty() {
            let mut drop_idx = Vec::new();
            for col in &cfg.suppress {
                if let Some(i) = headers.iter().position(|h| h == col) {
                    drop_idx.push(i);
                } else {
                    return Err(validation("PREPROCESS_COLUMN_MISSING", "Suppression column not found in CSV header", col));
                }
            }
            drop_idx.sort_unstable();
            drop_idx.dedup();
            headers = headers
                .into_iter()
                .enumerate()
                .filter(|(i, _)| !drop_idx.contains(i))
                .map(|(_, h)| h)
                .collect();
            for row in &mut rows {
                *row = row
                    .iter()
                    .enumerate()
                    .filter(|(i, _)| !drop_idx.contains(i))
                    .map(|(_, v)| v.clone())
                    .collect();
            }
        }

        for col in &cfg.hashing_with_salt {
            let idx = headers
                .iter()
                .position(|h| h == col)
                .ok_or_else(|| validation("PREPROCESS_COLUMN_MISSING", "Column not found in CSV header for salted hashing", col))?;
            let salt = format!("skald_salt_{}", hash_hex(col));
            for row in &mut rows {
                if idx >= row.len() {
                    continue;
                }
                let v = row[idx].clone();
                if should_skip_value(&v) {
                    continue;
                }
                row[idx] = hash_hex(&(salt.clone() + &v));
            }
        }

        for col in &cfg.hashing_without_salt {
            let idx = headers
                .iter()
                .position(|h| h == col)
                .ok_or_else(|| validation("PREPROCESS_COLUMN_MISSING", "Column not found in CSV header for hashing", col))?;
            for row in &mut rows {
                if idx >= row.len() {
                    continue;
                }
                let v = row[idx].clone();
                if should_skip_value(&v) {
                    continue;
                }
                row[idx] = hash_hex(&v);
            }
        }

        for m in &masking_cfgs {
            let idx = headers
                .iter()
                .position(|h| h == &m.column)
                .ok_or_else(|| validation("PREPROCESS_COLUMN_MISSING", "Column not found in CSV header for masking", &m.column))?;
            for (r_i, row) in rows.iter_mut().enumerate() {
                if idx >= row.len() {
                    continue;
                }
                let v = row[idx].clone();
                if should_skip_value(&v) {
                    continue;
                }
                let seed = fnv1a64(&format!("{}::{}::{}", m.column, r_i, v));
                row[idx] = apply_masking_value(&v, m, seed);
            }
        }

        for col in &cfg.charcloak {
            let idx = headers
                .iter()
                .position(|h| h == col)
                .ok_or_else(|| validation("PREPROCESS_COLUMN_MISSING", "Column not found in CSV header for charcloak", col))?;
            for (r_i, row) in rows.iter_mut().enumerate() {
                if idx >= row.len() {
                    continue;
                }
                let v = row[idx].clone();
                if should_skip_value(&v) {
                    continue;
                }
                let seed = fnv1a64(&format!("charcloak::{}::{}::{}", col, r_i, v));
                row[idx] = randomize_preserving_class(&v, seed);
            }
        }

        for tcfg in &token_cfgs {
            let idx = headers
                .iter()
                .position(|h| h == &tcfg.column)
                .ok_or_else(|| validation("PREPROCESS_COLUMN_MISSING", "Column not found in CSV header for tokenization", &tcfg.column))?;
            let vault = in_mem_vaults
                .get_mut(&tcfg.column)
                .ok_or_else(|| validation("INTERNAL_ERROR", "Token vault missing for column", &tcfg.column))?;

            for row in &mut rows {
                if idx >= row.len() {
                    continue;
                }
                let v = row[idx].clone();
                if should_skip_value(&v) {
                    continue;
                }
                // O(1) lookup in BTreeMap instead of traversing JSON Value tree
                if let Some(tok) = vault.forward.get(&v) {
                    row[idx] = tok.clone();
                } else {
                    // next_id is a simple counter — no collision loop needed
                    let token = format!("{}{:0width$}", tcfg.prefix, vault.next_id, width = tcfg.digits);
                    vault.forward.insert(v.clone(), token.clone());
                    vault.reverse.insert(token.clone(), v);
                    vault.next_id += 1;
                    row[idx] = token;
                }
            }
        }

        for fcfg in &fpe_cfgs {
            let idx = headers
                .iter()
                .position(|h| h == &fcfg.column)
                .ok_or_else(|| validation("PREPROCESS_COLUMN_MISSING", "Column not found in CSV header for FPE encryption", &fcfg.column))?;
            let key = fpe_keys
                .entry(fcfg.column.clone())
                .or_insert_with(generate_random_key_hex)
                .clone();

            for row in &mut rows {
                if idx >= row.len() {
                    continue;
                }
                let v = row[idx].clone();
                if should_skip_value(&v) {
                    continue;
                }
                row[idx] = if fcfg.format == "pan" {
                    fpe_pan_encrypt(&v, &key)
                } else {
                    fpe_digits_encrypt(&v, &key)
                };
            }
        }

        for ecfg in &encrypt_cfgs {
            let idx = headers
                .iter()
                .position(|h| h == &ecfg.column)
                .ok_or_else(|| validation("PREPROCESS_COLUMN_MISSING", "Column not found in CSV header for encryption", &ecfg.column))?;

            if ecfg.format_preserving {
                let key = fpe_encrypt_keys
                    .entry(ecfg.column.clone())
                    .or_insert_with(generate_random_key_hex)
                    .clone();
                for row in &mut rows {
                    if idx >= row.len() {
                        continue;
                    }
                    let v = row[idx].clone();
                    if should_skip_value(&v) {
                        continue;
                    }
                    row[idx] = format_preserving_encrypt_general(&v, &key, &ecfg.column);
                }
            } else {
                let key = symmetric_keys
                    .entry(ecfg.column.clone())
                    .or_insert_with(generate_random_key_hex)
                    .clone();
                for row in &mut rows {
                    if idx >= row.len() {
                        continue;
                    }
                    let v = row[idx].clone();
                    if should_skip_value(&v) {
                        continue;
                    }
                    row[idx] = pseudo_encrypt(&v, &key, &ecfg.column);
                }
            }
        }

        let tmp_path = chunk_path.with_extension("csv.tmp");
        let mut w = BufWriter::new(fs::File::create(&tmp_path)?);
        w.write_all(headers.join(",").as_bytes())?;
        w.write_all(b"\n")?;
        for row in rows {
            w.write_all(row.join(",").as_bytes())?;
            w.write_all(b"\n")?;
        }
        w.flush()?;
        fs::rename(tmp_path, chunk_path)?;
    }

    // Serialize in-memory vaults back to the JSON format expected by callers
    if !in_mem_vaults.is_empty() {
        let mut vault_json = serde_json::json!({});
        for (col, vault) in &in_mem_vaults {
            vault_json[col] = serde_json::json!({
                "forward": vault.forward.iter().map(|(k, v)| (k.clone(), Value::String(v.clone()))).collect::<serde_json::Map<_, _>>(),
                "reverse": vault.reverse.iter().map(|(k, v)| (k.clone(), Value::String(v.clone()))).collect::<serde_json::Map<_, _>>(),
            });
        }
        write_json_pretty(&token_vault_path, &vault_json)?;
    }
    write_json_pretty(
        &fpe_keys_path,
        &Value::Object(fpe_keys.into_iter().map(|(k, v)| (k, Value::String(v))).collect()),
    )?;
    write_json_pretty(
        &symmetric_keys_path,
        &Value::Object(symmetric_keys.into_iter().map(|(k, v)| (k, Value::String(v))).collect()),
    )?;
    write_json_pretty(
        &fpe_encrypt_keys_path,
        &Value::Object(fpe_encrypt_keys.into_iter().map(|(k, v)| (k, Value::String(v))).collect()),
    )?;

    Ok(())
}
