use crate::pipeline::bootstrap::{split_csv_line_basic, validation, PipelineError, RuntimeConfig};
use serde_json::Value;
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

fn shift_char_in_alphabet(c: char, alphabet: &[char], shift: usize) -> char {
    if let Some(pos) = alphabet.iter().position(|x| *x == c) {
        alphabet[(pos + shift) % alphabet.len()]
    } else {
        c
    }
}

fn format_preserving_encrypt_general(value: &str, key: &str, column: &str) -> String {
    let seed = fnv1a64(&format!("{column}:{key}"));
    let su = ((seed % 26) as usize).max(1);
    let sl = (((seed / 7) % 26) as usize).max(1);
    let sd = (((seed / 13) % 10) as usize).max(1);
    let upper: Vec<char> = "ABCDEFGHIJKLMNOPQRSTUVWXYZ".chars().collect();
    let lower: Vec<char> = "abcdefghijklmnopqrstuvwxyz".chars().collect();
    let digits: Vec<char> = "0123456789".chars().collect();
    value
        .chars()
        .map(|c| {
            if c.is_ascii_uppercase() {
                shift_char_in_alphabet(c, &upper, su)
            } else if c.is_ascii_lowercase() {
                shift_char_in_alphabet(c, &lower, sl)
            } else if c.is_ascii_digit() {
                shift_char_in_alphabet(c, &digits, sd)
            } else {
                c
            }
        })
        .collect()
}

fn pseudo_encrypt(value: &str, key: &str, column: &str) -> String {
    let seed = fnv1a64(&format!("{column}:{key}"));
    let mut out = String::from("ENC$");
    for (i, b) in value.as_bytes().iter().enumerate() {
        let k = ((seed >> ((i % 8) * 8)) & 0xff) as u8;
        out.push_str(&format!("{:02x}", b ^ k));
    }
    out
}

fn pseudo_fpe_pan(value: &str, key: &str, column: &str) -> String {
    let chars: Vec<char> = value.chars().collect();
    if chars.len() != 10 {
        return value.to_string();
    }
    if !chars[..5].iter().all(|c| c.is_ascii_uppercase())
        || !chars[5..9].iter().all(|c| c.is_ascii_digit())
        || !chars[9].is_ascii_uppercase()
    {
        return value.to_string();
    }
    let seed = fnv1a64(&format!("pan:{column}:{key}"));
    let su1 = ((seed % 26) as usize).max(1);
    let sd = (((seed / 11) % 10) as usize).max(1);
    let su2 = (((seed / 17) % 26) as usize).max(1);
    let upper: Vec<char> = "ABCDEFGHIJKLMNOPQRSTUVWXYZ".chars().collect();
    let digits: Vec<char> = "0123456789".chars().collect();

    let mut out = String::new();
    for c in &chars[..5] {
        out.push(shift_char_in_alphabet(*c, &upper, su1));
    }
    for c in &chars[5..9] {
        out.push(shift_char_in_alphabet(*c, &digits, sd));
    }
    out.push(shift_char_in_alphabet(chars[9], &upper, su2));
    out
}

fn pseudo_fpe_digits(value: &str, key: &str, column: &str) -> String {
    if value.is_empty() || !value.chars().all(|c| c.is_ascii_digit()) {
        return value.to_string();
    }
    let seed = fnv1a64(&format!("digits:{}:{}:{}", value.len(), column, key));
    let sd = ((seed % 10) as usize).max(1);
    let digits: Vec<char> = "0123456789".chars().collect();
    value
        .chars()
        .map(|c| shift_char_in_alphabet(c, &digits, sd))
        .collect()
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
        .ok_or_else(|| validation("PREPROCESSING_FAILED", "Each tokenization entry must be an object", "tokenization"))?;
    let column = obj
        .get("column")
        .and_then(Value::as_str)
        .ok_or_else(|| validation("PREPROCESSING_FAILED", "Tokenization config missing 'column'", "tokenization"))?
        .to_string();
    let prefix = obj.get("prefix").and_then(Value::as_str).unwrap_or("TK-").to_string();
    let digits = obj.get("digits").and_then(Value::as_i64).unwrap_or(6);
    if digits <= 0 {
        return Err(validation("PREPROCESSING_FAILED", "tokenization 'digits' must be > 0", &column));
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
        .ok_or_else(|| validation("PREPROCESSING_FAILED", "Each FPE entry must be an object", "fpe"))?;
    let column = obj
        .get("column")
        .and_then(Value::as_str)
        .ok_or_else(|| validation("PREPROCESSING_FAILED", "FPE config missing 'column'", "fpe"))?
        .to_string();
    let format = obj.get("format").and_then(Value::as_str).unwrap_or("pan").to_string();
    if format != "pan" && format != "digits" {
        return Err(validation("PREPROCESSING_FAILED", "FPE format must be 'pan' or 'digits'", &column));
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
        .ok_or_else(|| validation("PREPROCESSING_FAILED", "Encrypt entry must be string or object", "encrypt"))?;
    if obj.len() != 1 {
        return Err(validation("PREPROCESSING_FAILED", "Encrypt object entry must contain one column key", "encrypt"));
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
        .ok_or_else(|| validation("PREPROCESSING_FAILED", "Each masking entry must be an object", "masking"))?;
    let column = obj
        .get("column")
        .and_then(Value::as_str)
        .ok_or_else(|| validation("PREPROCESSING_FAILED", "Masking config missing 'column'", "masking"))?
        .to_string();

    let masking_char_s = obj
        .get("masking_char")
        .or_else(|| obj.get("masking_character"))
        .and_then(Value::as_str)
        .unwrap_or("*");
    let masking_char = masking_char_s
        .chars()
        .next()
        .ok_or_else(|| validation("PREPROCESSING_FAILED", "masking_char cannot be empty", &column))?;

    let mut characters_to_mask = Vec::new();
    if let Some(arr) = obj.get("characters_to_mask").and_then(Value::as_array) {
        for v in arr {
            let pos = v.as_i64().ok_or_else(|| {
                validation("PREPROCESSING_FAILED", "characters_to_mask must be positive integers", &column)
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

    let token_vault_path = out_dir.join("token_vault.json");
    let mut token_vault: Value = if token_vault_path.exists() {
        serde_json::from_str(&fs::read_to_string(&token_vault_path)?)?
    } else {
        serde_json::json!({})
    };

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
            .ok_or_else(|| validation("PREPROCESSING_FAILED", "Chunk is empty", &chunk_path.display().to_string()))?
            .map_err(|e| validation("PREPROCESSING_FAILED", "Failed reading header", &e.to_string()))?;
        let mut headers = split_csv_line_basic(&header_line);
        let mut rows: Vec<Vec<String>> = Vec::new();
        for line in lines {
            let line = line.map_err(|e| validation("PREPROCESSING_FAILED", "Failed reading row", &e.to_string()))?;
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
                    return Err(validation("PREPROCESSING_FAILED", "Suppression column not found", col));
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
                .ok_or_else(|| validation("PREPROCESSING_FAILED", "Column not found for salted hashing", col))?;
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
                .ok_or_else(|| validation("PREPROCESSING_FAILED", "Column not found for hashing", col))?;
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
                .ok_or_else(|| validation("PREPROCESSING_FAILED", "Column not found for masking", &m.column))?;
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
                .ok_or_else(|| validation("PREPROCESSING_FAILED", "Column not found for charcloak", col))?;
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
                .ok_or_else(|| validation("PREPROCESSING_FAILED", "Column not found for tokenization", &tcfg.column))?;

            if token_vault.get(&tcfg.column).is_none() {
                token_vault[tcfg.column.clone()] = serde_json::json!({ "forward": {}, "reverse": {} });
            }

            for row in &mut rows {
                if idx >= row.len() {
                    continue;
                }
                let v = row[idx].clone();
                if should_skip_value(&v) {
                    continue;
                }
                let col_obj = token_vault
                    .get_mut(&tcfg.column)
                    .and_then(Value::as_object_mut)
                    .ok_or_else(|| validation("PREPROCESSING_FAILED", "Invalid token vault shape", &tcfg.column))?;

                let existing_token = col_obj
                    .get("forward")
                    .and_then(Value::as_object)
                    .and_then(|m| m.get(&v))
                    .and_then(Value::as_str)
                    .map(|s| s.to_string());

                if let Some(tok) = existing_token {
                    row[idx] = tok;
                } else {
                    let reverse_len = col_obj
                        .get("reverse")
                        .and_then(Value::as_object)
                        .map(|m| m.len())
                        .unwrap_or(0);
                    let mut n = (reverse_len as u64) + 1;
                    let token = loop {
                        let candidate = format!("{}{:0width$}", tcfg.prefix, n, width = tcfg.digits);
                        let exists = col_obj
                            .get("reverse")
                            .and_then(Value::as_object)
                            .map(|m| m.contains_key(&candidate))
                            .unwrap_or(false);
                        if !exists {
                            break candidate;
                        }
                        n += 1;
                    };

                    let forward = col_obj
                        .get_mut("forward")
                        .and_then(Value::as_object_mut)
                        .ok_or_else(|| validation("PREPROCESSING_FAILED", "Invalid token vault forward map", &tcfg.column))?;
                    forward.insert(v.clone(), Value::String(token.clone()));

                    let reverse = col_obj
                        .get_mut("reverse")
                        .and_then(Value::as_object_mut)
                        .ok_or_else(|| validation("PREPROCESSING_FAILED", "Invalid token vault reverse map", &tcfg.column))?;
                    reverse.insert(token.clone(), Value::String(v.clone()));

                    row[idx] = token;
                }
            }
        }

        for fcfg in &fpe_cfgs {
            let idx = headers
                .iter()
                .position(|h| h == &fcfg.column)
                .ok_or_else(|| validation("PREPROCESSING_FAILED", "Column not found for FPE", &fcfg.column))?;
            let key = fpe_keys
                .entry(fcfg.column.clone())
                .or_insert_with(|| format!("{:032x}", fnv1a64(&format!("fpekey:{}", fcfg.column))))
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
                    pseudo_fpe_pan(&v, &key, &fcfg.column)
                } else {
                    pseudo_fpe_digits(&v, &key, &fcfg.column)
                };
            }
        }

        for ecfg in &encrypt_cfgs {
            let idx = headers
                .iter()
                .position(|h| h == &ecfg.column)
                .ok_or_else(|| validation("PREPROCESSING_FAILED", "Column not found for encryption", &ecfg.column))?;

            if ecfg.format_preserving {
                let key = fpe_encrypt_keys
                    .entry(ecfg.column.clone())
                    .or_insert_with(|| format!("{:032x}", fnv1a64(&format!("fpeenc:{}", ecfg.column))))
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
                    .or_insert_with(|| format!("{:032x}", fnv1a64(&format!("sym:{}", ecfg.column))))
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

    write_json_pretty(&token_vault_path, &token_vault)?;
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
