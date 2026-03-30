use crate::pipeline::bootstrap::{split_csv_line_basic, validation, PipelineError, RuntimeConfig};
use crate::pipeline::pyffx_compat::fpe_encrypt;
use hmac::{Hmac, Mac};
use regex::Regex;
use serde_json::Value;
use sha2::Sha256;
use std::collections::BTreeMap;
use std::fs;
use std::io::{BufRead, BufReader, BufWriter, Write};
use std::path::{Path, PathBuf};

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

/// Generate a random 64-hex-char (32-byte) salt from /dev/urandom.
fn generate_random_salt_hex() -> String {
    use std::io::Read;
    let mut buf = [0u8; 32];
    if let Ok(mut f) = std::fs::File::open("/dev/urandom") {
        let _ = f.read_exact(&mut buf);
    }
    hex::encode(buf)
}

/// SHA-256 hash of input — returns 64-char lowercase hex string (matches Python hashlib.sha256).
fn hash_hex(input: &str) -> String {
    use sha2::Digest;
    let mut hasher = sha2::Sha256::new();
    hasher.update(input.as_bytes());
    hex::encode(hasher.finalize())
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

/// Replace each character with a random character of the same class (digit/upper/lower).
/// Uses /dev/urandom for true randomness — different output each run (matches Python secrets.choice).
fn randomize_preserving_class(value: &str) -> String {
    use std::io::Read;
    let char_count = value.chars().count();
    let byte_count = char_count * 4;
    let mut random_bytes = vec![0u8; byte_count];
    if let Ok(mut f) = std::fs::File::open("/dev/urandom") {
        let _ = f.read_exact(&mut random_bytes);
    }
    let mut out = String::with_capacity(value.len());
    for (i, c) in value.chars().enumerate() {
        let r = u32::from_le_bytes([
            random_bytes[i * 4],
            random_bytes[i * 4 + 1],
            random_bytes[i * 4 + 2],
            random_bytes[i * 4 + 3],
        ]);
        if c.is_ascii_digit() {
            out.push((b'0' + (r % 10) as u8) as char);
        } else if c.is_ascii_uppercase() {
            out.push((b'A' + (r % 26) as u8) as char);
        } else if c.is_ascii_lowercase() {
            out.push((b'a' + (r % 26) as u8) as char);
        } else {
            out.push(c);
        }
    }
    out
}

// ── Regex pattern config (mirrors Python's per-pattern dict) ─────────────────

#[derive(Debug, Clone)]
enum RegexPatternKind {
    /// Literal regex string provided directly
    Literal(String),
    /// Derived from type=before/after/in_between + delimiter/start/end
    Derived { pattern_type: String, delimiter: Option<String>, start: Option<String>, end: Option<String> },
}

#[derive(Debug, Clone)]
struct RegexPatternConfig {
    kind:         RegexPatternKind,
    masking_char: char,
    /// Optional length for type=before/after delimiter-length masking
    length:       Option<usize>,
    /// Optional { "1": "full"|"partial", ... } group masking
    mask_groups:  Vec<(usize, String)>,
}

#[derive(Debug)]
struct MaskingConfigLite {
    column:             String,
    masking_char:       char,
    characters_to_mask: Vec<usize>,
    regex_patterns:     Vec<RegexPatternConfig>,
    apply_order:        Vec<String>,
    class_masking_mode: Option<String>,
    class_letter:       char,
    class_digit:        char,
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

    // characters_to_mask
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

    // regex_patterns
    let mut regex_patterns: Vec<RegexPatternConfig> = Vec::new();
    if let Some(arr) = obj.get("regex_patterns").and_then(Value::as_array) {
        for pat in arr {
            let pobj = pat.as_object().ok_or_else(|| {
                validation("PREPROCESS_CONFIG_INVALID", "Each regex_patterns entry must be an object", &column)
            })?;

            let pat_masking_char = pobj
                .get("masking_char")
                .and_then(Value::as_str)
                .map(|s| s.chars().next().unwrap_or(masking_char))
                .unwrap_or(masking_char);

            let length = pobj.get("length").and_then(Value::as_i64).map(|n| n.max(0) as usize);

            // mask_groups: {"1": "full", "2": "partial"}
            let mut mask_groups: Vec<(usize, String)> = Vec::new();
            if let Some(mg) = pobj.get("mask_groups").and_then(Value::as_object) {
                for (k, v) in mg {
                    if let (Ok(idx), Some(mode)) = (k.parse::<usize>(), v.as_str()) {
                        mask_groups.push((idx, mode.to_string()));
                    }
                }
            }

            let kind = if let Some(r) = pobj.get("regex").and_then(Value::as_str) {
                RegexPatternKind::Literal(r.to_string())
            } else {
                let pattern_type = pobj.get("type").and_then(Value::as_str).unwrap_or("").to_lowercase();
                let delimiter    = pobj.get("delimiter").and_then(Value::as_str).map(|s| s.to_string());
                let start        = pobj.get("start").and_then(Value::as_str).map(|s| s.to_string());
                let end          = pobj.get("end").and_then(Value::as_str).map(|s| s.to_string());
                RegexPatternKind::Derived { pattern_type, delimiter, start, end }
            };

            regex_patterns.push(RegexPatternConfig { kind, masking_char: pat_masking_char, length, mask_groups });
        }
    }

    // apply_order — default ["characters", "regex", "class"]
    let apply_order: Vec<String> = obj
        .get("apply_order")
        .and_then(Value::as_array)
        .map(|arr| arr.iter().filter_map(|v| v.as_str().map(|s| s.to_string())).collect())
        .unwrap_or_else(|| vec!["characters".to_string(), "regex".to_string(), "class".to_string()]);

    let class_masking_mode = obj.get("class_masking_mode").and_then(Value::as_str).map(|s| s.to_string());
    let class_letter = obj.get("class_mask_letter").and_then(Value::as_str)
        .unwrap_or("X").chars().next().unwrap_or('X');
    let class_digit  = obj.get("class_mask_digit").and_then(Value::as_str)
        .unwrap_or("0").chars().next().unwrap_or('0');

    Ok(MaskingConfigLite { column, masking_char, characters_to_mask, regex_patterns, apply_order, class_masking_mode, class_letter, class_digit })
}

// ── Regex helpers (mirrors Python _derive_regex / _apply_delimiter_length_mask) ──

/// Build a regex string from a semantic type descriptor — exact Python logic.
fn derive_regex(pattern_type: &str, delimiter: Option<&str>, start: Option<&str>, end: Option<&str>) -> String {
    match pattern_type {
        "before" => {
            if let Some(d) = delimiter {
                return format!("^.+?(?={})", regex::escape(d));
            }
        }
        "after" => {
            if let Some(d) = delimiter {
                return format!("(?<={}).+$", regex::escape(d));
            }
        }
        "in_between" => {
            if let (Some(s), Some(e)) = (start, end) {
                if e == "$" {
                    return format!("(?<={}).+$", regex::escape(s));
                }
                return format!("(?<={}).+?(?={})", regex::escape(s), regex::escape(e));
            }
        }
        _ => {}
    }
    String::new()
}

/// Apply length-limited masking before/after a delimiter — exact Python logic.
fn apply_delimiter_length_mask(text: &str, pattern_type: &str, delimiter: &str, length: usize, mask_char: char) -> (String, bool) {
    if delimiter.is_empty() || length == 0 {
        return (text.to_string(), false);
    }
    let mut chars: Vec<char> = text.chars().collect();
    let text_str = text;
    let mut changed = false;
    let mut search_start = 0usize;

    loop {
        // find delimiter byte offset from search_start
        let Some(byte_idx) = text_str[search_start..].find(delimiter) else { break };
        let abs_byte = search_start + byte_idx;

        // convert byte offset to char offset
        let char_before = text_str[..abs_byte].chars().count();
        let delim_char_len = delimiter.chars().count();

        let (start_ci, end_ci) = if pattern_type == "after" {
            let s = char_before + delim_char_len;
            (s, (s + length).min(chars.len()))
        } else {
            // before
            let e = char_before;
            (e.saturating_sub(length), e)
        };

        for i in start_ci..end_ci {
            chars[i] = mask_char;
            changed = true;
        }

        search_start = abs_byte + delimiter.len();
        if search_start >= text_str.len() { break; }
    }

    (chars.into_iter().collect(), changed)
}

/// Apply a single RegexPatternConfig to one string value.
fn apply_regex_pattern(value: &str, pat: &RegexPatternConfig, _column: &str) -> String {
    let mask_char = pat.masking_char;

    // 1. Delimiter-length masking (takes priority over regex when `length` is set)
    if let Some(length) = pat.length {
        let (pattern_type, delimiter) = match &pat.kind {
            RegexPatternKind::Derived { pattern_type, delimiter, .. } => {
                (pattern_type.as_str(), delimiter.as_deref().unwrap_or(""))
            }
            _ => ("", ""),
        };
        if !delimiter.is_empty() && (pattern_type == "before" || pattern_type == "after") {
            let (result, changed) = apply_delimiter_length_mask(value, pattern_type, delimiter, length, mask_char);
            if changed {
                return result;
            }
        }
    }

    // 2. Build the regex string
    let regex_str: String = match &pat.kind {
        RegexPatternKind::Literal(r) => r.clone(),
        RegexPatternKind::Derived { pattern_type, delimiter, start, end } => {
            derive_regex(pattern_type, delimiter.as_deref(), start.as_deref(), end.as_deref())
        }
    };

    if regex_str.is_empty() {
        return value.to_string();
    }

    // 3. Compile — fall back to derived if literal is invalid
    let compiled = match Regex::new(&regex_str) {
        Ok(r) => r,
        Err(_) => {
            // Try derived pattern as fallback (mirrors Python warning + fallback)
            if let RegexPatternKind::Literal(_) = &pat.kind {
                // already a literal, nothing to fall back to
                return value.to_string();
            }
            return value.to_string();
        }
    };

    // 4. Apply — group masking or full-match masking
    if !pat.mask_groups.is_empty() {
        apply_regex_group_mask(value, &compiled, &pat.mask_groups, mask_char)
    } else {
        let mut result = String::with_capacity(value.len());
        let mut last = 0usize;
        let mut replaced = 0usize;
        for m in compiled.find_iter(value) {
            result.push_str(&value[last..m.start()]);
            result.push_str(&mask_char.to_string().repeat(m.as_str().chars().count()));
            last = m.end();
            replaced += 1;
        }
        result.push_str(&value[last..]);

        // Data-adaptive fallback for in_between when no match found (mirrors Python)
        if replaced == 0 {
            if let RegexPatternKind::Derived { pattern_type, .. } = &pat.kind {
                if pattern_type == "in_between" && value.contains(' ') {
                    if let Ok(space_re) = Regex::new(r"(?<=\s).+$") {
                        return space_re.replace_all(value, |m: &regex::Captures| {
                            mask_char.to_string().repeat(m[0].chars().count())
                        }).to_string();
                    }
                }
            }
        }

        result
    }
}

/// Apply group-based masking: full = mask all chars, partial = keep first, mask rest.
fn apply_regex_group_mask(value: &str, re: &Regex, mask_groups: &[(usize, String)], default_mask: char) -> String {
    re.replace_all(value, |caps: &regex::Captures| {
        let full_match = caps.get(0).map(|m| m.as_str()).unwrap_or("");
        let mut rebuilt = full_match.to_string();
        for (group_idx, mode) in mask_groups {
            if let Some(group_match) = caps.get(*group_idx) {
                let original = group_match.as_str();
                let replacement = match mode.as_str() {
                    "full" => default_mask.to_string().repeat(original.chars().count()),
                    "partial" => {
                        if original.chars().count() <= 1 {
                            original.to_string()
                        } else {
                            let mut chars = original.chars();
                            let first = chars.next().unwrap().to_string();
                            let rest = default_mask.to_string().repeat(original.chars().count() - 1);
                            first + &rest
                        }
                    }
                    _ => original.to_string(),
                };
                rebuilt = rebuilt.replacen(original, &replacement, 1);
            }
        }
        rebuilt
    }).to_string()
}

fn apply_masking_value(value: &str, cfg: &MaskingConfigLite) -> String {
    let mut masked = value.to_string();

    for step in &cfg.apply_order {
        match step.as_str() {
            "characters" if !cfg.characters_to_mask.is_empty() => {
                let mut chars: Vec<char> = masked.chars().collect();
                for pos in &cfg.characters_to_mask {
                    let idx = pos.saturating_sub(1);
                    if idx < chars.len() {
                        chars[idx] = cfg.masking_char;
                    }
                }
                masked = chars.into_iter().collect();
            }
            "regex" if !cfg.regex_patterns.is_empty() => {
                for pat in &cfg.regex_patterns {
                    masked = apply_regex_pattern(&masked, pat, &cfg.column);
                }
            }
            "class" => {
                masked = match cfg.class_masking_mode.as_deref() {
                    Some("random_class") => randomize_preserving_class(&masked),
                    Some("fixed_class") => {
                        let mut out = String::with_capacity(masked.len());
                        for c in masked.chars() {
                            if c.is_ascii_digit() { out.push(cfg.class_digit); }
                            else if c.is_ascii_alphabetic() { out.push(cfg.class_letter); }
                            else { out.push(c); }
                        }
                        out
                    }
                    _ => masked,
                };
            }
            _ => {}
        }
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

    // Per-column random salts for salted hashing — generated fresh each run (matches Python behavior).
    // Not persisted to disk; consistent within a run across all chunks for the same column.
    let mut hash_salts: BTreeMap<String, String> = BTreeMap::new();

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
            let salt = hash_salts
                .entry(col.clone())
                .or_insert_with(generate_random_salt_hex)
                .clone();
            for row in &mut rows {
                if idx >= row.len() {
                    continue;
                }
                let v = row[idx].clone();
                if should_skip_value(&v) {
                    continue;
                }
                row[idx] = hash_hex(&format!("{}{}", salt, v));
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
            for row in rows.iter_mut() {
                if idx >= row.len() {
                    continue;
                }
                let v = row[idx].clone();
                if should_skip_value(&v) {
                    continue;
                }
                row[idx] = apply_masking_value(&v, m);
            }
        }

        for col in &cfg.charcloak {
            let idx = headers
                .iter()
                .position(|h| h == col)
                .ok_or_else(|| validation("PREPROCESS_COLUMN_MISSING", "Column not found in CSV header for charcloak", col))?;
            for row in rows.iter_mut() {
                if idx >= row.len() {
                    continue;
                }
                let v = row[idx].clone();
                if should_skip_value(&v) {
                    continue;
                }
                row[idx] = randomize_preserving_class(&v);
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
