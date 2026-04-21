# SKALD

Streaming k-anonymization pipeline built in Rust.
Implements the OLA-1 / OLA-2 lattice algorithms for optimal generalization with minimal information loss.

---

## What it does

1. Reads one CSV from `data/`
2. Applies preprocessing (suppress, hash, mask, encrypt, tokenize, FPE)
3. Computes k-anonymous generalizations using OLA-2 lattice search
4. Writes anonymized output and a structured status payload to `output/`

---

## Quick start — Docker (recommended)

```bash
# 1. Add your CSV
cp your_dataset.csv data/

# 2. Add your config (see Config Schema below)
cp your_config.json config/config.json

# 3. Build and run
docker compose up --build

# 4. Read results
cat output/status.json
cat output/pipeline.log        # full timestamped trace
```

Exit code `0` = success, `1` = error. All detail is in `output/status.json`.

---

## Quick start — local Rust

```bash
# Prerequisites: Rust toolchain (https://rustup.rs)

# Run pipeline
cargo run --manifest-path SKALD/Cargo.toml --release --bin skald_pipeline

# Run tests
cargo test --manifest-path SKALD/Cargo.toml --lib
```

---

## Config schema

Place a single JSON file in `config/`. Full example:

```json
{
  "operations": ["SKALD", "k-anonymity"],
  "data_type": "my_dataset",
  "my_dataset": {
    "output_path":        "anonymized.csv",
    "output_directory":   "output",
    "log_file":           "log.txt",

    "suppress":           ["name", "email"],
    "hashing_with_salt":  ["national_id"],
    "hashing_without_salt": [],
    "masking": [
      { "column": "phone", "masking_char": "*", "characters_to_mask": [1,2,3] }
    ],
    "encrypt":            ["account_number"],
    "charcloak":          [],
    "tokenization": [
      { "column": "patient_id", "prefix": "TK-", "digits": 8 }
    ],
    "fpe": [
      { "column": "pan", "format": "pan" }
    ],

    "quasi_identifiers": {
      "numerical": [
        { "column": "age",     "encode": false, "type": "int" },
        { "column": "zipcode", "encode": false, "type": "int" }
      ],
      "categorical": [
        { "column": "gender" },
        { "column": "blood_group" }
      ]
    },
    "size": {
      "age":     2,
      "zipcode": 100
    },
    "k_anonymize":      { "k": 15 },
    "suppression_limit": 0.05,
    "enable_l_diversity": false
  }
}
```

### Key fields

| Field | Type | Description |
|---|---|---|
| `k_anonymize.k` | int ≥ 1 | Minimum group size for k-anonymity |
| `suppression_limit` | float 0.0–1.0 | Max fraction of records allowed to be suppressed |
| `size.<column>` | int | Bin width (generalization step) for each numerical QI |
| `quasi_identifiers.numerical` | array | Numerical columns used as quasi-identifiers |
| `quasi_identifiers.categorical` | array | Categorical columns used as quasi-identifiers |

---

## Output files

| File | Description |
|---|---|
| `output/status.json` | Structured result payload (see below) |
| `output/pipeline.log` | Timestamped, phase-tagged execution trace |
| `output/<output_path>` | Anonymized CSV (name set in config) |
| `output/equivalence_class_stats.json` | EC size distribution |
| `output/top_ola2_nodes.json` | Top-ranked generalization nodes from OLA-2 |
| `output/token_vault.json` | Token↔value mapping (if tokenization used) |
| `output/fpe_keys.json` | FPE encryption keys (if FPE used) |
| `output/symmetric_keys.json` | Symmetric encryption keys (if encrypt used) |

### `status.json` — success

```json
{
  "status": "success",
  "phase":  "done",
  "outputs": {
    "total_records":           50000,
    "final_rf":                [2, 100],
    "lowest_dm_star":          1234.5,
    "num_equivalence_classes": 312,
    "chunk_count":             4,
    "final_output_path":       "output/anonymized.csv"
  },
  "log_file": "output/pipeline.log"
}
```

### `status.json` — error

```json
{
  "status": "error",
  "error": {
    "code":             "DATA_COLUMN_MISSING",
    "message":          "A quasi-identifier column was not found in the CSV header",
    "details":          "age",
    "suggested_fix":    "Column names are case-sensitive — verify they match the config exactly.",
    "http_status_code": 422
  },
  "log_file": "output/pipeline.log"
}
```

---

## Error codes

| HTTP | Code | Meaning |
|---|---|---|
| 400 | `CONFIG_NOT_FOUND` | No JSON file in `config/` |
| 400 | `CONFIG_PARSE_ERROR` | Config JSON is malformed |
| 400 | `CONFIG_MISSING_FIELD` | Required field absent |
| 400 | `CONFIG_INVALID_VALUE` | Field value out of range |
| 422 | `DATA_DIR_MISSING` | `data/` directory not found |
| 422 | `DATA_NO_CSV` | No CSV file in `data/` |
| 422 | `DATA_EMPTY` | CSV is empty |
| 422 | `DATA_COLUMN_MISSING` | QI column not in CSV header |
| 422 | `PREPROCESS_COLUMN_MISSING` | Preprocessing target column not found |
| 422 | `PREPROCESS_CONFIG_INVALID` | Malformed preprocessing entry |
| 422 | `ANON_INFEASIBLE` | k-anonymity unsatisfiable — raise `suppression_limit` or lower `k` |
| 422 | `ANON_NO_QIS` | No quasi-identifiers defined |
| 500 | `IO_READ_FAILED` | File not found or unreadable |
| 500 | `IO_WRITE_FAILED` | Disk full or output not writable |
| 500 | `IO_PERMISSION_DENIED` | Permission denied |
| 500 | `INTERNAL_ERROR` | Unexpected error — check `output/pipeline.log` |

Full reference with suggested fixes: [`error_codes.txt`](error_codes.txt)

---

## Project layout

```
config/                    Runtime config JSON (volume-mounted)
data/                      Input CSV (volume-mounted)
output/                    Pipeline outputs (volume-mounted)
SKALD/
  Cargo.toml               Rust crate manifest
  src/
    bin/skald_pipeline.rs  Binary entry point
    pipeline/
      bootstrap.rs         Config parsing, Logger, error types, CSV utilities
      pipeline.rs          Orchestrator — phase-tagged logging with elapsed time
      anonymization.rs     OLA-1, OLA-2, histogram, generalization
      preprocess.rs        Suppress, hash, mask, encrypt, tokenize, FPE
      pyffx_compat.rs      Pure-Rust pyffx-compatible FPE (HMAC-SHA1 Feistel)
      entry.rs             Error mapping → structured status.json
benchmark/
  generate_data.py         Streaming synthetic CSV generator (no RAM limit)
  run_benchmark.sh         Automated benchmark matrix (Rust and Python)
  compare_results.py       Side-by-side comparison report + speedup heatmap
  run_overnight.sh         nohup launcher for overnight runs
```

---

## Preprocessing operations

| Operation | Config key | Description |
|---|---|---|
| Column suppression | `suppress` | Drops column entirely from output |
| Salted hashing | `hashing_with_salt` | FNV-1a with per-column salt |
| Unsalted hashing | `hashing_without_salt` | FNV-1a, deterministic |
| Position masking | `masking` | Replaces specified character positions with mask char |
| Pseudo-encryption | `encrypt` | HMAC-SHA256 keystream XOR, `ENC$`-prefixed hex |
| Format-preserving encrypt | `encrypt` + `format_preserving: true` | Preserves character class layout |
| Character cloaking | `charcloak` | Random character within same class (digit/upper/lower) |
| Tokenization | `tokenization` | Stable `prefix + sequential ID`, vault persisted to `output/` |
| FPE — PAN | `fpe` + `"format": "pan"` | pyffx-compatible Feistel on ABCDE1234F format |
| FPE — digits | `fpe` + `"format": "digits"` | pyffx-compatible Feistel on digit strings |

---

## Benchmarking

```bash
# Rust branch
bash benchmark/run_benchmark.sh --output benchmark/results_rust.json

# Python (master) branch in VM
bash benchmark/run_benchmark.sh --output benchmark/results_python.json

# Compare
python benchmark/compare_results.py \
    benchmark/results_rust.json \
    benchmark/results_python.json
```
