# SKALD (Rust Pipeline)

Standalone Rust implementation of the SKALD anonymization pipeline.

## What This Repo Runs

- Reads input CSV from `data/`
- Reads pipeline config JSON from `config/`
- Writes runtime outputs to `output/`
- Uses OLA-1 + OLA-2 logic for RF selection
- Applies preprocessing + generalization in Rust

## Project Layout

- `SKALD/Cargo.toml` Rust crate manifest
- `SKALD/src/bin/skald_pipeline.rs` executable entrypoint
- `SKALD/src/pipeline/` pipeline modules
- `config/` runtime config JSON files
- `data/` input CSV files
- `output/` generated outputs

## Prerequisites

- Rust toolchain (`cargo`, `rustc`)

## Run Pipeline

From repository root:

```bash
cargo run --manifest-path SKALD/Cargo.toml --bin skald_pipeline
```

## Run Tests

```bash
cargo test --manifest-path SKALD/Cargo.toml --lib
```

## Expected Outputs

After successful run, check:

- `output/status.json`
- `output/generalized_*.csv` (based on config output path)
- `output/equivalence_class_stats.json`
- `output/top_ola2_nodes.json`
- key/vault outputs when preprocessing uses tokenization/FPE/encryption

## Notes

- Current flow is Rust-first; legacy Python modules were removed from active path.
- Keep `config/` and `data/` under version control for reproducible runs.
