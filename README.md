# SKALD

SKALD is a scalable, chunk-wise k‑anonymization pipeline based on the OLA (Optimal Lattice Anonymization) approach. It is designed for large CSVs, with streaming preprocessing and sparse histograms to keep memory stable on smaller machines.

## Highlights

- Chunk-based processing for large CSVs
- Numeric + categorical quasi-identifiers (QI)
- Streaming preprocessing (suppress, hash, mask, encrypt, charcloak, tokenize, FPE)
- Numerical encoding for sparse attributes
- Sparse histogram storage (only non‑zero equivalence classes)
- OLA‑1 for initial bin widths; OLA‑2 for final RF selection
- Detailed logging and machine‑readable status output

## What SKALD Does (Brief)

At a high level, SKALD takes a CSV, applies configurable privacy transformations (suppression, hashing, masking, encryption), then computes k‑anonymity generalization using OLA. It operates on chunks so large files can be processed without loading the full dataset into memory.

## Requirements

- Python 3.8+
- Dependencies in `requirements.txt`

## Quick Start

1. Put **one CSV** in `k-anonymisation-SKALD/data/`.
2. Put **one JSON config** in `k-anonymisation-SKALD/config/`.
3. Run:

```bash
python3 -m SKALD.core
```

The pipeline reads the first JSON file in `config/`, builds a temporary YAML config internally, and runs the full pipeline.

## Configuration (JSON)

The entrypoint expects a JSON file with this structure:

- `operations`: list of operations (e.g., `"SKALD"`, `"k-anonymity"`).
- `data_type`: name of the dataset block to use.
- `<data_type>`: the dataset‑specific settings.

**Example**

```json
{
  "operations": ["SKALD", "k-anonymity"],
  "data_type": "BeneficiaryData",
  "BeneficiaryData": {
    "enable_k_anonymity": true,
    "enable_l_diversity": false,
    "output_path": "generalized.csv",
    "output_directory": "output",
    "log_file": "log.txt",

    "suppress": ["FULLNAMEENGLISH"],
    "hashing_with_salt": [],
    "hashing_without_salt": [],
    "masking": [],
    "encrypt": [],

    "quasi_identifiers": {
      "categorical": [],
      "numerical": [
        {"column": "AGE", "encode": false, "type": "int"},
        {"column": "DISTRICTCODE", "encode": false, "type": "int"}
      ]
    },

    "k_anonymize": {"k": 2},
    "l_diversity": {"l": 1},
    "sensitive_parameter": null,
    "size": {"AGE": 2, "DISTRICTCODE": 2},
    "suppression_limit": 0.01
  }
}
```

**Notes**

- `size` contains multiplication factors (>1) used by OLA‑1/OLA‑2.
- `suppression_limit` is a fraction from 0 to 1.
- `enable_l_diversity` and `sensitive_parameter` are supported in config; k‑anonymity is enforced, l‑diversity hooks exist but are not strictly enforced in the current check path.

## Outputs

All outputs go to `output/` by default:

- `output/status.json`: status + error details
- `output/log.txt`: detailed logs and timings
- `output/equivalence_class_stats.json`: equivalence class stats
- `output/<output_path>`: final generalized CSV
- `encodings/`: numerical encoding maps

## How the Pipeline Works

1. **Config load** from `config/*.json` → temporary YAML
2. **Data validation** (`data/` must contain a single CSV)
3. **Chunking** (approx. 1/4 of RAM per chunk)
4. **Preprocessing** (suppress/hash/mask/encrypt/etc.)
5. **Numerical encoding** for selected QIs
6. **OLA‑1** to compute `initial_ri`
7. **Histogram build** (streaming, sparse)
8. **OLA‑2** to choose final RF
9. **Generalization** and output merge

## Preprocessing Steps (Detailed)

Preprocessing happens per chunk before k‑anonymization. The steps are optional and driven by config:

1. **Suppress**  
   Drops specified columns entirely.

2. **Hashing**  
   - With salt: irreversible hashing with a per‑run salt.  
   - Without salt: deterministic hashing (same input → same output).

3. **Masking**  
   Redacts patterns in string columns (e.g., phone/email), leaving only partial information.

4. **Encryption**  
   Symmetric encryption for sensitive fields, with keys written to `output/` (e.g., `symmetric_keys.json`).

5. **Charcloak / Tokenization / FPE**  
   Specialized transforms for preserving formats or generating tokens while hiding original values.

Each transformation is applied in the order above. All transformations run chunk‑wise to keep memory bounded.

## Performance and Memory

- Histogram construction streams each chunk in batches and stores **only non‑zero bins**.
- You can lower per‑batch memory by setting:

```bash
SKALD_CHUNK_PROCESSING_ROWS=50000 python3 -m SKALD.core
```

Lower values reduce memory at the cost of runtime.

## Troubleshooting

- **Process killed without error**: likely OS OOM kill. Reduce `SKALD_CHUNK_PROCESSING_ROWS` and/or use a smaller input CSV.
- **Config validation errors**: check key names and required fields. The pipeline validates against `SKALD/config_validation.py`.
- **Mixed dtype warnings**: ensure numeric QIs are clean. The pipeline coerces numeric values but will log non‑numeric rows.

## Project Layout

```
k-anonymisation-SKALD/
├── SKALD/
│   ├── core.py
│   ├── generalization_ri.py
│   ├── generalization_rf.py
│   ├── chunking.py
│   ├── chunk_processing.py
│   ├── preprocess.py
│   └── ...
├── config/
├── data/
├── output/
├── encodings/
└── README.md
```

## Development Notes

- The default entrypoint is `python3 -m SKALD.core`.
- `SKALD_main.py` provides an integration‑style entrypoint for external pipelines.

---
