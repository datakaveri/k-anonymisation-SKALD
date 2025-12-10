# SKALD

**SKALD** is a scalable, chunk-wise **K-anonymization** tool based on the **Optimal Lattice Anonymization (OLA)** algorithm. It is designed to handle large datasets by processing them in manageable chunks, ensuring data privacy while maintaining utility.

---

## Features

- Chunk-wise `k`-anonymization using the Optimal Lattice Anonymization (OLA) method  
- Supports numerical and categorical quasi-identifiers  
- Efficient encoding for sparse numerical attributes  
- Decoding of generalized encoded values for interpretability  
- Global histogram merging for optimal bin width selection  
- Suppression mechanism to meet `k`-anonymity without excessive data distortion  
- Clean architecture separating core logic, generalization, and utilities  
- Fully configurable via a YAML file  
- Logging support and reproducible output structure  

---

## Installation

```bash
git clone https://github.com/datakaveri/k-anonymisation-SKALD.git
cd k-anonymisation-SKALD
pip install -r requirements.txt
pip install -e .
```

**Requirements:** Python 3.8+ and the dependencies listed in `requirements.txt`.

---

## Usage

Run SKALD from the command line:

```bash
SKALD --config config.yaml [--k 10] [--chunks 5] [--chunk_dir chunks]
```

### CLI Arguments

| Argument      | Description                                 |
| ------------- | ------------------------------------------- |
| `--k`         | Desired k-anonymity level (e.g., 500)       |
| `--chunks`    | Number of chunks to process (e.g., 100)     |
| `--chunk_dir` | Directory containing the dataset chunks     |
| `--config`    | Path to custom YAML config file             |

---

## Configuration File (`config.yaml`)

```yaml
number_of_chunks: 1
k: 10
max_number_of_eq_classes: 15000000
suppression_limit: 0.001

chunk_directory: datachunks
output_path: generalized_chunk1.csv
log_file: log.txt
save_output: true

quasi_identifiers:
  numerical:
    - column: Age
      encode: true
      type: int
    - column: BMI
      encode: true
      type: float
    - column: PIN Code
      encode: true
      type: int
  categorical: 
    - column: Blood Group
    - column: Profession

bin_width_multiplication_factor:
  Age: 2
  BMI: 2
  PIN Code: 2

hardcoded_min_max:
  Age: [19, 85]
  BMI: [12.7, 35.8]
  PIN Code: [560001, 591346]
```

---

## Example Workflow

1. Prepare your chunked dataset (e.g., `datachunks/KanonMedicalData_chunk1.csv`, ..., `chunk100.csv`)
2. Define your quasi-identifiers in `config.yaml`
3. Run SKALD:

```bash
SKALD --config config.yaml
```

### Output

- `generalized_chunk.csv`: Anonymized first chunk  
- `encodings/`: JSON encoding maps  
- `log.txt`: Logging output, bin width info  
- Console: Runtime and generalization details  

---

## Project Structure

```
k-anonymisation-SKALD/
├── SKALD/
│   ├── core.py
│   ├── generalization_ri.py
│   ├── generalization_rf.py
│   ├── quasi_identifier.py
│   ├── utils.py
│   ├── cli.py
│   └── ...
├── encodings/
├── datachunks/
├── generalized_chunk.csv
├── config.yaml
├── README.md
└── requirements.txt
```

---

## How It Works

1. **Chunk Encoding**: Encodes high-cardinality numerical QIs and stores mappings as JSON.  
2. **OLA Phase 1**: Constructs a lattice of bin widths to meet equivalence class constraints.  
3. **OLA Phase 2**: Merges histograms from all chunks to refine bin widths.  
4. **Generalization**: Applies the finalized bin widths to the target chunk for anonymization.  

---

## Authors

- **Kailash R** — Core Developer

---

## Acknowledgements

- Based on the Optimal Lattice Anonymization (OLA) algorithm.  
- Utilizes open-source libraries such as `pandas`, `numpy`, and `PyYAML`.
