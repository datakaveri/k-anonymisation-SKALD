# ChunKanon

**ChunKanon** is a scalable, chunk-wise **K-anonymization** tool based on the **Optimal Lattice Anonymization (OLA)** algorithm. It is designed to handle large datasets by processing them in manageable chunks, ensuring data privacy while maintaining utility.

---

## ✨ Features

- ✅ Chunk-wise `k`-anonymization using the Optimal Lattice Anonymization (OLA) method  
- 🔢 Supports **numerical** and **categorical** quasi-identifiers  
- 📦 Efficient **encoding** for sparse numerical attributes  
- 🔁 **Decoding** of generalized encoded values for interpretability  
- 📊 Global histogram merging for optimal bin width selection  
- 🚫 Suppression mechanism to meet `k`-anonymity without distorting data excessively  
- 🧱 Clean architecture separating core logic, generalization, and utilities  
- ⚙️ Fully configurable via a `YAML` file  
- 📁 Logging support and reproducible output structure  

---

## 📦 Installation

```bash
git clone https://github.com/your-username/k-anonymisation-SKALD.git
cd k-anonymisation-SKALD
pip install -r requirements.txt
````

> Python 3.8+ is required. Dependencies are listed in `requirements.txt`.

---

## 🚀 Usage

Run ChunKanon via the command line:

```bash
chunkanon --k 500 --chunks 100 --chunk-dir datachunks
```

### CLI Arguments

| Argument      | Description                                |
| ------------- | ------------------------------------------ |
| `--k`         | Desired k-anonymity level (e.g., 500)      |
| `--chunks`    | Number of chunks to process (e.g., 100)    |
| `--chunk-dir` | Directory containing the dataset chunks    |
| `--config`    | (Optional) Path to custom YAML config file |

---

## ⚙️ Configuration (`config.yaml`)

```yaml
k: 500
number_of_chunks: 100
chunk_directory: datachunks
output_path: generalized_chunk.csv
log_file: log.txt
save_output: true

quasi_identifiers:
  numerical:
    - column: Age
      encode: false
    - column: PIN Code
      encode: true
  categorical:
    - Blood Group
    - Profession

suppression_limit: 0.01
max_number_of_eq_classes: 15000000
bin_width_multiplication_factor:
  Age: 1.5
  PIN Code: 2

hardcoded_min_max:
  Age: [0, 100]
```

---

## 🧪 Example Workflow

1. Prepare your chunked dataset (e.g., `datachunks/KanonMedicalData_chunk1.csv`, ..., `chunk100.csv`)
2. Define your QI attributes in `config.yaml`
3. Run ChunKanon:

```bash
chunkanon --config config.yaml
```

### Output Includes

* `generalized_chunk.csv`: Anonymized first chunk
* `encodings/`: Directory with JSON encoding maps
* `log.txt`: Logs and bin width info
* Console logs: Runtime and generalization details

---

## 📂 Folder Structure

```
k-anonymisation-SKALD/
├── chunkanon/
│   ├── core.py
│   ├── generalization_ri.py
│   ├── generalization_rf.py
│   ├── quasi_identifier.py
│   ├── utils.py
│   └── ...
├── encodings/
├── datachunks/
├── generalized_chunk.csv
├── config.yaml
├── cli.py
├── README.md
└── requirements.txt
```

---

## 🧠 How It Works

1. **Chunk Encoding**: High-cardinality QIs like PIN codes are encoded and stored in JSON format.
2. **OLA Phase 1**: Builds a lattice of bin widths to meet equivalence class constraints.
3. **OLA Phase 2**: Refines bin widths using global histograms from all chunks.
4. **Generalization**: Applies finalized bin widths to generalize the first chunk.


---

## 📜 License

MIT License. See the [LICENSE](LICENSE) file for details.

---

## 👨‍💻 Authors

* **Kailash** — Core Developer


---

## 📣 Acknowledgements

* Based on the principles of the OLA (Optimal Lattice Anonymization) algorithm.
* Uses `pandas`, `numpy`, `PyYAML`, and standard Python libraries.



