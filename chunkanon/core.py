import os
import time
import pandas as pd
import yaml
from chunkanon.quasi_identifier import QuasiIdentifier
from chunkanon.generalization_ri import OLA_1
from chunkanon.generalization_rf import OLA_2
from chunkanon.utils import get_progress_iter, log_to_file, format_time

def run_pipeline(config_path="config.yaml"):
    
    with open(config_path, "r") as f:
        config = yaml.safe_load(f)

    k = config.get("k", 500)
    chunk_dir = config.get("chunk_directory", "datachunks")
    output_path = config.get("output_path", "generalized_chunk1.csv")
    log_file = config.get("log_file", "log.txt")
    save_output = config.get("save_output", True)

    numerical_columns = config["quasi_identifiers"]["numerical"]
    categorical_columns = config["quasi_identifiers"]["categorical"]
    all_quasi_columns = numerical_columns + categorical_columns

    start_time = time.time()
    chunk_files = sorted([
        f for f in os.listdir(chunk_dir)
        if f.startswith("KanonMedicalData_chunk2") and f.endswith(".csv")
    ])
    n = len(chunk_files)
    print(f"Processing {n} chunks from {chunk_dir}...")
    print("Selected quasi-identifiers:", all_quasi_columns)

    unique_pins = set()
    for filename in chunk_files:
        chunk = pd.read_csv(os.path.join(chunk_dir, filename))
        unique_pins.update(chunk["PIN Code"].unique())

    sorted_pins = sorted(unique_pins)
    pin_encoding = {pin: idx for idx, pin in enumerate(sorted_pins)}
    print(f"Total unique PIN Codes found: {len(pin_encoding)}")

    for i in get_progress_iter(range(n), desc="Encoding PINs"):
        chunk_path = os.path.join(chunk_dir, chunk_files[i])
        chunk = pd.read_csv(chunk_path)
        chunk["encoded_PIN"] = chunk["PIN Code"].map(pin_encoding)
        chunk.to_csv(chunk_path, index=False)

    hardcoded_min_max = config.get("hardcoded_min_max", {
        "Age": [19, 85],
        "BMI": [12.7, 35.8],
        "encoded_PIN": [0, len(pin_encoding)]
    })

    quasi_identifiers = []
    for col in all_quasi_columns:
        if col in categorical_columns:
            qi = QuasiIdentifier(col, is_categorical=True)
        else:
            min_val, max_val = hardcoded_min_max[col]
            qi = QuasiIdentifier(col, is_categorical=False, min_value=min_val, max_value=max_val)
        quasi_identifiers.append(qi)

    print("\nBuilding initial tree and finding Ri values...")
    ola_1 = OLA_1(quasi_identifiers, n, max_equivalence_classes=15000000, doubling_step=2)
    ola_1.build_tree()
    initial_ri = ola_1.find_smallest_passing_ri(n)
    initial_ri = ola_1.get_optimal_ri()
    print("Initial bin widths (Ri):", initial_ri)
    log_to_file(f"Initial bin widths (Ri): {initial_ri}", log_file)

    ola_2 = OLA_2(quasi_identifiers, doubling_step=2)
    print("\nBuilding second tree with initial Ri values...")
    ola_2.build_tree(initial_ri)

    print("\nProcessing data in chunks for histograms...")
    histograms = []
    for i in range(n):
        chunk = pd.read_csv(os.path.join(chunk_dir, chunk_files[i]))
        chunk_histogram = ola_2.process_chunk(chunk, initial_ri)
        histograms.append(chunk_histogram)
        print(f"Processed chunk {i+1}/{n} for histograms.")
    print("Histograms collected.")

    print("\nMerging histograms and finding final bin widths...")
    global_histogram = ola_2.merge_histograms(histograms)
    final_rf = ola_2.get_final_binwidths(global_histogram, k)
    print("Final bin widths (RF):", final_rf)
    log_to_file(f"Final bin widths (RF): {final_rf}", log_file)

    if save_output:
        print("\nGeneralizing first chunk based on RF and decoding PIN ranges...")
        first_chunk = pd.read_csv(os.path.join(chunk_dir, chunk_files[0]))
        encoded_to_pin = {i: pin for i, pin in enumerate(sorted_pins)}
        generalized_chunk = ola_2.generalize_chunk(first_chunk, final_rf, encoded_to_pin)
        generalized_chunk.to_csv(output_path, index=False)
        print(f"Generalized first chunk saved to: {output_path}")

        with open("encoded_pins.txt", 'w') as filedata:
            for pin in sorted_pins:
                filedata.write("%s\n" % pin)

    elapsed_time = time.time() - start_time
    h, m, s = format_time(elapsed_time)
    print(f"\nTotal time taken: {h}h {m}m {s}s")
    log_to_file(f"Chunks: {n}, k: {k}", log_file)
    log_to_file(f"Total time taken: {h}h {m}m {s}s", log_file)

    return final_rf, elapsed_time
