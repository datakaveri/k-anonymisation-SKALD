import os
import time
import yaml
import logging
import psutil
import pandas as pd
import numpy as np
import json

# Import necessary components from SKALD
from SKALD.quasi_identifier import QuasiIdentifier
from SKALD.generalization_ri import OLA_1
from SKALD.generalization_rf import OLA_2
from SKALD.preprocess import suppress, pseudonymize
from SKALD.utils import get_progress_iter, log_to_file, format_time, ensure_folder
from SKALD.config_validation import load_config

def log_performance(logger,step_name: str, start_time: float):
    """Logs elapsed time and memory usage for a step."""
    process = psutil.Process(os.getpid())
    mem_mb = process.memory_info().rss / (1024 * 1024)

    elapsed = time.time() - start_time
    logger.info(f"[{step_name}] Time taken: {elapsed:.2f} sec | Memory: {mem_mb:.2f} MB")

def run_pipeline(config_path="config.yaml", k=None, chunks=None, chunk_dir=None):
    config = load_config(config_path)
    log_file = "log.txt"
    if not os.path.exists(log_file):
        with open(log_file, "w") as f:
            f.write("=== SKALD Log File Created ===\n")

    logging.basicConfig(
        filename=log_file,
        level=logging.INFO,
        format="%(asctime)s - %(levelname)s - %(message)s"
    )
    logger = logging.getLogger("SKALD")
    

    # Override config if parameters provided
    if k is not None:
        config.k = k
    if chunks is not None:
        config.number_of_chunks = chunks
    if chunk_dir is not None:
        config.chunk_directory = chunk_dir

    n = config.number_of_chunks
    k = config.k
    l = config.l
    chunk_dir = config.chunk_directory
    output_path = config.output_path
    suppression_limit = config.suppression_limit
    max_equivalence_classes = config.max_number_of_eq_classes
    suppressed_columns = config.suppress
    pseudonymized_columns = config.pseudonymize
    sensitive_parameter = config.sensitive_parameter
    log_file = config.log_file
    save_output = config.save_output
    categorical_columns = [cat_qi.column for cat_qi in config.quasi_identifiers.categorical]

    numerical_columns_info = [
        {"column": num_qi.column, "encode": num_qi.encode, "type": num_qi.type}
        for num_qi in config.quasi_identifiers.numerical
    ]

    hardcoded_min_max = config.hardcoded_min_max
    multiplication_factors = config.bin_width_multiplication_factor

    if not os.path.exists(chunk_dir):
        raise FileNotFoundError(f"Chunk directory not found: {chunk_dir}")

    all_files = sorted([f for f in os.listdir(chunk_dir) if f.endswith(".csv")])
    if not all_files:
        raise ValueError("No CSV files found in the chunk directory.")
    chunk_files = all_files[:n]

    print(f"Processing {n} chunks from {chunk_dir}...")

    encoding_maps = {}
    encoding_dir = "encodings"
    ensure_folder(encoding_dir)

    start_time = time.time()
    logger.info(f"Starting SKALD pipeline: chunks={n}, k={k}, l={l}")
    # === UTILITY: find max decimal places ===
    def find_max_decimal_places(series):
        decimals = series.dropna().map(lambda x: len(str(x).split(".")[1]) if "." in str(x) else 0)
        return decimals.max()

    # === ENCODE NUMERICAL COLUMNS ACROSS ALL CHUNKS ===
    for info in numerical_columns_info:
        column = info["column"]
        encode = info.get("encode", False)
        column_type = info.get("type", "float")

        if not encode:
            continue

        all_values = []

        # Collect all values from all chunks for consistent encoding
        for filename in chunk_files:
            chunk = pd.read_csv(os.path.join(chunk_dir, filename))
            if suppressed_columns:
                chunk = suppress(chunk, suppressed_columns)
            if pseudonymized_columns:
                chunk = pseudonymize(chunk, pseudonymized_columns)

            values = chunk[column].dropna()
            if column_type == "float":
                decimal_places = find_max_decimal_places(values)
                multiplier = 10 ** decimal_places
                values = (values * multiplier).round().astype(int)
            else:
                multiplier = 1
                values = values.astype(int)

            all_values.extend(values.tolist())

        # Create encoding map: value -> code starting from 0
        unique_sorted = sorted(set(all_values))
        encoding_map = {val: idx+1 for idx, val in enumerate(unique_sorted)}
        decoding_map = {idx: int(val) for val, idx in encoding_map.items()}

        encoding_maps[column] = {
            "encoding_map": encoding_map,
            "decoding_map": decoding_map,
            "multiplier": int(multiplier),
            "type": column_type
        }

        # Save decoding map for future use
        with open(os.path.join(encoding_dir, f"{column.replace(' ','_').lower()}_encoding.json"), "w") as f:
            json.dump(encoding_maps[column], f, indent=4)

    # === DEFINE QUASI-IDENTIFIERS ===
    quasi_identifiers = []
    all_quasi_columns = []

    for info in numerical_columns_info:
        column = info["column"]
        encode = info.get("encode", False)

        encoded_column = f"{column}_encoded" if encode else column
        all_quasi_columns.append(encoded_column)

        if encode:
            min_val, max_val = 1, len(encoding_maps[column]["encoding_map"])
        else:
            min_val, max_val = hardcoded_min_max.get(column, (0, 0))

        quasi_identifiers.append(
            QuasiIdentifier(
                encoded_column,
                is_categorical=False,
                is_encoded=encode,
                min_value=min_val,
                max_value=max_val
            )
        )

    # Add categorical QIs
    for column in categorical_columns:
        all_quasi_columns.append(column)
        quasi_identifiers.append(
            QuasiIdentifier(column, is_categorical=True, is_encoded=False)
        )

    if not quasi_identifiers:
        raise ValueError("No quasi-identifiers defined.")

    print("Selected quasi-identifiers:", all_quasi_columns)

    # Total records
    total_records = 0
    if chunk_files:
        first_chunk = pd.read_csv(os.path.join(chunk_dir, chunk_files[0]))
        total_records = len(first_chunk) * n

    # === BUILD INITIAL OLA_1 TREE ===
    print("\nBuilding initial tree and finding Ri values...")
    ola_1 = OLA_1(quasi_identifiers, n, max_equivalence_classes, multiplication_factors)
    ola_1.build_tree()
    ola_1.find_smallest_passing_ri(n)
    initial_ri = ola_1.get_optimal_ri()
    print("Initial bin widths (Ri):", initial_ri)
    logger.info(f"Initial bin widths (Ri): {initial_ri}")
    log_performance(logger,"OLA_1 tree", start_time)

    # === BUILD OLA_2 TREE ===
    ola_2 = OLA_2(quasi_identifiers, total_records, suppression_limit, multiplication_factors, sensitive_parameter)
    print("\nBuilding second tree with initial Ri values as root...")
    ola_2.build_tree(initial_ri)

    # === PROCESS CHUNKS FOR HISTOGRAMS ===
    print("\nProcessing data in chunks for histograms...")
    histograms = []

    for i, filename in enumerate(chunk_files):
        chunk = pd.read_csv(os.path.join(chunk_dir, filename))
        working_chunk = chunk.copy()

        if suppressed_columns:
            working_chunk = suppress(working_chunk, suppressed_columns)
        if pseudonymized_columns:
            working_chunk = pseudonymize(working_chunk, pseudonymized_columns)

        # Add encoded columns
        for info in numerical_columns_info:
            column = info["column"]
            encode = info.get("encode", False)
            if encode:
                enc_map = encoding_maps[column]["encoding_map"]
                if info.get("type") == "float":
                    multiplier = encoding_maps[column]["multiplier"]
                    working_chunk[f"{column}_encoded"] = (working_chunk[column] * multiplier).round().astype(int).map(enc_map)
                else:
                    working_chunk[f"{column}_encoded"] = working_chunk[column].map(enc_map)

        chunk_histogram = ola_2.process_chunk(working_chunk, initial_ri)
        histograms.append(chunk_histogram)
        print(f"Processed chunk {i+1}/{n} for histograms.")

    print("Histograms collected.")

    # === MERGE HISTOGRAMS AND FIND FINAL BIN WIDTHS ===
    print("\nMerging histograms and finding final bin widths...")
    global_histogram = ola_2.merge_histograms(histograms)
    final_rf = ola_2.get_final_binwidths(global_histogram, k, l)
    supp_percent = ola_2.get_suppressed_percent(final_rf, global_histogram, k)
    lowest_dm_star = ola_2.lowest_dm_star
    num_eq_classes = ola_2.best_num_eq_classes
    eq_class_stats = ola_2.get_equivalence_class_stats(global_histogram, final_rf, k)
    logger.info(f"Final bin widths (RF): {final_rf}")
    logger.info(f"Lowest DM*: {lowest_dm_star}, EQ Classes: {num_eq_classes}, Supp%: {supp_percent:.2f}")
    log_performance(logger,"OLA_2 tree", start_time)
    print("Lowest DM*:", lowest_dm_star)
    print("Number of equivalence classes:", num_eq_classes)
    print("EQ Class Stats:", eq_class_stats)
    print("Final bin widths (RF):", final_rf)
    print("Suppressed percentage of records:", supp_percent)
    #log_to_file(f"Final bin widths (RF): {final_rf}", log_file)

    # === GENERALIZE AND SAVE FIRST CHUNK ===
    if save_output and chunk_files:
        print("\nGeneralizing first chunk based on RF...")
        first_chunk = pd.read_csv(os.path.join(chunk_dir, chunk_files[0]))
        working_chunk = first_chunk.copy()

        if suppressed_columns:
            working_chunk = suppress(working_chunk, suppressed_columns)
        if pseudonymized_columns:
            working_chunk = pseudonymize(working_chunk, pseudonymized_columns)

        # Add encoded columns
        for info in numerical_columns_info:
            column = info["column"]
            encode = info.get("encode", False)
            if encode:
                enc_map = encoding_maps[column]["encoding_map"]
                if info.get("type") == "float":
                    multiplier = encoding_maps[column]["multiplier"]
                    working_chunk[f"{column}_encoded"] = (working_chunk[column] * multiplier).round().astype(int).map(enc_map)
                else:
                    working_chunk[f"{column}_encoded"] = working_chunk[column].map(enc_map)

        generalized_chunk = ola_2.generalize_chunk(working_chunk, final_rf)

        # Remove temporary encoded columns
        for info in numerical_columns_info:
            if info.get("encode", False):
                col_encoded = f"{info['column']}_encoded"
                if col_encoded in generalized_chunk.columns:
                    generalized_chunk.drop(columns=[col_encoded], inplace=True)

        generalized_chunk.to_csv(output_path, index=False)
        print(f"Generalized first chunk saved to: {output_path}")

    elapsed_time = time.time() - start_time
    h, m, s = format_time(elapsed_time)
    print(f"\nTotal time taken: {h}h {m}m {s}s")
    log_performance(logger,"SKALD full pipeline", start_time)
    with open(log_file, "a") as f:
        f.write("=== SKALD Completed ===\n\n")
    #log_to_file(f"Chunks: {n}, k: {k}", log_file)
    #og_to_file(f"Total time taken: {h}h {m}m {s}s", log_file)

    return final_rf, elapsed_time, lowest_dm_star, num_eq_classes, eq_class_stats
