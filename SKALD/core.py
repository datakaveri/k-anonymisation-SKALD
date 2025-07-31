import os
import time
import yaml
import pandas as pd
import numpy as np
import json

# Import necessary components from chunkanon
from SKALD.quasi_identifier import QuasiIdentifier
from SKALD.generalization_ri import OLA_1  # Initial generalization tree builder
from SKALD.generalization_rf import OLA_2  # Final generalization and histogram merger
from SKALD.utils import get_progress_iter, log_to_file, format_time, ensure_folder
from SKALD.config_validation import load_config

def run_pipeline(config_path="config.yaml", k=None, chunks=None, chunk_dir=None):
    config = load_config(config_path)

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
    sensitive_paramter = config.sensitive_parameter
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

    def find_max_decimal_places(series):
        decimals = series.dropna().map(lambda x: len(str(x).split(".")[1]) if "." in str(x) else 0)
        return decimals.max()

    if numerical_columns_info:
        for info in numerical_columns_info:
            column = info.get("column")
            column_type = info.get("type", "float")

            if not column:
                raise ValueError("Missing 'column' key in numerical_columns_info.")

            all_unique_values = set()
            multiplier = 1
            encoded_column = f"{column}_encoded"

            for filename in chunk_files:
                chunk = pd.read_csv(os.path.join(chunk_dir, filename))
                values = chunk[column].dropna()

                if column_type == "float":
                    decimal_places = find_max_decimal_places(values)
                    multiplier = 10 ** decimal_places
                    chunk[encoded_column] = (values * multiplier).round().astype(int)
                else:
                    chunk[encoded_column] = chunk[column].astype(int)

                all_unique_values.update(chunk[encoded_column].unique())
                chunk.to_csv(os.path.join(chunk_dir, filename), index=False)

            sorted_values = sorted(all_unique_values)
            encoding_map = {val: idx+1 for idx, val in enumerate(sorted_values)}
            encoding_map = {int(k): v for k, v in encoding_map.items()}
            decoding_map = {idx+1 : val for idx, val in enumerate(sorted_values)}

            encoding_maps[column] = {
                "decoding_map": {
                    int(k): int(v) if isinstance(v, (int, np.integer)) else float(v) if isinstance(v, (float, np.floating)) else v
                    for k, v in decoding_map.items()
                },
                "multiplier": int(multiplier) if isinstance(multiplier, (np.integer, np.int64)) else float(multiplier),
                "type": column_type
            }

            try:
                with open(os.path.join(encoding_dir, f"{column.replace(' ','_').lower()}_encoding.json"), "w") as f:
                    json.dump(encoding_maps[column], f, indent=4)
            except Exception as e:
                print(f"Error saving encoding map for {column}: {e}")

            for i in get_progress_iter(range(n), desc=f"Encoding {column}"):
                chunk_path = os.path.join(chunk_dir, chunk_files[i])
                try:
                    chunk = pd.read_csv(chunk_path)
                    chunk[encoded_column] = (chunk[column]*multiplier).map(encoding_map)
                    chunk.to_csv(chunk_path, index=False)
                except Exception as e:
                    print(f"Error encoding column {column} in {chunk_path}: {e}")
    else:
        print("No numerical columns to encode.")

    quasi_identifiers = []
    all_quasi_columns = []

    if numerical_columns_info:
        for info in numerical_columns_info:
            column = info.get("column")
            if not column:
                continue

            encode = info.get("encode", False)
            encoded_column = f"{column}_encoded" if encode else column

            if encode:
                decoding_map = encoding_maps.get(column, {}).get("decoding_map", {})
                if decoding_map:
                    min_val, max_val = 1, len(decoding_map)
                else:
                    print(f"Warning: No decoding map found for encoded column '{column}'. Skipping.")
                    continue
            else:
                min_max = hardcoded_min_max.get(column)
                if not min_max or len(min_max) != 2:
                    print(f"Warning: Missing or invalid hardcoded min/max for column '{column}'. Skipping.")
                    continue
                min_val, max_val = min_max

            all_quasi_columns.append(encoded_column)
            quasi_identifiers.append(
                QuasiIdentifier(
                    encoded_column,
                    is_categorical=False,
                    is_encoded=encode,
                    min_value=min_val,
                    max_value=max_val
                )
            )
    else:
        print("Warning: No numerical quasi-identifiers found.")

    if categorical_columns:
        for column in categorical_columns:
            if not column:
                continue
            all_quasi_columns.append(column)
            quasi_identifiers.append(
                QuasiIdentifier(column, is_categorical=True, is_encoded=False)
            )
    else:
        print("Warning: No categorical quasi-identifiers found.")

    if not quasi_identifiers:
        raise ValueError("No quasi-identifiers defined. Please specify at least one categorical or numerical QID.")

    print("Selected quasi-identifiers:", all_quasi_columns)

    total_records = 0
    if chunk_files:
        try:
            first_chunk = pd.read_csv(os.path.join(chunk_dir, chunk_files[0]))
            records_per_chunk = len(first_chunk)
            total_records = records_per_chunk * n
        except Exception as e:
            print(f"Error reading first chunk: {e}")

    print("\nBuilding initial tree and finding Ri values...")
    ola_1 = OLA_1(quasi_identifiers, n, max_equivalence_classes, multiplication_factors)
    ola_1.build_tree()
    initial_ri = ola_1.find_smallest_passing_ri(n)
    initial_ri = ola_1.get_optimal_ri()
    print("Initial bin widths (Ri):", initial_ri)
    log_to_file(f"Initial bin widths (Ri): {initial_ri}", log_file)

    ola_2 = OLA_2(quasi_identifiers, total_records, suppression_limit, multiplication_factors, sensitive_paramter)
    print("\nBuilding second tree with initial Ri values as root...")
    ola_2.build_tree(initial_ri)

    print("\nProcessing data in chunks for histograms...")
    histograms = []
    for i in range(n):
        try:
            chunk = pd.read_csv(os.path.join(chunk_dir, chunk_files[i]))
            chunk_histogram = ola_2.process_chunk(chunk, initial_ri)
            histograms.append(chunk_histogram)
            print(f"Processed chunk {i+1}/{n} for histograms.")
        except Exception as e:
            print(f"Error processing chunk {chunk_files[i]}: {e}")

    print("Histograms collected.")

    print("\nMerging histograms and finding final bin widths...")
    global_histogram = ola_2.merge_histograms(histograms)
    final_rf = ola_2.get_final_binwidths(global_histogram, k, l)
    supp_percent = ola_2.get_suppressed_percent(final_rf, global_histogram, k)
    print("Final bin widths (RF):", final_rf)
    print("suppressed percentage of records :", supp_percent)
    log_to_file(f"Final bin widths (RF): {final_rf}", log_file)

    if save_output and chunk_files:
        print("\nGeneralizing first chunk based on RF...")
        #print(type(final_rf))
        first_chunk = pd.read_csv(os.path.join(chunk_dir, chunk_files[0]))
        generalized_chunk = ola_2.generalize_chunk(first_chunk, final_rf)
        generalized_chunk.to_csv(output_path, index=False)
        print(f"Generalized first chunk saved to: {output_path}")

    for chunks in chunk_files:
        chunk_path = os.path.join(chunk_dir, chunks)
        chunk = pd.read_csv(chunk_path)

        for info in numerical_columns_info:
            column = info.get("column")
            if info.get("encode", True):
                encoded_column = f"{column}_encoded"
                if encoded_column in chunk.columns:
                    chunk = chunk.drop(columns=[encoded_column], errors='ignore')

        chunk.to_csv(chunk_path, index=False)

    elapsed_time = time.time() - start_time
    h, m, s = format_time(elapsed_time)
    print(f"\nTotal time taken: {h}h {m}m {s}s")
    log_to_file(f"Chunks: {n}, k: {k}", log_file)
    log_to_file(f"Total time taken: {h}h {m}m {s}s", log_file)

    return final_rf, elapsed_time
