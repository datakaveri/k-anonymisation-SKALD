import os
import time
import yaml
import pandas as pd
import numpy as np
import json

from chunkanon.quasi_identifier import QuasiIdentifier
from chunkanon.generalization_ri import OLA_1
from chunkanon.generalization_rf import OLA_2
from chunkanon.utils import get_progress_iter, log_to_file, format_time, ensure_folder

def run_pipeline(config_path="config.yaml"):
    if not os.path.exists(config_path):
        raise FileNotFoundError(f"Config file not found at path: {config_path}")

    with open(config_path, "r") as f:
        config = yaml.safe_load(f)

    n = config.get("number_of_chunks", 1)
    k = config.get("k", 500)
    max_equivalence_classes = config.get("max_number_of_eq_classes", 15000000)
    suppression_limit = config.get("suppression_limit", 0.01)
    chunk_dir = config.get("chunk_directory", "datachunks")
    output_path = config.get("output_path", "generalized_chunk.csv")
    log_file = config.get("log_file", "log.txt")
    save_output = config.get("save_output", True)

    categorical_columns = config.get("quasi_identifiers", {}).get("categorical", [])
    numerical_columns_info = config.get("quasi_identifiers", {}).get("numerical", [])
    hardcoded_min_max = config.get("hardcoded_min_max", {})
    multiplication_factors = config.get("bin_width_multiplication_factor", {})

    if not os.path.exists(chunk_dir):
        raise FileNotFoundError(f"Chunk directory not found: {chunk_dir}")

    all_files = sorted([
        f for f in os.listdir(chunk_dir)
        if f.endswith(".csv")
    ])

    if not all_files:
        raise ValueError("No CSV files found in the chunk directory.")

    chunk_files = all_files[:n]

    print(f"Processing {n} chunks from {chunk_dir}...")

    encoding_maps = {}
    encoding_dir = "encodings"
    ensure_folder(encoding_dir)

    start_time = time.time()

    # Helper to find decimal resolution
    def find_max_decimal_places(series):
        decimals = series.dropna().map(lambda x: len(str(x).split(".")[1]) if "." in str(x) else 0)
        return decimals.max()

    # Step 1: Encode necessary numerical columns
    for info in numerical_columns_info:
        if info.get("encode", False):
            column = info.get("column")
            column_type = info.get("type", "float")  # default type float

            if not column:
                raise ValueError("Missing 'column' key in numerical_columns_info.")

            all_unique_values = set()
            multiplier = 1  # default multiplier

            encoded_column = f"{column}_encoded"  

            for filename in chunk_files:
                try:
                    chunk = pd.read_csv(os.path.join(chunk_dir, filename))
                    values = chunk[column].dropna()

                    if column_type == "float":
                        decimal_places = find_max_decimal_places(values)
                        multiplier = 10 ** decimal_places
                        chunk[encoded_column] = (values * multiplier).round().astype(int)
                        
                    else:
                        chunk[encoded_column] = chunk[column].astype(int)

                    all_unique_values.update(chunk[encoded_column].unique())
                    chunk.to_csv(os.path.join(chunk_dir, filename), index=False)  # save modified chunk immediately
                    
                    
                except (FileNotFoundError, pd.errors.ParserError) as e:
                    print(f"Error reading chunk {filename}: {e}")
                    continue

            sorted_values = sorted(all_unique_values)
            encoding_map = {val: idx for idx, val in enumerate(sorted_values)}
            encoding_map = {int(k): v for k, v in encoding_map.items()}
            decoding_map = {idx: val for idx, val in enumerate(sorted_values)}

            encoding_maps[column] = {
                "decoding_map": {int(k): int(v) if isinstance(v, (int, np.integer)) else float(v) if isinstance(v, (float, np.floating)) else v for k, v in decoding_map.items()},
                "multiplier": int(multiplier) if isinstance(multiplier, (np.integer, np.int64)) else float(multiplier),
                "type": column_type
            }

            try:
                with open(os.path.join(encoding_dir, f"{column.replace(' ','_').lower()}_encoding.json"), "w") as f:
                    json.dump(encoding_maps[column], f, indent=4)
            except Exception as e:
                print(f"Error saving encoding map for {column}: {e}")

            # Encoding process on chunks
            for i in get_progress_iter(range(n), desc=f"Encoding {column}"):
                chunk_path = os.path.join(chunk_dir, chunk_files[i])
                try:
                    chunk = pd.read_csv(chunk_path)
                    chunk[encoded_column] = (chunk[column]*multiplier).map(encoding_map)
                    chunk.to_csv(chunk_path, index=False)
                except Exception as e:
                    print(f"Error encoding column {column} in {chunk_path}: {e}")

    # Step 2: Define quasi-identifiers (using the new encoded columns)
    quasi_identifiers = []
    all_quasi_columns = []
    
    if numerical_columns_info:
        for info in numerical_columns_info:
            column = info.get("column")
            if not column:
                continue

            encode = info.get("encode", False)
            if encode:
                encoded_column = f"{column}_encoded"
                min_val, max_val = 0, len(encoding_maps.get(column, {}).get("decoding_map", {}))
            else:
                min_val, max_val = hardcoded_min_max.get(column, (None, None))

            if min_val is None or max_val is None:
                print(f"Warning: Missing hardcoded min/max for numerical column '{column}'. Skipping.")
                continue

            all_quasi_columns.append(encoded_column if encode else column)
            quasi_identifiers.append(
                QuasiIdentifier(
                    encoded_column if encode else column, 
                    is_categorical=False,
                    is_encoded=True if encode else False,  
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

    ola_2 = OLA_2(quasi_identifiers, total_records, suppression_limit, multiplication_factors)
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
    final_rf = ola_2.get_final_binwidths(global_histogram, k)
    print("Final bin widths (RF):", final_rf)
    log_to_file(f"Final bin widths (RF): {final_rf}", log_file)

    if save_output and chunk_files:
        print("\nGeneralizing first chunk based on RF...")
        first_chunk = pd.read_csv(os.path.join(chunk_dir, chunk_files[0]))
        generalized_chunk = ola_2.generalize_chunk(first_chunk, final_rf)
        generalized_chunk.to_csv(output_path, index=False)
        print(f"Generalized first chunk saved to: {output_path}")

    
    for chunks in chunk_files:
        chunk_path = os.path.join(chunk_dir, chunks)
        chunk = pd.read_csv(chunk_path)

        for info in numerical_columns_info:
            if info.get("encode", True):
                if encoded_column in chunk.columns:
                    chunk = chunk.drop(columns=[encoded_column], errors='ignore')

        chunk.to_csv(chunk_path, index=False)


    elapsed_time = time.time() - start_time
    h, m, s = format_time(elapsed_time)
    print(f"\nTotal time taken: {h}h {m}m {s}s")
    log_to_file(f"Chunks: {n}, k: {k}", log_file)
    log_to_file(f"Total time taken: {h}h {m}m {s}s", log_file)

    return final_rf, elapsed_time
