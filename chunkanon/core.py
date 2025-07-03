import os
import time
import yaml
import pandas as pd
import numpy as np
import json

# Import necessary components from chunkanon
from chunkanon.quasi_identifier import QuasiIdentifier
from chunkanon.generalization_ri import OLA_1  # Initial generalization tree builder
from chunkanon.generalization_rf import OLA_2  # Final generalization and histogram merger
from chunkanon.utils import get_progress_iter, log_to_file, format_time, ensure_folder
from chunkanon.config_validation import load_config


def run_pipeline(config_path="config.yaml", k=None, chunks=None, chunk_dir=None):
    """
    Executes the chunk-based k-anonymization pipeline based on the provided configuration.

    Pipeline Steps:
    1. Load configuration and prepare environment.
    2. Encode numerical quasi-identifiers (if required).
    3. Define quasi-identifiers for tree-building.
    4. Build RI tree and derive initial bin widths.
    5. Build RF tree and calculate final bin widths.
    6. Apply generalization to the first chunk and save.
    7. Clean up temporary encoded columns from all chunks.

    Args:
        config_path (str): Path to the configuration YAML file.
        k (int, optional): Override for k-anonymity value.
        chunks (int, optional): Override for number of chunks.
        chunk_dir (str, optional): Override for chunk directory.

    Returns:
        tuple: (final_rf, elapsed_time)
            - final_rf: Final bin widths per quasi-identifier.
            - elapsed_time: Total time taken for the pipeline.
    """
    # Load configuration object
    config = load_config(config_path)

    # Override config values if provided via CLI
    if k is not None:
        config.k = k
    if chunks is not None:
        config.number_of_chunks = chunks
    if chunk_dir is not None:
        config.chunk_directory = chunk_dir

    # Use the updated values from config
    n = config.number_of_chunks
    k = config.k
    chunk_dir = config.chunk_directory
    output_path = config.output_path
    suppression_limit = config.suppression_limit
    max_equivalence_classes = config.max_number_of_eq_classes
    suppression_limit = config.suppression_limit
    output_path = config.output_path
    log_file = config.log_file
    save_output = config.save_output

    # Extract categorical and numerical QI info
    categorical_columns = [col.column for col in config.quasi_identifiers.categorical]
    numerical_columns_info = [
        {"column": col.column, "encode": col.encode, "type": col.type}
        for col in config.quasi_identifiers.numerical
    ]

    # Hardcoded min/max for numerical QIs and multiplication factors
    hardcoded_min_max = config.hardcoded_min_max
    multiplication_factors = config.bin_width_multiplication_factor

    # Ensure chunk directory exists
    if not os.path.exists(chunk_dir):
        raise FileNotFoundError(f"Chunk directory not found: {chunk_dir}")

    # Find all CSV files in chunk directory
    all_files = sorted([
        f for f in os.listdir(chunk_dir)
        if f.endswith(".csv")
    ])

    if not all_files:
        raise ValueError("No CSV files found in the chunk directory.")

    # Take only the first `n` files
    chunk_files = all_files[:n]

    print(f"Processing {n} chunks from {chunk_dir}...")

    encoding_maps = {}
    encoding_dir = "encodings"
    ensure_folder(encoding_dir)

    start_time = time.time()

    def find_max_decimal_places(series):
        """Helper function to find maximum number of decimal places in a float column."""
        decimals = series.dropna().map(lambda x: len(str(x).split(".")[1]) if "." in str(x) else 0)
        return decimals.max()

    # ------------------------------
    # Step 1: Encode numerical columns
    # ------------------------------
    for info in numerical_columns_info:
        if info.get("encode", False):
            column = info.get("column")
            column_type = info.get("type", "float")  # Default type is float

            if not column:
                raise ValueError("Missing 'column' key in numerical_columns_info.")

            all_unique_values = set()
            multiplier = 1  # Used to eliminate decimals for float values

            encoded_column = f"{column}_encoded"

            # Encode values in each chunk
            for filename in chunk_files:
                chunk = pd.read_csv(os.path.join(chunk_dir, filename))
                values = chunk[column].dropna()

                if column_type == "float":
                    # Determine precision to preserve decimals
                    decimal_places = find_max_decimal_places(values)
                    multiplier = 10 ** decimal_places
                    chunk[encoded_column] = (values * multiplier).round().astype(int)
                else:
                    chunk[encoded_column] = chunk[column].astype(int)

                all_unique_values.update(chunk[encoded_column].unique())
                chunk.to_csv(os.path.join(chunk_dir, filename), index=False)


            # Create encoding and decoding maps
            sorted_values = sorted(all_unique_values)
            encoding_map = {val: idx+1 for idx, val in enumerate(sorted_values)}
            encoding_map = {int(k): v for k, v in encoding_map.items()}
            decoding_map = {idx+1 : val for idx, val in enumerate(sorted_values)}

            encoding_maps[column] = {
                "decoding_map": {
                    int(k): int(v) if isinstance(v, (int, np.integer))
                    else float(v) if isinstance(v, (float, np.floating)) else v
                    for k, v in decoding_map.items()
                },
                "multiplier": int(multiplier) if isinstance(multiplier, (np.integer, np.int64)) else float(multiplier),
                "type": column_type
            }

            # Save encoding map to file
            try:
                with open(os.path.join(encoding_dir, f"{column.replace(' ','_').lower()}_encoding.json"), "w") as f:
                    json.dump(encoding_maps[column], f, indent=4)
            except Exception as e:
                print(f"Error saving encoding map for {column}: {e}")

            # Apply encoding map to each chunk
            for i in get_progress_iter(range(n), desc=f"Encoding {column}"):
                chunk_path = os.path.join(chunk_dir, chunk_files[i])
                try:
                    chunk = pd.read_csv(chunk_path)
                    chunk[encoded_column] = (chunk[column]*multiplier).map(encoding_map)
                    chunk.to_csv(chunk_path, index=False)
                except Exception as e:
                    print(f"Error encoding column {column} in {chunk_path}: {e}")

    # ------------------------------
    # Step 2: Define Quasi-Identifiers
    # ------------------------------
    quasi_identifiers = []
    all_quasi_columns = []

    # Handle numerical QIs
    if numerical_columns_info:
        for info in numerical_columns_info:
            column = info.get("column")
            if not column:
                continue

            encode = info.get("encode", False)
            encoded_column = f"{column}_encoded" if encode else column

            if encode:
                # Use encoded range
                min_val, max_val = 1, len(encoding_maps.get(column, {}).get("decoding_map", {}))
            else:
                # Use hardcoded range
                min_val, max_val = hardcoded_min_max.get(column, (None, None))

            if min_val is None or max_val is None:
                print(f"Warning: Missing hardcoded min/max for numerical column '{column}'. Skipping.")
                continue

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

    # Handle categorical QIs
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

    # ------------------------------
    # Step 3: Build RI Tree
    # ------------------------------
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
    initial_ri = ola_1.find_smallest_passing_ri(n)  # May be unused depending on implementation
    initial_ri = ola_1.get_optimal_ri()
    print("Initial bin widths (Ri):", initial_ri)
    log_to_file(f"Initial bin widths (Ri): {initial_ri}", log_file)

    # ------------------------------
    # Step 4: Build RF Tree and Histograms
    # ------------------------------
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
    supp_percent = ola_2.get_suppressed_percent(final_rf,global_histogram,k)
    print("Final bin widths (RF):", final_rf)
    print("suppressed percentage of records :", supp_percent)

    log_to_file(f"Final bin widths (RF): {final_rf}", log_file)

    # ------------------------------
    # Step 5: Apply Generalization and Save
    # ------------------------------
    if save_output and chunk_files:
        print("\nGeneralizing first chunk based on RF...")
        first_chunk = pd.read_csv(os.path.join(chunk_dir, chunk_files[0]))
        generalized_chunk = ola_2.generalize_chunk(first_chunk, final_rf)
        generalized_chunk.to_csv(output_path, index=False)
        print(f"Generalized first chunk saved to: {output_path}")

    # ------------------------------
    # Step 6: Clean up encoded columns
    # ------------------------------
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

    # ------------------------------
    # Step 7: Log total runtime
    # ------------------------------
    elapsed_time = time.time() - start_time
    h, m, s = format_time(elapsed_time)
    print(f"\nTotal time taken: {h}h {m}m {s}s")
    log_to_file(f"Chunks: {n}, k: {k}", log_file)
    log_to_file(f"Total time taken: {h}h {m}m {s}s", log_file)

    return final_rf, elapsed_time
