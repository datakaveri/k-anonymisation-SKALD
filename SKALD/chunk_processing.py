import os
import pandas as pd
from SKALD.preprocess import suppress, pseudonymize

def process_chunks_for_histograms(
    chunk_files,
    chunk_dir,
    numerical_columns_info,
    encoding_maps,
    ola_2,
    initial_ri
):
    """
    Process each chunk safely:
    - Reads chunk
    - Applies encoding if needed
    - Builds histogram using OLA_2
    Returns list of histograms.

    Errors are caught per file, logged, and skipped
    instead of crashing the full pipeline.
    """

    histograms = []

    if not isinstance(chunk_files, list):
        raise TypeError("chunk_files must be a list of filenames.")

    if not os.path.isdir(chunk_dir):
        raise FileNotFoundError(f"Chunk directory does not exist: {chunk_dir}")

    for i, filename in enumerate(chunk_files):
        file_path = os.path.join(chunk_dir, filename)

        # --- Robust file validation ---
        if not os.path.exists(file_path):
            print(f"[WARN] Chunk file missing: {file_path}. Skipping.")
            continue

        try:
            if os.path.getsize(file_path) == 0:
                print(f"[WARN] Chunk file is empty: {file_path}. Skipping.")
                continue
        except Exception as e:
            print(f"[ERROR] Could not check size of {file_path}: {e}")
            continue

        # --- Read CSV safely ---
        try:
            chunk = pd.read_csv(file_path)
        except Exception as e:
            print(f"[ERROR] Failed to read CSV {file_path}: {e}. Skipping.")
            continue

        if chunk.empty:
            print(f"[WARN] Chunk {file_path} is empty after read. Skipping.")
            continue

        working_chunk = chunk.copy()

        # --- Apply numeric encoding ---
        for info in numerical_columns_info:
            column = info.get("column")
            encode = info.get("encode", False)
            col_type = info.get("type", "float")

            if encode:
                if column not in chunk.columns:
                    print(f"[WARN] Missing column '{column}' in chunk {filename}. Skipping encoding for this column.")
                    continue

                if column not in encoding_maps:
                    print(f"[ERROR] Encoding map missing for column '{column}'. Skipping this encoding.")
                    continue

                enc_map = encoding_maps[column].get("encoding_map", {})
                multiplier = encoding_maps[column].get("multiplier", 1)

                try:
                    if col_type == "float":
                        values = (working_chunk[column] * multiplier).round().astype(int)
                    else:
                        values = working_chunk[column].astype(int)

                    encoded = values.map(enc_map)

                    if encoded.isna().any():
                        print(f"[WARN] Some values in '{column}' not found in encoding map. They will become NaN.")

                    working_chunk[f"{column}_encoded"] = encoded

                except Exception as e:
                    print(f"[ERROR] Failed to encode column '{column}' in {filename}: {e}")
                    continue

        # --- Build histogram using OLA_2 ---
        try:
            chunk_histogram = ola_2.process_chunk(working_chunk, initial_ri)
            histograms.append(chunk_histogram)
        except Exception as e:
            print(f"[ERROR] OLA_2 failed to process chunk {filename}: {e}")
            continue

        print(f"Processed chunk {i+1}/{len(chunk_files)} for histograms.")

    if not histograms:
        raise ValueError("No valid histograms produced from chunks. Check input data.")

    return histograms
