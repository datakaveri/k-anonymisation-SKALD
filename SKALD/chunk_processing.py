import os
import pandas as pd
import numpy as np
from typing import List
import logging
logger = logging.getLogger("SKALD")


def process_chunks_for_histograms(
    chunk_files: List[str],
    chunk_dir: str,
    numerical_columns_info: list,
    encoding_maps: dict,
    ola_2,
    initial_ri: list,
):
    """
    Process chunks and build histograms using OLA_2.

    Rules:
    - I/O errors → skip chunk
    - Configuration / encoding errors → fail fast
    - At least one histogram must be produced

    Raises:
        TypeError, FileNotFoundError, ValueError, RuntimeError
    """

    if not isinstance(chunk_files, list) or not chunk_files:
        raise ValueError("chunk_files must be a non-empty list")

    if not os.path.isdir(chunk_dir):
        raise FileNotFoundError(f"Chunk directory does not exist: {chunk_dir}")

    if not isinstance(initial_ri, list) or not all(isinstance(v, int) and v > 0 for v in initial_ri):
        raise ValueError("initial_ri must be a list of positive integers")

    histograms = []

    for filename in chunk_files:
        file_path = os.path.join(chunk_dir, filename)

        # -----------------------------
        # I/O validation (skippable)
        # -----------------------------
        if not os.path.isfile(file_path):
            continue

        try:
            if os.path.getsize(file_path) == 0:
                continue
        except OSError:
            continue

        try:
            chunk = pd.read_csv(file_path)
        except Exception:
            continue

        if chunk.empty:
            continue

        working_chunk = chunk.copy()

        # -----------------------------
        # Apply numerical scaling and encoding 
        # -----------------------------
        for info in numerical_columns_info:
            column = info.get("column")
            scale = info.get("scale", False)
            s = int(info.get("s", 0)) if scale else 0
            encode = info.get("encode", False)
            col_type = info.get("type")

            # -----------------------------
            # Scaling
            # -----------------------------
            if scale:
                if col_type == "int":
                    if s < 0:
                        raise ValueError("s must be a non-negative integer for integer scaling")
                    else:
                        scaled = np.floor(working_chunk[column] / (10 ** s)).astype(int)
                        if encode:
                            enc_info = encoding_maps[column]
                            enc_map = enc_info.get("encoding_map")
                            values = scaled.astype(int)
                            encoded = values.map(enc_map)
                            working_chunk[f"{column}_scaled_encoded"] = encoded
                        else:
                            values = scaled.astype(int)
                            working_chunk[f"{column}_scaled"] = values

                else:
                    if s <= 0:
                        if not encode:
                            raise ValueError(
                                f"Scaling with non-positive s not allowed for float column '{column}' without encoding"
                            )
                        else:
                            scaled = np.floor(working_chunk[column] / (10 ** s))
                            enc_info = encoding_maps[column]
                            enc_map = enc_info.get("encoding_map")
                            values = scaled.astype(int)
                            encoded = values.map(enc_map)
                            working_chunk[f"{column}_scaled_encoded"] = encoded 
                    else:
                        scaled = np.floor(working_chunk[column] / (10 ** s))
                        if encode:
                            enc_info = encoding_maps[column]
                            enc_map = enc_info.get("encoding_map")
                            values = scaled.astype(int)
                            encoded = values.map(enc_map)
                            working_chunk[f"{column}_scaled_encoded"] = encoded
                        else:
                            values = scaled.astype(int)
                            working_chunk[f"{column}_scaled"] = values

            else:
                if encode:
                    if column not in working_chunk.columns:
                        raise KeyError(
                            f"Column '{column}' missing in chunk '{filename}' during encoding"
                        )

                    if column not in encoding_maps:
                        raise KeyError(
                            f"Encoding map missing for encoded column '{column}'"
                        )

                    enc_info = encoding_maps[column]
                    enc_map = enc_info.get("encoding_map")

                    if not isinstance(enc_map, dict) or not enc_map:
                        raise ValueError(
                            f"Invalid encoding_map for column '{column}'"
                        )

                    try:
                        values = working_chunk[column].astype(int)
                        encoded = values.map(enc_map)
                        logger.info("Applied integer encoding for column '%s'", column)
                    except Exception as e:
                        raise ValueError(
                            f"Failed converting values for encoding column '{column}': {e}"
                        )

                    if encoded.isna().any():
                        raise ValueError(
                            f"Found values in '{column}' not present in encoding_map"
                        )

                    working_chunk[f"{column}_encoded"] = encoded

        # Persist scaled / encoded columns for later generalization
        working_chunk.to_csv(file_path, index=False)
        # -----------------------------
        # Build histogram (STRICT)
        # -----------------------------

    for filename in chunk_files:
        file_path = os.path.join(chunk_dir, filename)
        chunk = pd.read_csv(file_path)
        try:
            histogram = ola_2.process_chunk(chunk, initial_ri)
        except Exception as e:
            raise RuntimeError(
                f"Failed to build histogram for chunk '{filename}': {e}"
            )

        histograms.append(histogram)

    if not histograms:
        raise ValueError(
            "No valid histograms produced. All chunks were empty or unreadable."
        )

    return histograms
