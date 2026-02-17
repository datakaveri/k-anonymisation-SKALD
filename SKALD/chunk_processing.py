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
            if column not in working_chunk.columns:
                raise KeyError(
                    f"Column '{column}' missing in chunk '{filename}' during numerical processing"
                )

            raw_vals = working_chunk[column]
            if raw_vals.dtype == object:
                raw_vals = raw_vals.replace(r"^\s*$", np.nan, regex=True)

            numeric_vals = pd.to_numeric(raw_vals, errors="coerce")
            non_numeric = raw_vals.notna().sum() - numeric_vals.notna().sum()
            if non_numeric > 0:
                logger.warning(
                    "Column '%s' has %d non-numeric value(s) in chunk '%s' that will be ignored for numerical processing.",
                    column,
                    non_numeric,
                    filename,
                )
            if numeric_vals.notna().sum() == 0:
                raise ValueError(
                    f"Column '{column}' has no valid numeric values in chunk '{filename}'"
                )

            # -----------------------------
            # Scaling
            # -----------------------------
            if scale:
                if col_type == "int":
                    if s < 0:
                        raise ValueError("s must be a non-negative integer for integer scaling")
                    else:
                        scaled = pd.Series(
                            np.floor(numeric_vals / (10 ** s)),
                            index=numeric_vals.index,
                        ).astype("Int64")
                        if encode:
                            enc_info = encoding_maps[column]
                            enc_map = enc_info.get("encoding_map")
                            values = scaled
                            encoded = values.map(enc_map)
                            working_chunk[f"{column}_scaled_encoded"] = encoded
                        else:
                            working_chunk[f"{column}_scaled"] = scaled

                else:
                    if s <= 0:
                        if not encode:
                            raise ValueError(
                                f"Scaling with non-positive s not allowed for float column '{column}' without encoding"
                            )
                        else:
                            scaled = pd.Series(
                                np.floor(numeric_vals / (10 ** s)),
                                index=numeric_vals.index,
                            ).astype("Int64")
                            enc_info = encoding_maps[column]
                            enc_map = enc_info.get("encoding_map")
                            values = scaled
                            encoded = values.map(enc_map)
                            working_chunk[f"{column}_scaled_encoded"] = encoded 
                    else:
                        scaled = pd.Series(
                            np.floor(numeric_vals / (10 ** s)),
                            index=numeric_vals.index,
                        ).astype("Int64")
                        if encode:
                            enc_info = encoding_maps[column]
                            enc_map = enc_info.get("encoding_map")
                            values = scaled
                            encoded = values.map(enc_map)
                            working_chunk[f"{column}_scaled_encoded"] = encoded
                        else:
                            working_chunk[f"{column}_scaled"] = scaled

            else:
                if encode:
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
                        values = numeric_vals.astype("Int64")
                        encoded = values.map(enc_map)
                        logger.info("Applied integer encoding for column '%s'", column)
                    except Exception as e:
                        raise ValueError(
                            f"Failed converting values for encoding column '{column}': {e}"
                        )

                    if (encoded.isna() & values.notna()).any():
                        raise ValueError(
                            f"Found values in '{column}' not present in encoding_map"
                        )

                    working_chunk[f"{column}_encoded"] = encoded
                else:
                    # Ensure unencoded numeric columns are numeric for downstream histogramming
                    working_chunk[column] = numeric_vals

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
