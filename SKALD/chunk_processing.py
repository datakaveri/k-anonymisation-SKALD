import os
import pandas as pd
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
            chunk = pd.read_csv(file_path, low_memory=False)
        except Exception:
            continue

        if chunk.empty:
            continue

        working_chunk = chunk.copy()

        # -----------------------------
        # Apply numerical encoding (STRICT)
        # -----------------------------
        for info in numerical_columns_info:
            column = info.get("column")
            encode = info.get("encode", False)
            col_type = info.get("type")

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
                multiplier = enc_info.get("multiplier", 1)

                if not isinstance(enc_map, dict) or not enc_map:
                    raise ValueError(
                        f"Invalid encoding_map for column '{column}'"
                    )

                try:
                    numeric = pd.to_numeric(working_chunk[column], errors="coerce")
                    invalid_count = int(numeric.isna().sum() - working_chunk[column].isna().sum())
                    if invalid_count > 0:
                        logger.warning(
                            "Column '%s' has %d non-numeric value(s) in chunk '%s' that will be ignored for histogram encoding.",
                            column,
                            invalid_count,
                            filename,
                        )

                    if col_type == "float":
                        values = (numeric * multiplier).round()
                        logger.info("Applied float encoding for column '%s' with multiplier %s", column, multiplier)
                    else:
                        values = numeric.round()
                        logger.info("Applied integer encoding for column '%s'", column)
                    values = values.astype("Int64")
                except Exception as e:
                    raise ValueError(
                        f"Failed converting values for encoding column '{column}': {e}"
                    )

                encoded = values.map(enc_map)
                missing_encoded = int(encoded.isna().sum())
                if missing_encoded > 0:
                    logger.warning(
                        "Column '%s' produced %d unmapped encoded value(s) in chunk '%s'; affected rows will be skipped in histogram processing.",
                        column,
                        missing_encoded,
                        filename,
                    )

                working_chunk[f"{column}_encoded"] = encoded

        # -----------------------------
        # Build histogram (STRICT)
        # -----------------------------
        try:
            histogram = ola_2.process_chunk(working_chunk, initial_ri)
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
