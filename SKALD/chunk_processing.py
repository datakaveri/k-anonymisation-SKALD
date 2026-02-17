import os
import pandas as pd
import numpy as np
from typing import List
import logging

logger = logging.getLogger("SKALD")


def _apply_numerical_processing(
    working_chunk: pd.DataFrame,
    numerical_columns_info: list,
    encoding_maps: dict,
    filename: str,
) -> pd.DataFrame:
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

    return working_chunk


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

    global_hist = None
    batch_rows_env = os.getenv("SKALD_CHUNK_PROCESSING_ROWS", "200000")
    try:
        batch_rows = max(1000, int(batch_rows_env))
    except Exception:
        raise ValueError("SKALD_CHUNK_PROCESSING_ROWS must be an integer")

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

        while True:
            temp_path = file_path + ".tmp"
            wrote_any = False
            local_hist = None
            try:
                reader = pd.read_csv(file_path, chunksize=batch_rows, low_memory=False)
            except Exception:
                break

            try:
                for batch in reader:
                    if batch.empty:
                        continue

                    working_chunk = batch.copy()
                    working_chunk = _apply_numerical_processing(
                        working_chunk,
                        numerical_columns_info,
                        encoding_maps,
                        filename,
                    )

                    # Persist scaled / encoded columns for later generalization
                    mode = "w" if not wrote_any else "a"
                    header = not wrote_any
                    working_chunk.to_csv(temp_path, mode=mode, header=header, index=False)
                    wrote_any = True

                    # -----------------------------
                    # Build histogram (STRICT)
                    # -----------------------------
                    try:
                        batch_hist = ola_2.process_chunk(working_chunk, initial_ri)
                    except Exception as e:
                        raise RuntimeError(
                            f"Failed to build histogram for chunk '{filename}': {e}"
                        )

                    if local_hist is None:
                        local_hist = batch_hist
                    else:
                        if isinstance(local_hist, dict) and isinstance(batch_hist, dict):
                            for k, v in batch_hist.items():
                                local_hist[k] = local_hist.get(k, 0) + v
                        else:
                            local_hist = local_hist + batch_hist
            except MemoryError:
                # Reduce batch size and retry to be fail-safe on small machines
                batch_rows = max(1000, batch_rows // 2)
                if os.path.exists(temp_path):
                    os.remove(temp_path)
                if batch_rows <= 1000:
                    raise RuntimeError(
                        "Out of memory while processing chunks even at minimum batch size"
                    )
                logger.warning(
                    "MemoryError while processing '%s'. Retrying with batch_rows=%d",
                    filename,
                    batch_rows,
                )
                continue

            if wrote_any:
                os.replace(temp_path, file_path)
            else:
                if os.path.exists(temp_path):
                    os.remove(temp_path)

            if local_hist is not None:
                if global_hist is None:
                    global_hist = local_hist
                else:
                    if isinstance(global_hist, dict) and isinstance(local_hist, dict):
                        for k, v in local_hist.items():
                            global_hist[k] = global_hist.get(k, 0) + v
                    else:
                        global_hist = global_hist + local_hist
            break

    if global_hist is None:
        raise ValueError(
            "No valid histograms produced. All chunks were empty or unreadable."
        )

    return global_hist
