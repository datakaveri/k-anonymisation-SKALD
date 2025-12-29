import os
import pandas as pd
import json
from typing import List, Dict, Tuple
from SKALD.utils import find_max_decimal_places
import logging
logger = logging.getLogger("SKALD")


# --------------------------------------------------
# Encoding directory
# --------------------------------------------------
def get_encoding_dir() -> str:
    try:
        d = os.getenv("SKALD_ENCODING_DIR", "encodings")
        os.makedirs(d, exist_ok=True)
        return d
    except Exception as e:
        raise OSError(f"Failed to create or access encoding directory: {e}")


# --------------------------------------------------
# Numerical encoding
# --------------------------------------------------
def encode_numerical_columns(
    chunk_files: List[str],
    chunk_dir: str,
    numerical_columns_info: List[Dict]
) -> Tuple[Dict, Dict]:
    """
    Encodes numerical columns across all chunks.

    Returns:
        encoding_maps: per-column encoding metadata
        dynamic_min_max: {column: [min, max]}
    """

    if not isinstance(chunk_files, list) or not chunk_files:
        raise ValueError("chunk_files must be a non-empty list")

    if not isinstance(numerical_columns_info, list):
        raise TypeError("numerical_columns_info must be a list")

    if not os.path.isdir(chunk_dir):
        raise FileNotFoundError(f"Chunk directory not found: {chunk_dir}")

    encoding_dir = get_encoding_dir()
    encoding_maps = {}
    dynamic_min_max = {}

    for info in numerical_columns_info:
        if not isinstance(info, dict):
            raise ValueError(f"Invalid numerical QI info: {info}")

        col = info.get("column")
        encode = bool(info.get("encode", False))
        dtype = info.get("type", "float")

        if not col:
            raise ValueError("Numerical QI missing 'column'")

        all_vals = []
        col_min, col_max = None, None
        multiplier = 1
        logger.info("Encoding column '%s' (type=%s)", col, dtype)

        # --------------------------------------------------
        # Scan all chunks
        # --------------------------------------------------
        for filename in chunk_files:
            file_path = os.path.join(chunk_dir, filename)

            if not os.path.isfile(file_path):
                raise FileNotFoundError(f"Chunk file not found: {file_path}")

            try:
                df = pd.read_csv(file_path)
            except Exception as e:
                raise RuntimeError(f"Failed to read chunk '{filename}': {e}")

            if col not in df.columns:
                raise KeyError(f"Column '{col}' not found in chunk '{filename}'")

            vals = df[col].dropna()
            if vals.empty:
                continue

            # Update min/max
            col_min = vals.min() if col_min is None else min(col_min, vals.min())
            col_max = vals.max() if col_max is None else max(col_max, vals.max())

            # Collect values for encoding
            if encode:
                if dtype == "float":
                    decimals = find_max_decimal_places(vals)
                    multiplier = 10 ** decimals
                    vals = (vals * multiplier).round().astype("int64")
                else:
                    vals = vals.astype("int64")

                all_vals.extend(vals.tolist())

        # --------------------------------------------------
        # Validate min/max
        # --------------------------------------------------
        if col_min is None or col_max is None:
            raise ValueError(f"Column '{col}' has no valid numeric values")

        dynamic_min_max[col] = [float(col_min), float(col_max)]

        # --------------------------------------------------
        # Skip encoding if not required
        # --------------------------------------------------
        if not encode:
            continue

        if not all_vals:
            raise ValueError(f"Column '{col}' has no values to encode")

        # --------------------------------------------------
        # Build encoding maps
        # --------------------------------------------------
        unique_sorted = sorted(set(int(v) for v in all_vals))

        encoding_map = {v: i + 1 for i, v in enumerate(unique_sorted)}
        decoding_map = {i + 1: v for i, v in enumerate(unique_sorted)}

        encoding_maps[col] = {
            "encoding_map": encoding_map,
            "decoding_map": decoding_map,
            "multiplier": int(multiplier),
            "type": dtype,
        }

        # --------------------------------------------------
        # Persist encoding atomically
        # --------------------------------------------------
        encoding_file = os.path.join(encoding_dir, f"{col.lower()}_encoding.json")
        tmp_file = encoding_file + ".tmp"

        try:
            with open(tmp_file, "w") as f:
                json.dump(encoding_maps[col], f, indent=4)
            os.replace(tmp_file, encoding_file)
        except Exception as e:
            raise OSError(f"Failed to write encoding file for '{col}': {e}")

    return encoding_maps, dynamic_min_max
