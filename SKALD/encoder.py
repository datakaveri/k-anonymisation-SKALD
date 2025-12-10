# SKALD/encoder.py
import os
import pandas as pd
import json
from SKALD.utils import find_max_decimal_places


def get_encoding_dir():
    """Return the encoding directory, creating it if needed."""
    try:
        d = os.getenv("SKALD_ENCODING_DIR", "encodings")
        os.makedirs(d, exist_ok=True)
        return d
    except Exception as e:
        raise OSError(f"Failed to create or access encoding directory '{d}': {e}") from e


def to_py_int(x):
    """Convert numpy.int64 or pandas Int64 to plain Python int."""
    try:
        return int(x)
    except Exception as e:
        raise ValueError(f"Failed to convert value '{x}' to int: {e}") from e


def encode_numerical_columns(chunk_files, chunk_dir, numerical_columns_info):
    """
    Encode numerical columns across all chunks (only when encode=True).
    Always computes min/max for ALL numerical QIs.
    Returns:
        encoding_maps: dict for encoded columns
        dynamic_min_max: dict[min,max] for all numerical QIs
    """

    encoding_dir = get_encoding_dir()
    encoding_maps = {}
    dynamic_min_max = {}

    if not isinstance(chunk_files, list) or not chunk_files:
        raise ValueError("chunk_files must be a non-empty list of filenames.")

    if not os.path.isdir(chunk_dir):
        raise FileNotFoundError(f"Chunk directory does not exist: {chunk_dir}")

    for info in numerical_columns_info:
        col = info.get("column")
        encode = info.get("encode", False)
        dtype = info.get("type", "float")

        if not col:
            raise ValueError(f"Invalid numerical QI info (missing 'column'): {info}")

        all_vals = []               # used only if encoding
        col_min, col_max = None, None

        # -------- scan all chunks for min/max (ALWAYS DONE) --------
        for filename in chunk_files:
            file_path = os.path.join(chunk_dir, filename)

            if not os.path.isfile(file_path):
                raise FileNotFoundError(f"Chunk file not found: {file_path}")

            try:
                df = pd.read_csv(file_path)
            except Exception as e:
                raise RuntimeError(f"Failed to read chunk '{filename}': {e}")

            if col not in df.columns:
                raise KeyError(f"Column '{col}' not found in chunk '{filename}'.")

            vals = df[col].dropna()

            # update global min/max
            if col_min is None:
                col_min, col_max = vals.min(), vals.max()
            else:
                col_min = min(col_min, vals.min())
                col_max = max(col_max, vals.max())

            # -------- collect values ONLY for encoded columns --------
            if encode:
                if dtype == "float":
                    decimals = find_max_decimal_places(vals)
                    multiplier = 10 ** decimals
                    vals = (vals * multiplier).round().astype("int64")
                else:
                    multiplier = 1
                    vals = vals.astype("int64")

                all_vals.extend(int(v) for v in vals.tolist())

        # store min/max for THIS column
        dynamic_min_max[col] = [float(col_min), float(col_max)]

        # -------- skip encoding if encode=False --------
        if not encode:
            continue

        # ===== encoding happens only below =====
        unique_sorted = sorted(set(all_vals))
        if not unique_sorted:
            raise ValueError(f"Column '{col}' has no valid values to encode.")

        encoding_map = {int(v): i + 1 for i, v in enumerate(unique_sorted)}
        decoding_map = {i + 1: int(v) for i, v in enumerate(unique_sorted)}

        encoding_maps[col] = {
            "encoding_map": encoding_map,
            "decoding_map": decoding_map,
            "multiplier": int(multiplier),
            "type": dtype,
        }

        # write encoding json
        fname = f"{col.lower()}_encoding.json"
        with open(os.path.join(encoding_dir, fname), "w") as f:
            json.dump(encoding_maps[col], f, indent=4)

    return encoding_maps, dynamic_min_max
