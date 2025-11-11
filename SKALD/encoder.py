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
    Encode all numerical columns across all chunks.
    Safely handles reading, encoding, decimal detection, and file output.

    Returns:
        encoding_maps (dict)
    """

    encoding_dir = get_encoding_dir()
    encoding_maps = {}

    if not isinstance(chunk_files, list) or not chunk_files:
        raise ValueError("chunk_files must be a non-empty list of filenames.")

    if not os.path.isdir(chunk_dir):
        raise FileNotFoundError(f"Chunk directory does not exist: {chunk_dir}")

    # Loop through numerical QIs
    for info in numerical_columns_info:
        col = info.get("column")
        if not col:
            raise ValueError(f"Invalid numerical QI info (missing 'column'): {info}")

        should_encode = info.get("encode", False)
        dtype = info.get("type", "float")

        if not should_encode:
            continue  # skip non-encoded columns

        if dtype not in {"int", "float"}:
            raise ValueError(f"Invalid dtype '{dtype}' for column '{col}'. Expected 'int' or 'float'.")

        all_vals = []

        # Read all chunks to collect values
        for filename in chunk_files:
            file_path = os.path.join(chunk_dir, filename)

            if not os.path.isfile(file_path):
                raise FileNotFoundError(f"Chunk file not found: {file_path}")

            try:
                df = pd.read_csv(file_path)
            except pd.errors.EmptyDataError:
                raise ValueError(f"Chunk file '{filename}' is empty.")
            except Exception as e:
                raise RuntimeError(f"Failed to read chunk '{filename}': {e}") from e

            if col not in df.columns:
                raise KeyError(f"Column '{col}' not found in chunk '{filename}'.")

            vals = df[col].dropna()

            # Float encoding
            if dtype == "float":
                try:
                    decimals = find_max_decimal_places(vals)
                except Exception as e:
                    raise RuntimeError(
                        f"Failed to determine decimal places for column '{col}' in chunk '{filename}': {e}"
                    ) from e

                try:
                    multiplier = 10 ** decimals
                    vals = (vals * multiplier).round().astype("int64")
                except Exception as e:
                    raise RuntimeError(
                        f"Failed to apply multiplier to float column '{col}' in chunk '{filename}': {e}"
                    ) from e

            else:
                multiplier = 1
                try:
                    vals = vals.astype("int64")
                except Exception as e:
                    raise ValueError(f"Column '{col}' contains non-integer values: {e}") from e

            # Convert to python ints
            try:
                all_vals.extend(int(v) for v in vals.tolist())
            except Exception as e:
                raise RuntimeError(
                    f"Failed converting encoded values for column '{col}' to Python int: {e}"
                ) from e

        # Build sorted uniques
        try:
            unique_sorted = sorted(set(all_vals))
        except Exception as e:
            raise RuntimeError(f"Failed sorting values for column '{col}': {e}") from e

        if not unique_sorted:
            raise ValueError(f"Column '{col}' has no non-null values across chunks; cannot encode.")

        # Build encoding + decoding maps
        try:
            encoding_map = {to_py_int(v): to_py_int(i + 1) for i, v in enumerate(unique_sorted)}
            decoding_map = {to_py_int(i + 1): to_py_int(v) for i, v in enumerate(unique_sorted)}
        except Exception as e:
            raise RuntimeError(f"Failed building encoding maps for '{col}': {e}") from e

        encoding_maps[col] = {
            "encoding_map": encoding_map,
            "decoding_map": decoding_map,
            "multiplier": int(multiplier),
            "type": dtype,
        }

        # Write encoding JSON to file
        try:
            fname = f"{col.lower()}_encoding.json"
            outpath = os.path.join(encoding_dir, fname)

            with open(outpath, "w") as f:
                json.dump(encoding_maps[col], f, indent=4)

        except Exception as e:
            raise OSError(
                f"Failed writing encoding file '{fname}' in '{encoding_dir}': {e}"
            ) from e

    return encoding_maps
