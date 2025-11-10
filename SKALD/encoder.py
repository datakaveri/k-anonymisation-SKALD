import os
import pandas as pd
import json
from SKALD.preprocess import suppress, pseudonymize
from SKALD.utils import find_max_decimal_places

def encode_numerical_columns(chunk_files, chunk_dir, numerical_columns_info):
    """
    Encode numerical columns across all chunks consistently.
    
    Returns a dictionary: {column_name: {encoding_map, decoding_map, multiplier, type}}
    """
    encoding_maps = {}

    for info in numerical_columns_info:
        column = info["column"]
        encode = info.get("encode", False)
        column_type = info.get("type", "float")

        if not encode:
            continue

        all_values = []

        # Collect all values from all chunks
        for filename in chunk_files:
            chunk = pd.read_csv(os.path.join(chunk_dir, filename))
            values = chunk[column].dropna()
            if column_type == "float":
                decimal_places = find_max_decimal_places(values)
                multiplier = 10 ** decimal_places
                values = (values * multiplier).round().astype(int)
            else:
                multiplier = 1
                values = values.astype(int)

            all_values.extend(values.tolist())

        # Create encoding map
        unique_sorted = sorted(set(all_values))
        encoding_map = {val: idx + 1 for idx, val in enumerate(unique_sorted)}
        decoding_map = {idx: int(val) for val, idx in encoding_map.items()}

        encoding_maps[column] = {
            "encoding_map": encoding_map,
            "decoding_map": decoding_map,
            "multiplier": int(multiplier),
            "type": column_type
        }

        # Save encoding map
        with open(os.path.join(encoding_dir, f"{column.replace(' ','_').lower()}_encoding.json"), "w") as f:
            json.dump(encoding_maps[column], f, indent=4)

    return encoding_maps
