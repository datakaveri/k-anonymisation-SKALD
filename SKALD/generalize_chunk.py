import pandas as pd
from SKALD.preprocess import suppress, pseudonymize

def generalize_first_chunk(chunk_file, output_path, numerical_columns_info,
                           suppressed_columns, pseudonymized_columns, encoding_maps, ola_2):
    """
    Generalize the first chunk using final RF bin widths and save to CSV.
    """
    print("\nGeneralizing first chunk based on RF...")
    chunk = pd.read_csv(chunk_file)
    working_chunk = chunk.copy()

    if suppressed_columns:
        working_chunk = suppress(working_chunk, suppressed_columns)
    if pseudonymized_columns:
        working_chunk = pseudonymize(working_chunk, pseudonymized_columns)

    # Add encoded columns
    for info in numerical_columns_info:
        column = info["column"]
        encode = info.get("encode", False)
        if encode:
            enc_map = encoding_maps[column]["encoding_map"]
            multiplier = encoding_maps[column]["multiplier"]
            if info.get("type") == "float":
                working_chunk[f"{column}_encoded"] = (working_chunk[column] * multiplier).round().astype(int).map(enc_map)
            else:
                working_chunk[f"{column}_encoded"] = working_chunk[column].map(enc_map)

    generalized_chunk = ola_2.generalize_chunk(working_chunk, ola_2.final_binwidths)

    # Remove encoded columns
    for info in numerical_columns_info:
        if info.get("encode", False):
            col_encoded = f"{info['column']}_encoded"
            if col_encoded in generalized_chunk.columns:
                generalized_chunk.drop(columns=[col_encoded], inplace=True)

    generalized_chunk.to_csv(output_path, index=False)
    print(f"Generalized first chunk saved to: {output_path}")
