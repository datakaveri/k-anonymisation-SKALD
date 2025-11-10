import os
import pandas as pd
from SKALD.preprocess import suppress, pseudonymize

def process_chunks_for_histograms(chunk_files, chunk_dir, numerical_columns_info, encoding_maps,ola_2,initial_ri):
    """
    Process each chunk: apply suppression/pseudonymization, encode numerical columns,
    and generate chunk histograms using OLA_2.
    Returns a list of histograms.
    """
    histograms = []

    for i, filename in enumerate(chunk_files):
        chunk = pd.read_csv(os.path.join(chunk_dir, filename))
        working_chunk = chunk.copy()
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

        chunk_histogram = ola_2.process_chunk(working_chunk, initial_ri)
        histograms.append(chunk_histogram)
        print(f"Processed chunk {i+1}/{len(chunk_files)} for histograms.")

    return histograms
