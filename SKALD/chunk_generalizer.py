import os
import json
import numpy as np
import pandas as pd

from SKALD.encoder import get_encoding_dir
from SKALD.categorical import CategoricalGeneralizer
from SKALD.SKALDError import SKALDError


class ChunkGeneralizer:
    def __init__(self, quasi_identifiers):
        self.quasi_identifiers = quasi_identifiers
        self.categorical_generalizer = CategoricalGeneralizer()

    def generalize(self, chunk: pd.DataFrame, bin_widths):
        gen_chunk = chunk.copy(deep=False)

        for qi, bw in zip(self.quasi_identifiers, bin_widths):
            col = qi.column_name

            # ---------- CATEGORICAL ----------
            if qi.is_categorical:
                level = int(bw)
                mapping = {
                    "Blood Group": self.categorical_generalizer.generalize_blood_group,
                    "Gender": self.categorical_generalizer.generalize_gender,
                }
                mapper = mapping.get(
                    qi.column_name,
                    self.categorical_generalizer.generalize_profession,
                )
                gen_chunk[col] = gen_chunk[col].map(lambda x: mapper(x, level))
                continue

            # ---------- NUMERICAL ----------
            original_col = col.removesuffix("_encoded") if qi.is_encoded else col

            if qi.is_encoded:
                enc_series = gen_chunk[col].astype(int)

                encoding_file = os.path.join(
                    get_encoding_dir(),
                    f"{original_col.replace(' ', '_').lower()}_encoding.json",
                )

                with open(encoding_file, "r") as f:
                    raw_map = json.load(f)

                encoding_map = raw_map["encoding_map"]
                multiplier = raw_map.get("multiplier", 1)
                decoding_map = {int(v): int(k) for k, v in encoding_map.items()}
            else:
                enc_series = gen_chunk[col].astype(float)
                decoding_map = None
                multiplier = 1

            min_val = int(enc_series.min())
            max_val = int(enc_series.max())
            step = int(max(1, bw))

            encoded_edges = list(range(min_val, max_val + 1, step))
            encoded_edges.append(encoded_edges[-1] + step)
            encoded_edges = np.array(encoded_edges)

            decoded_edges = []
            for v in encoded_edges:
                if decoding_map:
                    real_v = decoding_map.get(v, v) / multiplier
                else:
                    real_v = v
                decoded_edges.append(real_v)

            labels = []
            for i in range(len(decoded_edges) - 1):
                labels.append(f"[{decoded_edges[i]}-{decoded_edges[i + 1] - 1}]")

            gen_chunk[original_col] = pd.cut(
                enc_series,
                bins=encoded_edges,
                labels=labels,
                include_lowest=True,
                right=False,
            )

        return gen_chunk


def generalize_single_chunk(
    chunk_file: str,
    chunk_dir: str,
    output_path: str,
    numerical_columns_info,
    encoding_maps,
    chunk_generalizer: ChunkGeneralizer,
    final_rf,
):
    """
    Pipeline wrapper: read → generalize → write.
    """

    chunk_path = os.path.join(chunk_dir, chunk_file)
    if not os.path.isfile(chunk_path):
        raise SKALDError("DATA_MISSING", "Chunk file not found", chunk_path)

    chunk = pd.read_csv(chunk_path)
    if chunk.empty:
        raise SKALDError("DATA_MISSING", "Chunk is empty", chunk_file)

    working_chunk = chunk.copy()

    # ---------- Apply encoding ----------
    for info in numerical_columns_info:
        if not info.get("encode", False):
            continue

        column = info["column"]
        enc_map = encoding_maps[column]["encoding_map"]
        multiplier = encoding_maps[column].get("multiplier", 1)

        if info.get("type") == "float":
            encoded = (working_chunk[column] * multiplier).round().astype(int)
        else:
            encoded = working_chunk[column].astype(int)

        working_chunk[f"{column}_encoded"] = encoded.map(enc_map)

    # ---------- Apply generalization ----------
    generalized = chunk_generalizer.generalize(working_chunk, final_rf)

    # ---------- Drop encoded columns ----------
    for info in numerical_columns_info:
        if info.get("encode", False):
            col = f"{info['column']}_encoded"
            if col in generalized.columns:
                generalized.drop(columns=[col], inplace=True)

    os.makedirs(os.path.dirname(output_path), exist_ok=True)
    generalized.to_csv(output_path, index=False)

    return output_path
