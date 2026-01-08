# SKALD/generalize_chunk.py

import os
import pandas as pd
import numpy as np
from SKALD.SKALDError import SKALDError
import logging

logger = logging.getLogger("SKALD")


def generalize_single_chunk(
    chunk_file: str,
    chunk_dir: str,
    output_path: str,
    numerical_columns_info,
    encoding_maps,
    ola_2,
    final_rf
):
    """
    Generalize a single chunk using final RF bin widths and save to CSV.
    """

    # -------------------------
    # Validate inputs
    # -------------------------
    if not os.path.isdir(chunk_dir):
        raise SKALDError(
            code="DATA_MISSING",
            message="Chunk directory not found",
            details=chunk_dir,
            suggested_fix="Ensure chunking completed successfully"
        )

    chunk_path = os.path.join(chunk_dir, chunk_file)

    if not os.path.isfile(chunk_path):
        raise SKALDError(
            code="DATA_MISSING",
            message="Chunk file not found",
            details=chunk_path
        )

    # -------------------------
    # Read chunk
    # -------------------------
    try:
        chunk = pd.read_csv(chunk_path)
    except Exception as e:
        raise SKALDError(
            code="DATA_MISSING",
            message="Failed to read chunk CSV",
            details=str(e)
        )

    if chunk.empty:
        raise SKALDError(
            code="DATA_MISSING",
            message="Chunk is empty",
            details=chunk_file
        )

    working_chunk = chunk.copy()
    s_list = []

    # -------------------------
    # Apply numerical scaling and encoding
    # -------------------------
    for info in numerical_columns_info:
        column = info["column"]
        scale  = info.get("scale", False)
        encode = info.get("encode", False)
        s = int(info.get("s", 0)) if scale else 0


        if encode:
            encoded_col = (
                f"{column}_scaled_encoded" if scale else f"{column}_encoded"
            )

            if encoded_col not in working_chunk.columns:
                raise SKALDError(
                    code="ENCODING_FAILED",
                    message=f"Missing encoded column '{encoded_col}' during generalization",
                    suggested_fix="Ensure encoding is applied before generalization"
                )

    s_list.append(s)


    # -------------------------
    # Generalize using OLA_2
    # -------------------------
    try:
        generalized_chunk = ola_2.generalize_chunk(
            working_chunk, final_rf, s_list
        )
    except Exception as e:
        raise SKALDError(
            code="GENERALIZATION_FAILED",
            message="Failed during generalization step",
            details=str(e)
        )

    # -------------------------
    # Drop intermediate columns
    # -------------------------
    for info in numerical_columns_info:
        base = info["column"]
        for suffix in ["_scaled_encoded", "_scaled", "_encoded"]:
            col = f"{base}{suffix}"
            if col in generalized_chunk.columns:
                generalized_chunk.drop(columns=[col], inplace=True)

    # -------------------------
    # Write output
    # -------------------------
    try:
        os.makedirs(os.path.dirname(output_path), exist_ok=True)
        generalized_chunk.to_csv(output_path, index=False)
    except Exception as e:
        raise SKALDError(
            code="INTERNAL_ERROR",
            message="Failed to write generalized output",
            details=str(e)
        )

    return output_path


