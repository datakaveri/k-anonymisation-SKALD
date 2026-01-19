# SKALD/generalize_chunk.py

import os
import pandas as pd
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
    # Read chunk safely
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

    # -------------------------
    # Apply encoding if needed
    # -------------------------
    for info in numerical_columns_info:
        column = info.get("column")
        encode = info.get("encode", False)

        if not encode:
            continue

        if column not in working_chunk.columns:
            raise SKALDError(
                code="DATA_MISSING",
                message=f"Numerical column '{column}' missing in chunk",
                suggested_fix="Verify input dataset columns"
            )

        if column not in encoding_maps:
            raise SKALDError(
                code="ENCODING_FAILED",
                message=f"Encoding map missing for column '{column}'",
                suggested_fix="Enable encoding and rerun pipeline"
            )

        try:
            enc_map = encoding_maps[column]["encoding_map"]
            multiplier = encoding_maps[column].get("multiplier", 1)

            if info.get("type") == "float":
                encoded = (working_chunk[column] * multiplier).round().astype(int)
            else:
                encoded = working_chunk[column].astype(int)

            working_chunk[f"{column}_encoded"] = encoded.map(enc_map)

        except Exception as e:
            raise SKALDError(
                code="ENCODING_FAILED",
                message=f"Failed to encode column '{column}'",
                details=str(e)
            )

    # -------------------------
    # Generalize using OLA_2
    # -------------------------
    try:
        generalized_chunk = ola_2.generalize_chunk(working_chunk, final_rf)
    except Exception as e:
        raise SKALDError(
            code="GENERALIZATION_FAILED",
            message="Failed during generalization step",
            details=str(e)
        )

    # -------------------------
    # Drop encoded columns
    # -------------------------
    for info in numerical_columns_info:
        if info.get("encode", False):
            col = f"{info['column']}_encoded"
            if col in generalized_chunk.columns:
                generalized_chunk.drop(columns=[col], inplace=True)

    # -------------------------
    # Write output safely
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
