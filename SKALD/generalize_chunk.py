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
    s_by_column = {}

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
        s_by_column[column] = s


    # -------------------------
    # Generalize using OLA_2
    # -------------------------
    try:
        # Align scaling factor list with OLA_2 quasi-identifier order.
        aligned_s_list = []
        for qi in ola_2.quasi_identifiers:
            base_col = qi.column_name
            for suffix in ("_scaled_encoded", "_encoded", "_scaled"):
                if base_col.endswith(suffix):
                    base_col = base_col[: -len(suffix)]
                    break
            aligned_s_list.append(int(s_by_column.get(base_col, 0)))

        generalized_chunk = ola_2.generalize_chunk(
            working_chunk, final_rf, aligned_s_list
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
    # Mark suppressed records (QI columns only)
    # -------------------------
    try:
        if hasattr(ola_2, "mark_suppressed_qi_values"):
            generalized_chunk = ola_2.mark_suppressed_qi_values(
                generalized_chunk,
                getattr(ola_2, "current_k", None),
            )
    except Exception as e:
        raise SKALDError(
            code="GENERALIZATION_FAILED",
            message="Failed while marking suppressed records",
            details=str(e)
        )

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
