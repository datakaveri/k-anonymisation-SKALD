from typing import List, Dict, Tuple
from SKALD.quasi_identifier import QuasiIdentifier
import logging
logger = logging.getLogger("SKALD")


def build_quasi_identifiers(
    numerical_columns_info: List[Dict],
    categorical_columns: List[str],
    encoding_maps: Dict[str, Dict],
    dynamic_min_max: Dict[str, List[float]]
) -> Tuple[List[QuasiIdentifier], List[str]]:
    """
    Build quasi-identifiers from numerical and categorical configuration.

    Raises:
        TypeError: Incorrect argument types
        KeyError: Missing required fields or encoding info
        ValueError: Invalid ranges or empty QI definitions
        RuntimeError: QuasiIdentifier construction failures
    """

    # ------------------------------------------------------------------
    # Validate high-level inputs
    # ------------------------------------------------------------------
    if not isinstance(numerical_columns_info, list):
        raise TypeError("numerical_columns_info must be a list of dictionaries")

    if not isinstance(categorical_columns, list):
        raise TypeError("categorical_columns must be a list of strings")

    if not isinstance(encoding_maps, dict):
        raise TypeError("encoding_maps must be a dictionary")

    if not isinstance(dynamic_min_max, dict):
        raise TypeError("dynamic_min_max must be a dictionary")

    quasi_identifiers: List[QuasiIdentifier] = []
    all_quasi_columns: List[str] = []

    # ------------------------------------------------------------------
    # Numerical quasi-identifiers
    # ------------------------------------------------------------------
    for info in numerical_columns_info:
        if not isinstance(info, dict):
            raise TypeError(
                f"Each numerical QI must be a dict, got {type(info).__name__}"
            )

        if "column" not in info:
            raise KeyError("Numerical QI entry missing required key: 'column'")

        column = info["column"]
        if not isinstance(column, str) or not column.strip():
            raise ValueError("Numerical QI 'column' must be a non-empty string")
        scale = bool(info.get("scale", False))
        s = int(info.get("s", 0))
        encode = bool(info.get("encode", False))
        if scale and encode:
            effective_column = f"{column}_scaled_encoded"
        elif encode:
            effective_column = f"{column}_encoded"
        elif scale:
            effective_column = f"{column}_scaled"
        else:
            effective_column = column

        all_quasi_columns.append(effective_column)

        # --------------------
        # Encoded numerical QI
        # --------------------
        if encode:
            if column not in encoding_maps:
                raise KeyError(
                    f"Encoding requested for column '{column}' but encoding_maps entry is missing"
                )

            enc_info = encoding_maps[column]
            encoding_map = enc_info.get("encoding_map")
            logger.info("Building encoded QI for column '%s'", column)
            if not isinstance(encoding_map, dict) or not encoding_map:
                raise ValueError(
                    f"Invalid or empty encoding_map for encoded column '{column}'"
                )

            min_val, max_val = 1, len(encoding_map)

        # --------------------
        # Unencoded numerical QI
        # --------------------
        else:
            if column not in dynamic_min_max:
                raise KeyError(
                    f"Missing dynamic_min_max for numerical column '{column}'"
                )

            min_max = dynamic_min_max[column]
            logger.info("Building unencoded QI for column '%s'", column)
            if (
                not isinstance(min_max, (list, tuple))
                or len(min_max) != 2
            ):
                raise ValueError(
                    f"dynamic_min_max['{column}'] must be [min, max]"
                )

            min_val, max_val = min_max

            if min_val is None or max_val is None:
                raise ValueError(
                    f"dynamic_min_max for '{column}' contains None values"
                )

            if min_val > max_val:
                raise ValueError(
                    f"Invalid range for '{column}': min ({min_val}) > max ({max_val})"
                )

        # --------------------
        # Construct QuasiIdentifier
        # --------------------
        try:
            qi = QuasiIdentifier(
                effective_column,
                is_categorical=False,
                is_scaled=scale,
                is_encoded=encode,
                min_value=min_val,
                max_value=max_val,
            )
        except Exception as e:
            raise RuntimeError(
                f"Failed to construct QuasiIdentifier for numerical column '{column}': {e}"
            )

        quasi_identifiers.append(qi)

    # ------------------------------------------------------------------
    # Categorical quasi-identifiers
    # ------------------------------------------------------------------
    for column in categorical_columns:
        if not isinstance(column, str) or not column.strip():
            raise ValueError(
                f"Categorical column names must be non-empty strings, got '{column}'"
            )

        all_quasi_columns.append(column)

        try:
            qi = QuasiIdentifier(
                column,
                is_categorical=True,
                is_scaled=False,
                is_encoded=False
            )
        except Exception as e:
            raise RuntimeError(
                f"Failed to construct QuasiIdentifier for categorical column '{column}': {e}"
            )

        quasi_identifiers.append(qi)

    # ------------------------------------------------------------------
    # Final validation
    # ------------------------------------------------------------------
    if not quasi_identifiers:
        raise ValueError(
            "No quasi-identifiers defined. Provide at least one numerical or categorical QI"
        )

    # Ensure deterministic order & uniqueness of column names
    all_quasi_columns = list(dict.fromkeys(all_quasi_columns))

    return quasi_identifiers, all_quasi_columns
