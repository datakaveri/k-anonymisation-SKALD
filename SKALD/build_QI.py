from SKALD.quasi_identifier import QuasiIdentifier


def build_quasi_identifiers(
    numerical_columns_info,
    categorical_columns,
    encoding_maps,
    hardcoded_min_max
):
    """
    Generates quasi-identifiers and list of column names with full error handling.

    Args:
        numerical_columns_info (list[dict])
        categorical_columns (list[str])
        encoding_maps (dict[str, dict])
        hardcoded_min_max (dict[str, (min, max)])

    Returns:
        (list[QuasiIdentifier], list[str])

    Raises:
        TypeError: Incorrect argument types.
        KeyError: Missing required fields in numerical config.
        ValueError: Invalid ranges, missing encoding info, or no QIs defined.
    """

    # --------------------
    # Validate high-level types
    # --------------------
    if not isinstance(numerical_columns_info, list):
        raise TypeError("numerical_columns_info must be a list of dictionaries.")

    if not isinstance(categorical_columns, list):
        raise TypeError("categorical_columns must be a list of strings.")

    if not isinstance(encoding_maps, dict):
        raise TypeError("encoding_maps must be a dictionary.")

    if not isinstance(hardcoded_min_max, dict):
        raise TypeError("hardcoded_min_max must be a dictionary.")

    quasi_identifiers = []
    all_quasi_columns = []

    # --------------------
    # Numerical QIs
    # --------------------
    for info in numerical_columns_info:
        if not isinstance(info, dict):
            raise TypeError(
                f"Each numerical column info must be a dict, got {type(info).__name__}."
            )

        # Required key
        if "column" not in info:
            raise KeyError("Each numerical column entry must contain a 'column' key.")

        column = info["column"]
        encode = bool(info.get("encode", False))

        encoded_column = f"{column}_encoded" if encode else column
        all_quasi_columns.append(encoded_column)

        # ---- Handle encoded numerical column
        if encode:
            if column not in encoding_maps:
                raise KeyError(
                    f"Encoding requested for column '{column}', but it is missing in encoding_maps."
                )

            map_obj = encoding_maps[column].get("encoding_map")
            if not isinstance(map_obj, dict):
                raise ValueError(
                    f"Invalid encoding_map for column '{column}'. Must be a dict."
                )

            if len(map_obj) == 0:
                raise ValueError(
                    f"Encoding map for column '{column}' is empty."
                )

            min_val, max_val = 1, len(map_obj)

        else:
            # ---- Unencoded numerical QI must have min/max
            if column not in hardcoded_min_max:
                raise KeyError(
                    f"Missing hardcoded_min_max entry for unencoded numerical column '{column}'. "
                    f"Provide min/max in the configuration."
                )

            min_max = hardcoded_min_max[column]
            if (
                not isinstance(min_max, (list, tuple))
                or len(min_max) != 2
            ):
                raise ValueError(
                    f"hardcoded_min_max['{column}'] must be a 2-element list or tuple [min, max]."
                )

            min_val, max_val = min_max

            if min_val > max_val:
                raise ValueError(
                    f"Invalid hardcoded_min_max for '{column}': min_val cannot be > max_val."
                )

        quasi_identifiers.append(
            QuasiIdentifier(
                encoded_column,
                is_categorical=False,
                is_encoded=encode,
                min_value=min_val,
                max_value=max_val,
            )
        )

    # --------------------
    # Categorical QIs
    # --------------------
    for column in categorical_columns:
        if not isinstance(column, str):
            raise TypeError(f"Categorical column names must be strings, got {type(column).__name__}.")

        all_quasi_columns.append(column)
        quasi_identifiers.append(
            QuasiIdentifier(column, is_categorical=True, is_encoded=False)
        )

    # --------------------
    # Final safety check
    # --------------------
    if not quasi_identifiers:
        raise ValueError("No quasi-identifiers defined. Provide numerical or categorical QIs.")

    return quasi_identifiers, all_quasi_columns
