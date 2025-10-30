from SKALD.quasi_identifier import QuasiIdentifier

def build_quasi_identifiers(numerical_columns_info, categorical_columns, encoding_maps, hardcoded_min_max):
    """
    Generates quasi-identifiers and list of column names.

    Args:
        numerical_columns_info (list): List of numerical QI info dicts.
        categorical_columns (list): List of categorical QI column names.
        encoding_maps (dict): Encoding maps for numerical columns.
        hardcoded_min_max (dict): Hardcoded min/max for numerical columns.

    Returns:
        tuple: (quasi_identifiers (list of QuasiIdentifier), all_quasi_columns (list of str))
    """
    quasi_identifiers = []
    all_quasi_columns = []

    # Numerical QIs
    for info in numerical_columns_info:
        column = info["column"]
        encode = info.get("encode", False)

        encoded_column = f"{column}_encoded" if encode else column
        all_quasi_columns.append(encoded_column)

        if encode:
            min_val, max_val = 1, len(encoding_maps[column]["encoding_map"])
        else:
            min_val, max_val = hardcoded_min_max.get(column, (0, 0))

        quasi_identifiers.append(
            QuasiIdentifier(
                encoded_column,
                is_categorical=False,
                is_encoded=encode,
                min_value=min_val,
                max_value=max_val
            )
        )

    # Categorical QIs
    for column in categorical_columns:
        all_quasi_columns.append(column)
        quasi_identifiers.append(
            QuasiIdentifier(column, is_categorical=True, is_encoded=False)
        )

    if not quasi_identifiers:
        raise ValueError("No quasi-identifiers defined.")

    return quasi_identifiers, all_quasi_columns
