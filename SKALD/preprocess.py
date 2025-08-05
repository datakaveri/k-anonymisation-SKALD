import hashlib
import pandas as pd

def suppress(dataframe, suppressed_columns):
    """
    Suppresses (removes) the specified columns from the dataframe.

    Args:
        dataframe (pd.DataFrame): Input data.
        suppressed_columns (List[str]): List of column names to drop.

    Returns:
        pd.DataFrame: DataFrame with specified columns removed.
    """
    dataframe.drop(columns=suppressed_columns, inplace=True, errors='ignore')
    return dataframe

def pseudonymize(dataframe, pseudonymized_columns):
    """
    Pseudonymizes by hashing a combination of specified columns.

    Args:
        dataframe (pd.DataFrame): Input data.
        pseudonymized_columns (List[str]): List of column names to combine and hash.

    Returns:
        pd.DataFrame: DataFrame with hashed UID and original columns removed.
    """
    if len(pseudonymized_columns) < 2:
        return dataframe  # Skip if not enough columns

    uid_col = dataframe[pseudonymized_columns[0]].astype(str)
    for col in pseudonymized_columns[1:]:
        uid_col += dataframe[col].astype(str)

    dataframe["UID"] = uid_col
    dataframe["Hashed Value"] = dataframe["UID"].apply(lambda x: hashlib.sha256(x.encode()).hexdigest())

    dataframe.drop(columns=["UID"] + pseudonymized_columns, inplace=True, errors='ignore')
    return dataframe
