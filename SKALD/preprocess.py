import hashlib
import pandas as pd
from cryptography.fernet import Fernet
import json
import os
import base64
import secrets

def generate_global_salt(length_bytes=32):
    """
    Generates a cryptographically secure global salt.
    """
    return base64.urlsafe_b64encode(
        secrets.token_bytes(length_bytes)
    ).decode()

def suppress(dataframe, suppressed_columns):
    """
    Suppresses (removes) the specified columns from the dataframe.
    """
    dataframe.drop(columns=suppressed_columns, inplace=True, errors='ignore')
    return dataframe


def hash_columns(dataframe, columns_with_salt, columns_without_salt):
    """
    Hashes specified columns using SHA-256.
    Columns in 'columns_with_salt' are hashed with a salt.
    Columns in 'columns_without_salt' are hashed without a salt.

    Args:
        dataframe (pd.DataFrame): Input data.
        columns_with_salt (List[str]): Columns to hash with salt.
        columns_without_salt (List[str]): Columns to hash without salt.
        salt (str): Salt value for hashing.

    Returns:
        pd.DataFrame: DataFrame with hashed columns.
    """

    
    # Hash columns with salt
    for col in columns_with_salt:
        salt = generate_global_salt()
        if col in dataframe.columns:
            dataframe[col] = dataframe[col].astype(str).apply(
                lambda x: hashlib.sha256((salt + x).encode()).hexdigest()
            )
        else:
            print(f"[WARN] Column '{col}' not found in dataset — skipping.")

    # Hash columns without salt
    for col in columns_without_salt:
        if col in dataframe.columns:
            dataframe[col] = dataframe[col].astype(str).apply(
                lambda x: hashlib.sha256(x.encode()).hexdigest()
            )
        else:
            print(f"[WARN] Column '{col}' not found in dataset — skipping.")

    return dataframe


def encrypt_columns(dataframe, columns_to_encrypt, key_dir="keys"):
    """
    Encrypts each specified column with Fernet encryption.
    If a column already has a key in 'column_keys.json', reuse it; 
    otherwise generate and append a new one.

    Args:
        dataframe (pd.DataFrame): Input data.
        columns_to_encrypt (List[str]): Columns to encrypt.
        key_dir (str): Directory to store encryption keys.

    Returns:
        pd.DataFrame: DataFrame with encrypted columns.
    """
    os.makedirs(key_dir, exist_ok=True)
    key_file = os.path.join(key_dir, "column_keys.json")

    # Load existing key map if available
    if os.path.exists(key_file):
        with open(key_file, "r") as f:
            try:
                key_map = json.load(f)
            except json.JSONDecodeError:
                key_map = {}
    else:
        key_map = {}

    # Encrypt specified columns
    for col in columns_to_encrypt:
        if col not in dataframe.columns:
            print(f"[WARN] Column '{col}' not found in dataset — skipping.")
            continue

        # Reuse existing key if present, otherwise generate new one
        if col in key_map:
            key = key_map[col].encode()
            print(f"[INFO] Reusing existing key for column '{col}'")
        else:
            key = Fernet.generate_key()
            key_map[col] = key.decode()
            print(f"[INFO] Generated new key for column '{col}'")

        fernet = Fernet(key)

        # Encrypt values (convert to str, handle NaNs)
        dataframe[col] = dataframe[col].astype(str).apply(
            lambda x: fernet.encrypt(x.encode()).decode() if x.lower() != 'nan' else x
        )

    # Save updated key map
    with open(key_file, "w") as f:
        json.dump(key_map, f, indent=4)

    print(f"[INFO] Encryption keys saved to: {key_file}")
    return dataframe

def mask_columns(dataframe, masking_info):
    """
    Masks specified columns based on provided masking information.

    Args:
        dataframe (pd.DataFrame): Input data.
        masking_info (List[Dict]): List of masking configurations.

    Returns:
        pd.DataFrame: DataFrame with masked columns.
    """
    for mask in masking_info:
        column = mask.get("column")
        start_digits = mask.get("start_digits", 0)
        end_digits = mask.get("end_digits", 0)
        masking_character = mask.get("masking_character", "*")

        if column not in dataframe.columns:
            print(f"[WARN] Column '{column}' not found in dataset — skipping.")
            continue

        def mask_value(value):
            str_value = str(value)
            if len(str_value) <= start_digits + end_digits:
                return masking_character * len(str_value)
            return (
                str_value[:start_digits] +
                masking_character * (len(str_value) - start_digits - end_digits) +
                str_value[-end_digits:]
            )

        dataframe[column] = dataframe[column].apply(mask_value)

    return dataframe



