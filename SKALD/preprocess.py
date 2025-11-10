import hashlib
import pandas as pd
from cryptography.fernet import Fernet
import json
import os


def suppress(dataframe, suppressed_columns):
    """
    Suppresses (removes) the specified columns from the dataframe.
    """
    dataframe.drop(columns=suppressed_columns, inplace=True, errors='ignore')
    return dataframe


def pseudonymize(dataframe, pseudonymized_columns):
    """
    Pseudonymizes by hashing a combination of specified columns.
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


def decrypt_columns(dataframe, key_file):
    """
    Decrypts encrypted columns using stored keys.

    Args:
        dataframe (pd.DataFrame): Input data (with encrypted columns).
        key_file (str): Path to JSON file with keys.

    Returns:
        pd.DataFrame: Decrypted dataframe.
    """
    if not os.path.exists(key_file):
        raise FileNotFoundError(f"Key file not found: {key_file}")

    with open(key_file, "r") as f:
        key_map = json.load(f)

    for col, key in key_map.items():
        if col not in dataframe.columns:
            continue

        fernet = Fernet(key.encode())
        dataframe[col] = dataframe[col].apply(
            lambda x: fernet.decrypt(x.encode()).decode()
        )

    return dataframe
