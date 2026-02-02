import hashlib
import pandas as pd
from cryptography.fernet import Fernet
import json
import os
import base64
import secrets
from typing import List, Dict
import logging
logger = logging.getLogger("SKALD")


# --------------------------------------------------
# Utilities
# --------------------------------------------------
def generate_global_salt(length_bytes: int = 32) -> str:
    """
    Generates a cryptographically secure global salt.
    """
    return base64.urlsafe_b64encode(
        secrets.token_bytes(length_bytes)
    ).decode()


# --------------------------------------------------
# Suppression
# --------------------------------------------------
def suppress(dataframe: pd.DataFrame, suppressed_columns: List[str]) -> pd.DataFrame:
    if not isinstance(suppressed_columns, list):
        logger.error("suppressed_columns is not a list: %s", type(suppressed_columns).__name__)
        raise TypeError("suppressed_columns must be a list")

    missing = [c for c in suppressed_columns if c not in dataframe.columns]
    if missing:
        raise KeyError(f"Columns not found for suppression: {missing}")
    logger.info("Suppressed columns: %s", suppressed_columns)
    return dataframe.drop(columns=suppressed_columns)


# --------------------------------------------------
# Hashing
# --------------------------------------------------
def hash_columns(
    dataframe: pd.DataFrame,
    columns_with_salt: List[str],
    columns_without_salt: List[str]
) -> pd.DataFrame:

    if not isinstance(columns_with_salt, list) or not isinstance(columns_without_salt, list):
        logger.error("Hashing column lists are not lists: %s, %s",
                     type(columns_with_salt).__name__,
                     type(columns_without_salt).__name__)
        raise TypeError("Hashing column lists must be lists")

    salt = generate_global_salt() if columns_with_salt else None
    logger.debug("Generated global salt for hashing")
    for col in columns_with_salt:
        if col not in dataframe.columns:
            raise KeyError(f"Column '{col}' not found for salted hashing")

        dataframe[col] = dataframe[col].astype(str).apply(
            lambda x: hashlib.sha256((salt + x).encode()).hexdigest()
            if x.lower() != "nan" else x
        )
        logger.info("Applied salted hashing to column: %s", col)
    for col in columns_without_salt:
        if col not in dataframe.columns:
            raise KeyError(f"Column '{col}' not found for hashing")

        dataframe[col] = dataframe[col].astype(str).apply(
            lambda x: hashlib.sha256(x.encode()).hexdigest()
            if x.lower() != "nan" else x
        )
        logger.info("Applied hashing to column: %s", col)
    return dataframe


# --------------------------------------------------
# Encryption
# --------------------------------------------------
def encrypt_columns(
    dataframe: pd.DataFrame,
    columns_to_encrypt: List[str],
    output_directory: str
) -> pd.DataFrame:

    if not isinstance(columns_to_encrypt, list):
        logger.error("columns_to_encrypt is not a list: %s", type(columns_to_encrypt).__name__)
        raise TypeError("columns_to_encrypt must be a list")

    os.makedirs(output_directory, exist_ok=True)
    key_file = os.path.join(output_directory, "symmetric_keys.json")

    # Load existing keys
    if os.path.exists(key_file):
        try:
            with open(key_file, "r") as f:
                key_map = json.load(f)
        except json.JSONDecodeError as e:
            raise RuntimeError(f"Corrupted key file '{key_file}': {e}")
    else:
        key_map = {}

    for col in columns_to_encrypt:
        if col not in dataframe.columns:
            raise KeyError(f"Column '{col}' not found for encryption")

        # Get or create key
        if col in key_map:
            key = key_map[col].encode()
        else:
            key = Fernet.generate_key()
            key_map[col] = key.decode()

        fernet = Fernet(key)

        try:
            dataframe[col] = dataframe[col].astype(str).apply(
                lambda x: fernet.encrypt(x.encode()).decode()
                if x.lower() != "nan" else x
            )
        except Exception as e:
            raise RuntimeError(f"Encryption failed for column '{col}': {e}")
        logger.info("Encrypted column: %s", col)
    # Persist keys atomically
    tmp_key_file = key_file + ".tmp"
    try:
        with open(tmp_key_file, "w") as f:
            json.dump(key_map, f, indent=4)
        os.replace(tmp_key_file, key_file)
    except Exception as e:
        raise OSError(f"Failed to write encryption key file: {e}")

    return dataframe


# --------------------------------------------------
# Masking
# --------------------------------------------------
def mask_columns(dataframe: pd.DataFrame, masking_info: List[Dict]) -> pd.DataFrame:
    if not isinstance(masking_info, list):
        logger.error("masking_info is not a list: %s", type(masking_info).__name__)
        raise TypeError("masking_info must be a list of dictionaries")

    for mask in masking_info:
        if not isinstance(mask, dict):
            raise ValueError("Each masking entry must be a dictionary")

        column = mask.get("column")
        if not column:
            raise ValueError("Masking config missing 'column'")

        if column not in dataframe.columns:
            raise KeyError(f"Column '{column}' not found for masking")
        
        
        masking_character = mask.get("masking_character", "*")
        characters_to_mask = mask.get("characters_to_mask", [])
        if (not isinstance(characters_to_mask, list) or
            not all(isinstance(i, int) and i > 0 for i in characters_to_mask)):
            raise ValueError("'characters_to_mask' must be a list of positive integers")
        if not characters_to_mask:
            raise ValueError("'characters_to_mask' cannot be empty")
        


        def mask_value(value):
            s = list(str(value))
            n = len(s)

            for pos in characters_to_mask:
                idx = pos - 1
                if 0 <= idx < n:
                    s[idx] = masking_character

            return ''.join(s)


        dataframe[column] = dataframe[column].apply(mask_value)
        logger.info("Masked column: %s", column)
    return dataframe
