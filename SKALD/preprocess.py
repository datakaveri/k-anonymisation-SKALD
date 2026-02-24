import hashlib
import pandas as pd
from cryptography.fernet import Fernet
import json
import os
import base64
import secrets
from typing import List, Dict
import logging
from cryptography.hazmat.primitives.ciphers.aead import AESGCM


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

    os.makedirs(output_directory, exist_ok=True)
    key_file = os.path.join(output_directory, "symmetric_keys.json")

    if os.path.exists(key_file):
        with open(key_file, "r") as f:
            key_map = json.load(f)
    else:
        key_map = {}

    for col in columns_to_encrypt:
        if col not in dataframe.columns:
            raise KeyError(f"Column '{col}' not found")

        # Get or create AES key (256-bit)
        if col in key_map:
            key = base64.b64decode(key_map[col])
        else:
            key = AESGCM.generate_key(bit_length=256)
            key_map[col] = base64.b64encode(key).decode()

        aesgcm = AESGCM(key)

        def encrypt_value(x):
            if pd.isna(x):
                return x

            plaintext = str(x).encode()
            nonce = os.urandom(12)  # GCM standard nonce size

            ciphertext = aesgcm.encrypt(nonce, plaintext, None)

            # store nonce + ciphertext together
            token = base64.b64encode(nonce + ciphertext).decode()
            return token

        dataframe[col] = dataframe[col].apply(encrypt_value)

    # Save keys
    tmp_key_file = key_file + ".tmp"
    with open(tmp_key_file, "w") as f:
        json.dump(key_map, f, indent=4)
    os.replace(tmp_key_file, key_file)

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

        start_digits = int(mask.get("start_digits", 0))
        end_digits = int(mask.get("end_digits", 0))
        masking_character = mask.get("masking_character", "*")

        if start_digits < 0 or end_digits < 0:
            raise ValueError("start_digits and end_digits must be non-negative")

        if end_digits < start_digits:
            raise ValueError("end_digits cannot be less than start_digits")

        def mask_value(value):
            s = str(value)
            end_digits_final = min(end_digits, len(s))
            if len(s) <= start_digits:
                logger.log("Value too short to mask: %s returning the actual value", s)
                return s
            
            return (
                s[:start_digits-1] +
                masking_character * (end_digits_final - start_digits + 1) +
                s[end_digits_final:]
            )

        dataframe[column] = dataframe[column].apply(mask_value)
        logger.info("Masked column: %s", column)
    return dataframe
