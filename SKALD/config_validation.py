"""
Configuration schema and validation logic for the chunkanon pipeline.
"""

from pydantic import BaseModel, Field, condecimal, field_validator, model_validator, ValidationError
from typing import List, Dict, Optional
import os
import yaml


# -------------------------------
# Numerical QI
# -------------------------------
class NumericalQuasiIdentifier(BaseModel):
    column: str
    encode: bool
    type: str

    @field_validator("type")
    def validate_type(cls, v):
        if v not in {"int", "float"}:
            raise ValueError(f"Invalid type '{v}'. Must be 'int' or 'float'.")
        return v

    @model_validator(mode="before")
    def check_encode_for_float(cls, values):
        if not isinstance(values, dict):
            raise ValueError("Invalid structure for NumericalQuasiIdentifier.")

        t = values.get("type")
        encode = values.get("encode")

        if t == "float" and not encode:
            raise ValueError("When type is 'float', 'encode' must be True.")

        return values


# -------------------------------
# Categorical QI
# -------------------------------
class CategoricalQuasiIdentifier(BaseModel):
    column: str


# -------------------------------
# QI Containers
# -------------------------------
class QuasiIdentifiers(BaseModel):
    numerical: List[NumericalQuasiIdentifier] = Field(default_factory=list)
    categorical: List[CategoricalQuasiIdentifier] = Field(default_factory=list)

    @model_validator(mode="after")
    def validate_qis(cls, values):
        if not isinstance(values.numerical, list) or not isinstance(values.categorical, list):
            raise ValueError("Numerical and categorical QI lists must be valid lists.")

        if len(values.numerical) == 0 and len(values.categorical) == 0:
            raise ValueError("At least one quasi-identifier must be defined.")

        return values


# -------------------------------
# Main Config Object
# -------------------------------
class Config(BaseModel):
    k: Optional[int] = None
    l: Optional[int] = None
    suppression_limit: Optional[condecimal(ge=0, le=1)] = 0

    suppress: List[str] = Field(default_factory=list)
    pseudonymize: List[str] = Field(default_factory=list)
    encrypt: List[str] = Field(default_factory=list)
    enable_k_anonymity: bool = True
    enable_l_diversity: bool = False
    output_path: str
    output_directory: str
    key_directory: str
    log_file: str

    quasi_identifiers: Optional[QuasiIdentifiers] = None
    sensitive_parameter: Optional[str] = None
    size: Optional[Dict[str, int]] = {}


    @model_validator(mode="after")
    def check_k_fields(cls, values):
        """
        Validate k, l, and quasi_identifiers only when k-anonymity is enabled.
        """
        if not values.enable_k_anonymity:
            # In non-k-anonymity mode: no QI, k, l required
            return values

        # k-anonymity mode ON → all must exist
        if values.k is None:
            raise ValueError("'k' must be provided when k-anonymity is enabled.")
        if values.enable_l_diversity and values.l is None:
            raise ValueError("'l' must be provided when l-diversity is enabled.")

        if values.quasi_identifiers is None:
            raise ValueError("'quasi_identifiers' must be provided when k-anonymity is enabled.")

        # Must have at least one QI
        if (
            not values.quasi_identifiers.numerical 
            and not values.quasi_identifiers.categorical
        ):
            raise ValueError("At least one quasi-identifier must be defined.")

        return values


    @field_validator("size")
    def validate_multiplication_factors(cls, v):
        if not isinstance(v, dict):
            raise ValueError("size must be a dictionary.")

        for col, factor in v.items():
            if not isinstance(factor, int):
                raise ValueError(f"Factor for '{col}' must be an integer.")
            if factor <= 1:
                raise ValueError(f"Multiplication factor for '{col}' must be > 1. Found {factor}.")
        return v

    @model_validator(mode="after")
    def validate_paths(cls, values):
        # Output directory
        if not os.path.isdir(values.output_directory):
            raise ValueError(f"Output directory does not exist: {values.output_directory}")

        # Key directory
        if not os.path.isdir(values.key_directory):
            raise ValueError(f"Key directory does not exist: {values.key_directory}")

        # Log file directory
        log_dir = os.path.dirname(values.log_file)
        if log_dir and not os.path.isdir(log_dir):
            raise ValueError(f"Log directory does not exist: {log_dir}")

        # Output file parent check
        out_dir_check = os.path.dirname(values.output_path)
        if out_dir_check and not os.path.isdir(out_dir_check):
            raise ValueError(f"Directory for output_path does not exist: {out_dir_check}")

        return values


# -------------------------------
# Config Loader Function
# -------------------------------
def load_config(config_path: str):
    """
    Load and validate YAML configuration with proper error handling.
    Handles both k-anonymity enabled and disabled modes safely.
    """
    if not isinstance(config_path, str):
        raise TypeError("Config path must be a string.")

    if not os.path.isfile(config_path):
        raise FileNotFoundError(f"Config file not found: {config_path}")

    # --- Read YAML safely ---
    try:
        with open(config_path, "r") as f:
            config_data = yaml.safe_load(f)
    except yaml.YAMLError as e:
        raise ValueError(f"Invalid YAML syntax in config file: {e}")
    except Exception as e:
        raise RuntimeError(f"Unable to read config file '{config_path}': {e}")

    if not isinstance(config_data, dict):
        raise ValueError("Config file must contain a valid YAML dictionary.")

    # --- Special handling when k-anonymity disabled ---
    enable_k = config_data.get("enable_k_anonymity", True)

    # If disabled, ensure the fields that Pydantic expects exist (but can be empty)
    if not enable_k:
        # If quasi_identifiers exists as empty dict or empty structure, disable it
        if "quasi_identifiers" in config_data:
            if not config_data["quasi_identifiers"]:
                config_data["quasi_identifiers"] = None
        else:
            config_data["quasi_identifiers"] = None

        # Disable k & l too
        config_data["k"] = None
        config_data["l"] = None


    # --- Validate using Pydantic ---
    try:
        config = Config(**config_data)
        print("Configuration validated successfully!")
        return config

    except ValidationError as e:
        print("Configuration validation error:")
        print(e)
        raise ValueError("Invalid configuration structure. See details above.")

