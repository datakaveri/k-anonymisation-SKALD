"""
Configuration schema and validation logic for the SKALD pipeline.
"""

from pydantic import BaseModel, Field, condecimal, field_validator, model_validator, ValidationError
from typing import List, Dict, Optional
import os
import yaml
import logging
logger = logging.getLogger("SKALD")


# -------------------------------
# Numerical QI
# -------------------------------
class NumericalQuasiIdentifier(BaseModel):
    column: str
    scale: Optional[bool] = None
    s: Optional[int] = 0
    encode: Optional[bool] = None
    type: str

    @field_validator("type")
    def validate_type(cls, v):
        if v not in {"int", "float"}:
            logger.error("Invalid type for numerical QI: %s", v)
            raise ValueError("type must be 'int' or 'float'")
        logger.debug("Validated numerical QI type: %s", v)
        return v
    '''
    @model_validator(mode="before")
    def check_encode_for_float(cls, values):
        if not isinstance(values, dict):
            logger.error("Invalid structure for numerical QI: %s", values)
            raise ValueError("Invalid numerical QI structure")

        if values.get("type") == "float" and not values.get("encode"):
            logger.error("Float numerical QI must have encode=True: %s", values)
            raise ValueError("Float numerical QIs must have encode=True")
        logger.debug("Validated numerical QI encoding requirement: %s", values)
        return values
    '''

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
        
        if not values.numerical and not values.categorical:
            logger.error("No quasi-identifiers defined")
            raise ValueError("At least one quasi-identifier must be defined")
        logger.debug("Validated quasi-identifiers: %s", values)
        return values


# -------------------------------
# Main Config Object
# -------------------------------
class Config(BaseModel):
    k: Optional[int] = None
    l: Optional[int] = None
    suppression_limit: Optional[condecimal(ge=0, le=1)] = 0

    suppress: List[str] = Field(default_factory=list)
    hashing_with_salt: List[str] = Field(default_factory=list)
    hashing_without_salt: List[str] = Field(default_factory=list)
    masking: List[Dict] = Field(default_factory=list)
    encrypt: List[str] = Field(default_factory=list)

    enable_k_anonymity: bool = True
    enable_l_diversity: bool = False

    output_path: str
    output_directory: str
    log_file: str

    quasi_identifiers: Optional[QuasiIdentifiers] = None
    sensitive_parameter: Optional[str] = None
    size: Optional[Dict[str, int]] = {}


    @model_validator(mode="after")
    def validate_k_l_and_qis(cls, values):
        if not values.enable_k_anonymity:
            logger.debug("K-anonymity disabled; skipping k, l, and quasi-identifier validations")
            return values

        if values.k is None:
            logger.error("k value must be provided when k-anonymity is enabled")
            raise ValueError("k must be provided when k-anonymity is enabled")

        if values.enable_l_diversity and values.l is None:
            logger.error("l value must be provided when l-diversity is enabled")
            raise ValueError("l must be provided when l-diversity is enabled")

        if values.quasi_identifiers is None:
            logger.error("quasi_identifiers must be provided when k-anonymity is enabled")
            raise ValueError("quasi_identifiers must be provided when k-anonymity is enabled")
        logger.debug("Validated k, l, and quasi-identifiers: %s", values)   
        return values


    @field_validator("size")
    def validate_multiplication_factors(cls, v):
        if not isinstance(v, dict):
            logger.error("Invalid size configuration: %s", v)
            raise ValueError("size must be a dictionary")

        for col, factor in v.items():
            if not isinstance(factor, int) or factor <= 1:
                logger.error("Invalid multiplication factor for column '%s': %s", col, factor)
                raise ValueError(f"Multiplication factor for '{col}' must be > 1")
        logger.debug("Validated multiplication factors: %s", v)
        return v


# -------------------------------
# Config Loader
# -------------------------------
def load_config(config_path: str) -> Config:
    if not isinstance(config_path, str):
        logger.error("Config path must be a string")
        raise TypeError("Config path must be a string")

    if not os.path.isfile(config_path):
        logger.error("Config file not found: %s", config_path)
        raise FileNotFoundError(f"Config file not found: {config_path}")

    try:
        with open(config_path, "r") as f:
            config_data = yaml.safe_load(f)
    except yaml.YAMLError as e:
        logger.error("Invalid YAML syntax in config file: %s", e)
        raise ValueError(f"Invalid YAML syntax: {e}")
    except Exception as e:
        logger.error("Unable to read config file: %s", e)
        raise RuntimeError(f"Unable to read config file: {e}")

    if not isinstance(config_data, dict):
        logger.error("Config file must contain a YAML dictionary")
        raise ValueError("Config file must contain a YAML dictionary")

    # Handle non-k-anonymity mode
    if not config_data.get("enable_k_anonymity", True):
        config_data["quasi_identifiers"] = None
        config_data["k"] = None
        config_data["l"] = None

    try:
        return Config(**config_data)
    except ValidationError as e:
        # IMPORTANT: preserve full details
        raise ValueError(str(e))
