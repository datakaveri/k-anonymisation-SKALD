"""
Configuration schema and validation logic for the chunkanon pipeline.

This module defines Pydantic models to load and validate a YAML-based configuration
file for the chunk-based anonymization pipeline. It ensures data types, directory paths,
and logical rules are enforced before processing begins.
"""

from pydantic import BaseModel, Field, conlist, condecimal, field_validator, model_validator
from typing import List, Dict, Optional
import os
import yaml


class NumericalQuasiIdentifier(BaseModel):
    """
    Represents a numerical quasi-identifier (QID) in the dataset.

    Attributes:
        column (str): Name of the column.
        encode (bool): Whether the column should be encoded.
        type (str): Data type of the column, must be either 'int' or 'float'.
    """

    column: str
    encode: bool
    type: str  # Must be either 'int' or 'float'

    @field_validator("type")
    def validate_type(cls, v):
        """Ensure type is either 'int' or 'float'."""
        if v not in {"int", "float"}:
            raise ValueError(f"Invalid type '{v}'. Must be 'int' or 'float'.")
        return v

    @model_validator(mode="before")
    def check_encode_for_float(cls, values):
        """
        Ensure that if type is 'float', then 'encode' must be True.
        This is required for consistent binning logic.
        """
        type_value = values.get("type")
        encode_value = values.get("encode")
        if type_value == "float" and not encode_value:
            raise ValueError(f"When type is 'float', 'encode' must be True.")
        return values


class CategoricalQuasiIdentifier(BaseModel):
    """
    Represents a categorical quasi-identifier (QID) in the dataset.

    Attributes:
        column (str): Name of the column.
    """
    column: str


class QuasiIdentifiers(BaseModel):
    """
    Container for both numerical and categorical quasi-identifiers.

    Attributes:
        numerical (List[NumericalQuasiIdentifier]): List of numerical QIDs.
        categorical (List[CategoricalQuasiIdentifier]): List of categorical QIDs.
    """

    numerical: List[NumericalQuasiIdentifier]
    categorical: List[CategoricalQuasiIdentifier]
    '''
    @model_validator(mode="after")
    def check_at_least_one_numerical(cls, values):
        """Ensure that at least one numerical QID is provided."""
        if not values.numerical:
            raise ValueError("There must be at least one numerical quasi-identifier")
        return values
    '''

class Config(BaseModel):
    """
    Root configuration model for the chunkanon pipeline.

    Attributes:
        number_of_chunks (int): Number of data chunks to create.
        k (int): Value for k-anonymity.
        max_number_of_eq_classes (int): Maximum number of equivalence classes allowed.
        suppression_limit (Decimal): Suppression threshold (0 ≤ value ≤ 1).

        chunk_directory (str): Directory containing chunked input files.
        output_path (str): Path to save final output file.
        log_file (str): Path to the log file.
        save_output (bool): Whether to write output to file.

        quasi_identifiers (QuasiIdentifiers): Configuration for QIDs.
        bin_width_multiplication_factor (Dict[str, int]): Bin width scaling factors per QID.
        hardcoded_min_max (Optional[Dict[str, List[Decimal]]]): Manually specified min/max values per QID.
    """

    number_of_chunks: int
    k: int
    l: int
    max_number_of_eq_classes: int
    suppression_limit: condecimal(ge=0, le=1)

    chunk_directory: str
    output_path: str
    log_file: str
    save_output: bool

    quasi_identifiers: QuasiIdentifiers
    sensitive_parameter:str
    bin_width_multiplication_factor: Dict[str, int]
    hardcoded_min_max: Optional[Dict[str, List[condecimal(ge=0)]]] = {}

    @field_validator("bin_width_multiplication_factor")
    @classmethod
    def validate_multiplication_factors(cls, v):
        """
        Ensure all bin width multiplication factors are greater than 1.
        A factor of 1 or less does not scale the bin width.
        """
        for column, factor in v.items():
            if factor <= 1:
                raise ValueError(f"Multiplication factor for '{column}' must be greater than 1, but found {factor}")
        return v

    @model_validator(mode="after")
    def check_paths(cls, values):
        """
        Check that the directories for chunk input, output path, and log file exist.
        This helps catch missing directories before runtime errors occur.
        """
        chunk_directory = values.chunk_directory
        if not os.path.isdir(chunk_directory):
            raise ValueError(f"Chunk directory does not exist: {chunk_directory}")

        output_dir = os.path.dirname(values.output_path)
        if output_dir and not os.path.exists(output_dir):
            raise ValueError(f"Output directory does not exist: {output_dir}")

        log_dir = os.path.dirname(values.log_file)
        if log_dir and not os.path.exists(log_dir):
            raise ValueError(f"Log directory does not exist: {log_dir}")

        return values


def load_config(config_path: str):
    """
    Load and validate the configuration from a YAML file.

    Args:
        config_path (str): Path to the YAML configuration file.

    Returns:
        Config: Validated configuration object.

    Raises:
        ValidationError: If the configuration does not meet schema or validation rules.
    """
    from pydantic import ValidationError

    with open(config_path, "r") as f:
        config_data = yaml.safe_load(f)

    try:
        config = Config(**config_data)
        print("Configuration validated successfully!")
        return config
    except ValidationError as e:
        print("Validation errors occurred:")
        print(e.json())
        raise
