import logging
logger = logging.getLogger("SKALD")


class QuasiIdentifier:
    """
    Represents a quasi-identifier column in a dataset.
    Handles both categorical and numerical attributes.
    """

    CATEGORICAL_RANGES = {
        "Blood Group": 3,
        "Gender" : 2,  
        "Profession": 4     
    }

    def __init__(self, column_name: str, is_categorical: bool = False, is_encoded: bool = False, min_value=None, max_value=None):
        """
        Initializes a QuasiIdentifier object.

        Args:
            column_name (str): Name of the column.
            is_categorical (bool): True if the column is categorical.
            min_value (float/int, optional): Minimum value for numerical attributes.
            max_value (float/int, optional): Maximum value for numerical attributes.
        """
        self.column_name = column_name
        self.is_categorical = is_categorical
        self.is_encoded = is_encoded

        if self.is_categorical and column_name in self.CATEGORICAL_RANGES:
            self.min_value = 1.0  # Default to float for categorical attributes
            self.max_value = float(self.CATEGORICAL_RANGES[column_name])
        else:
            # Ensure numerical attributes are converted to float
            self.min_value = float(min_value) if min_value is not None else None
            self.max_value = float(max_value) if max_value is not None else None


    def get_range(self):
        """
        Compute the generalization range.

        Returns:
            int or float: The range (max - min) for numerical QIs, or fixed level count for categoricals.
        """
        if self.is_categorical:
            return float(self.CATEGORICAL_RANGES.get(self.column_name, 0))

        if self.min_value is None or self.max_value is None:
            return 0.0  
        
        return float(self.max_value - self.min_value + 1)
