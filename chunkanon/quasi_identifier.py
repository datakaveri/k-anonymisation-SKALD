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
            self.max_value = float(self.CATEGORICAL_RANGES[column_name] + 1)
        else:
            # Ensure numerical attributes are converted to float
            self.min_value = float(min_value) if min_value is not None else None
            self.max_value = float(max_value) if max_value is not None else None

    def update_min_max(self, chunk):
        """
        Update min and max for numerical attributes based on a new data chunk.

        Args:
            chunk (pd.DataFrame): The chunk to extract min/max from.
        """
        if self.is_categorical:
            return  # No action needed for categorical attributes

        column = chunk[self.column_name]
        chunk_min = column.min(skipna=True)
        chunk_max = column.max(skipna=True)

        if chunk_min is not None:
            self.min_value = min(self.min_value, float(chunk_min)) if self.min_value is not None else float(chunk_min)
        if chunk_max is not None:
            self.max_value = max(self.max_value, float(chunk_max)) if self.max_value is not None else float(chunk_max)

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

        return float(self.max_value - self.min_value)
