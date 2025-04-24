class QuasiIdentifier:
    """
    Represents a quasi-identifier column in a dataset.
    Handles both categorical and numerical attributes.
    """

    # Defines max generalization level for categorical QIs
    CATEGORICAL_RANGES = {
        "Blood Group": 2,  
        "Profession": 3     
    }

    def __init__(self, column_name: str, is_categorical: bool = False, min_value=None, max_value=None):
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

        if self.is_categorical and column_name in self.CATEGORICAL_RANGES:
            self.min_value = 0
            self.max_value = self.CATEGORICAL_RANGES[column_name]
        else:
            self.min_value = min_value
            self.max_value = max_value

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
            self.min_value = min(self.min_value, chunk_min) if self.min_value is not None else chunk_min
        if chunk_max is not None:
            self.max_value = max(self.max_value, chunk_max) if self.max_value is not None else chunk_max

    def get_range(self):
        """
        Compute the generalization range.

        Returns:
            int or float: The range (max - min) for numerical QIs, or fixed level count for categoricals.
        """
        if self.is_categorical:
            return self.CATEGORICAL_RANGES.get(self.column_name, 0)

        if self.min_value is None or self.max_value is None:
            return 0  # Safeguard in case min/max are not defined

        return self.max_value - self.min_value
