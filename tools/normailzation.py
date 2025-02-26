import copy
import pandas as pd
import numpy as np

class Normalization:
    """
    Normalization class for preprocessing a dataset.

    This class provides various methods for normalizing and encoding a pandas DataFrame,
    including one-hot encoding, scaling, log scaling, outlier pruning, and z-score normalization.

    Attributes:
        model (pandas.DataFrame): A deep copy of the input DataFrame to avoid modifying the original data.
    """
    def __init__(self, model: pd.DataFrame):
        """
        Initializes the Normalization class with a deep copy of the provided DataFrame.

        Parameters:
            model (pandas.DataFrame): The DataFrame to be normalized.
        """
        self.model = copy.deepcopy(model)

    def hot_encode(self, x: str):
        """
        Performs one-hot encoding on the specified column.

        Parameters:
            x (str): The column name to be one-hot encoded.

        Returns:
            Normalization: The updated instance with one-hot encoding applied.
        """
        self.model.data = pd.get_dummies(self.model.data, columns=[x], drop_first=True)
        return self

    def scale(self, x: str):
        """
        Scales the values of the specified column to a range between 0 and 1.

        Parameters:
            x (str): The column name to be scaled.

        Returns:
            Normalization: The updated instance with scaling applied.
        """
        max_value = self.model.data[x].max()
        min_value = self.model.data[x].min()
        self.model.data[x] = (self.model.data[x] - min_value) / (max_value - min_value)
        return self

    def scale_log(self, x: str):
        """
        Applies logarithmic scaling to the specified column.

        Parameters:
            x (str): The column name to be log-scaled.

        Returns:
            Normalization: The updated instance with log scaling applied.
        """
        min_value = self.model.data[x].min()
        if min_value <= 0:
            self.model.data[x] = np.log10(self.model.data[x] - min_value + 1)
        else:
            self.model.data[x] = np.log10(self.model.data[x] + 1)
        return self

    def prune_outliers(self, x: str, top: float, bottom: float):
        """
        Prunes outliers in the specified column based on quantile thresholds.

        Parameters:
            x (str): The column name to prune outliers from.
            top (float): Upper quantile threshold (e.g., 0.95 for 95th percentile).
            bottom (float): Lower quantile threshold (e.g., 0.05 for 5th percentile).

        Returns:
            Normalization: The updated instance with outlier pruning applied.
        """
        top_value = self.model.data[x].quantile(top)
        bottom_value = self.model.data[x].quantile(bottom)
        self.model.data[x] = self.model.data[x].clip(lower=bottom_value, upper=top_value)
        return self

    def z_score(self, x: str):
        """
        Normalizes the specified column using the z-score method.

        Parameters:
            x (str): The column name to normalize.

        Returns:
            Normalization: The updated instance with z-score normalization applied.
        """
        mean = self.model.data[x].mean()
        std = self.model.data[x].std()
        self.model.data[x] = (self.model.data[x] - mean) / std
        return self

    def __str__(self):
        """
        Returns a string representation of the normalized DataFrame.

        Returns:
            str: String representation of the DataFrame.
        """
        return str(self.model.data)

    def __call__(self):
        """
        Returns the normalized DataFrame.

        Returns:
            pandas.DataFrame: The modified DataFrame.
        """
        return self.model