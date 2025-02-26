import copy
import numpy as np
import pandas as pd

class DecisionTree:
    """
    Decision Tree model for classification based on the Gini impurity criterion.

    Attributes:
        data (pandas.DataFrame): Dataset loaded from a CSV file.
        max_depth (int): Maximum depth of the decision tree (default is 1).
    """
    def __init__(self, path):
        """
        Initializes the DecisionTree model by loading data from a CSV file.

        Parameters:
            path (str): Path to the CSV file containing the dataset.
        """
        self.data = pd.read_csv(path)
        self.max_depth = 1

    def with_max_depth(self, max_depth):
        """
        Sets the maximum depth of the decision tree.

        Parameters:
            max_depth (int): The maximum depth to set.

        Returns:
            DecisionTree: The updated instance.
        """
        self.max_depth = max_depth
        return self

    def with_fields(self, fields):
        """
        Filters the dataset to include only specified fields. Ensures that the target variable 'y' is always included.

        Parameters:
            fields (list): List of column names to keep in the dataset.

        Returns:
            DecisionTree: The updated instance.
        """
        if 'y' not in fields:
            fields.append('y')
        self.data = self.data[fields]
        return self

    def gini(self, x: list):
        """
        Computes the Gini impurity of a given list of values.

        Parameters:
            x (list): List of class labels.

        Returns:
            float: The Gini impurity score.
        """
        _, freq = np.unique(x, return_counts=True)
        proportions = freq / freq.sum()
        return 1 - np.sum(proportions ** 2)

    def best_gini(self, x: str, y: str):
        """
        Finds the best partition point for a given feature based on the Gini impurity.

        Parameters:
            x (str): The feature column used for partitioning.
            y (str): The target variable column.

        Returns:
            tuple: A tuple containing the minimum Gini impurity (rounded to 3 decimal places) and the best partition value.
        """
        data = copy.deepcopy(self.data)
        data = data.sort_values(by=x).reset_index(drop=True)
        gini = float('inf')
        partition = None
        for i in range(len(data[x].values)-1):
            middle = (data[x].values[i] + data[x].values[i+1]) / 2
            left = data[data[x] < middle]
            right = data[data[x] >= middle]
            prev_gini = copy.deepcopy(gini)
            gini = min(gini, (len(left)/len(data)) * self.gini(left[y]) + (len(right)/len(data)) * self.gini(right[y]))
            if prev_gini != gini:
                partition = middle
        return round(gini, 3), partition