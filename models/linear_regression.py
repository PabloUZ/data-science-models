import copy
import pandas as pd

class LinearRegression:
    """
    Linear regression model.

    Attributes:
        data (pandas.DataFrame): Dataset loaded from a CSV file.
        w (list): List of weights for the model.
        b (float): Bias term.
        y (str): Target variable name.
    """
    def __init__(self, path: str, y: str):
        """
        Initializes the Linear Regression model.

        Parameters:
            path (str): Path to the CSV file containing the dataset.
            y (str): Name of the target variable.
        """

        self.data = pd.read_csv(path)
        self.w: list = [1 for _ in range(len(self.data.columns) - 1)]
        self.b: float = 0
        self.y: str = y

    def with_w(self, w: list):
        """
        Sets the weights of the model.

        Parameters:
            w (list): List of weights.

        Returns:
            LinearRegression: The updated instance.
        """
        self.w = w
        return self

    def with_b(self, b: float):
        """
        Sets the bias term of the model.

        Parameters:
            b (float): Bias value.

        Returns:
            LinearRegression: The updated instance.
        """
        self.b = b
        return self

    def with_fields(self, fields: list):
        """
        Selects specific fields from the dataset.

        Parameters:
            fields (list): List of column names to keep.

        Returns:
            LinearRegression: The updated instance.
        """
        if self.y not in fields:
            fields.append(self.y)
        self.data = self.data[fields]
        return self

    def get_real_values(self):
        """
        Retrieves the actual target values from the dataset.

        Returns:
            numpy.ndarray: Array of actual target values.
        """
        return self.data[self.y].values

    def get_prediction(self, row: list, w: list, b: float):
        """
        Computes the predicted value for a single row.

        Parameters:
            row (list): List of feature values.
            w (list): List of weights.
            b (float): Bias term.

        Returns:
            float: The predicted value.
        """
        if len(row) != len(w):
            raise ValueError("row length must be equal to w length")
        return sum([row[i] * w[i] for i in range(len(row))]) + b

    def get_prediction_list(self, w = None, b = None):
        """
        Computes predictions for all instances in the dataset.

        Parameters:
            w (list, optional): Model weights. Defaults to self.w.
            b (float, optional): Bias term. Defaults to self.b.

        Returns:
            list: List of predicted values.
        """
        if w is None:
            w = self.w
        if b is None:
            b = self.b
        d = self.data.drop(columns=[self.y])
        d = d.values.tolist()
        return [float(self.get_prediction(row, w, b)) for row in d]

    def get_mae(self, y_predict: list):
        """
        Computes the Mean Absolute Error (MAE).

        Parameters:
            y_predict (list): List of predicted values.

        Returns:
            float: The MAE score.
        """
        y_real = self.get_real_values()
        return sum([abs(y_real[i] - y_predict[i]) for i in range(len(y_real))]) / len(y_real)

    def get_mse(self, y_predict: list):
        """
        Computes the Mean Squared Error (MSE).

        Parameters:
            y_predict (list): List of predicted values.

        Returns:
            float: The MSE score.
        """
        y_real = self.get_real_values()
        return sum([(y_real[i] - y_predict[i]) ** 2 for i in range(len(y_real))]) / len(y_real)

    def get_rmse(self, y_predict: list):
        """
        Computes the Root Mean Squared Error (RMSE).

        Parameters:
            y_predict (list): List of predicted values.

        Returns:
            float: The RMSE score.
        """
        return self.get_mse(y_predict) ** 0.5

    def get_w_derivate(self, i: int, y_predict: list):
        """
        Computes the derivative of the loss function with respect to weight w[i].

        Parameters:
            i (int): Index of the weight.
            y_predict (list): List of predicted values.

        Returns:
            float: The computed derivative.
        """
        y_real = self.get_real_values()
        d = self.data.drop(columns=[self.y])
        d = d.values.tolist()
        if len(y_real) != len(y_predict) or len(y_real) != len(d):
            raise ValueError("y_real, y_predict and d must have the same length")
        return 2 * sum([(y_predict[j] - y_real[j]) * d[j][i] for j in range(len(y_real))]) / len(y_real)

    def get_b_derivate(self, y_predict: list):
        """
        Computes the derivative of the loss function with respect to the bias term.

        Parameters:
            y_predict (list): List of predicted values.

        Returns:
            float: The computed derivative.
        """
        y_real = self.get_real_values()
        if len(y_real) != len(y_predict):
            raise ValueError("y_real and y_predict must have the same length")
        return 2 * sum([y_predict[j] - y_real[j] for j in range(len(y_real))]) / len(y_real)

    def gradient_descent(self, alpha: float, iterations: int):
        """
        Performs gradient descent optimization to update weights and bias.

        Parameters:
            alpha (float): Learning rate.
            iterations (int): Number of iterations.

        Returns:
            tuple: Updated weights, bias, and updated model.
        """
        w = copy.deepcopy(self.w)
        b = copy.deepcopy(self.b)
        for _ in range(iterations):
            pred = self.get_prediction_list(w, b)
            w = [w[i] - alpha * self.get_w_derivate(i, pred) for i in range(len(w))]
            b = b - alpha * self.get_b_derivate(pred)
        lr = copy.deepcopy(self)
        return [float(x) for x in w], b, lr.with_b(b).with_w(w)

