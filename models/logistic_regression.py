import copy
import pandas as pd
import numpy as np

class LogisticRegression:
    """
    Logistic regression model for binary classification.

    Attributes:
        data (pandas.DataFrame): Dataset loaded from a CSV file.
        w (list): List of weights for the model.
        b (float): Bias term.
        y (str): Target variable name.
    """
    def __init__(self, path: str, y: str):
        """
        Initializes the Logistic Regression model.

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
            LogisticRegression: The updated instance.
        """
        self.w = w
        return self

    def with_b(self, b: float):
        """
        Sets the bias term of the model.

        Parameters:
            b (float): Bias value.

        Returns:
            LogisticRegression: The updated instance.
        """
        self.b = b
        return self

    def with_fields(self, fields: list):
        """
        Selects specific fields from the dataset.

        Parameters:
            fields (list): List of column names to keep.

        Returns:
            LogisticRegression: The updated instance.
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

    def get_z(self, row: list, w: list, b: float):
        if len(row) != len(w):
            raise ValueError("params length must be equal to w length")
        return sum([row[i] * w[i] for i in range(len(row))]) + b

    def sigmoid(self, z: float):
        """
        Applies the sigmoid function.

        Parameters:
            z (float): Input value.

        Returns:
            float: Sigmoid output.
        """
        return 1 / (1 + np.exp(-z))

    def get_prediction(self, row: list, w: list, b: float):
        """
        Computes the predicted probability for a single row.

        Parameters:
            row (list): List of feature values.
            w (list): List of weights.
            b (float): Bias term.

        Returns:
            float: The predicted probability.
        """
        if len(row) != len(w):
            raise ValueError("row length must be equal to w length")
        return self.sigmoid(self.get_z(row, w, b))

    def get_prediction_list(self, w = None, b = None):
        """
        Computes predictions for all instances in the dataset.

        Parameters:
            w (list, optional): Model weights. Defaults to self.w.
            b (float, optional): Bias term. Defaults to self.b.

        Returns:
            list: List of predicted probabilities.
        """
        if w is None:
            w = self.w
        if b is None:
            b = self.b
        d = self.data.drop(columns=[self.y])
        d = d.values.tolist()
        return [float(self.get_prediction(row, w, b)) for row in d]


    def get_logloss(self, y_predict: list = None):
        """
        Computes the log loss (cross-entropy loss) for the model.

        Parameters:
            y_predict (list, optional): Predicted probabilities. Defaults to self.get_prediction_list().

        Returns:
            float: Log loss value.
        """
        if y_predict is None:
            y_predict = self.get_prediction_list()
        y_real = self.get_real_values()
        return -sum([y_real[i] * np.log(y_predict[i]) + (1 - y_real[i]) * np.log(1 - y_predict[i]) for i in range(len(y_real))]) / len(y_real)

    def get_confussion_matrix(self, y_predict: list = None, threshold: float = 0.5, as_table: bool = False):
        """
        Computes the confusion matrix.

        Parameters:
            y_predict (list, optional): Predicted probabilities. Defaults to self.get_prediction_list().
            threshold (float, optional): Threshold for classification. Defaults to 0.5.
            as_table (bool, optional): If True, returns a formatted table instead of a tuple. Defaults to False.

        Returns:
            tuple: (TP, TN, FP, FN) if as_table is False.
            str: Formatted confusion matrix table if as_table is True.
        """
        if y_predict is None:
            y_predict = self.get_prediction_list()
        y_real = self.get_real_values()
        y_predict = [1 if y > threshold else 0 for y in y_predict]
        tp = sum(1 for i in range(len(y_real)) if y_real[i] == 1 and y_predict[i] == 1)
        tn = sum(1 for i in range(len(y_real)) if y_real[i] == 0 and y_predict[i] == 0)
        fp = sum(1 for i in range(len(y_real)) if y_real[i] == 0 and y_predict[i] == 1)
        fn = sum(1 for i in range(len(y_real)) if y_real[i] == 1 and y_predict[i] == 0)

        if as_table:
            table = f"""
            Confusion Matrix:

            |        | Pred 1 | Pred 0 |
            |--------|--------|--------|
            | Real 1 | {tp:^6} | {fn:^6} |
            | Real 0 | {fp:^6} | {tn:^6} |

            - TP: {tp}
            - TN: {tn}
            - FP: {fp}
            - FN: {fn}

            """
            return table
        return tp, tn, fp, fn

    def get_metrics(self, y_predict: list = None, threshold: float = 0.5, as_table: bool = False):
        """
        Computes classification metrics based on predictions and a given threshold.

        Parameters:
            y_predict (list, optional): List of predicted values. Defaults to self.get_prediction_list().
            threshold (float): Threshold to classify predictions as positive or negative. Defaults to 0.5.

        Returns:
            tuple: (accuracy, precision, recall, f1-score)
        """
        if y_predict is None:
            y_predict = self.get_prediction_list()
        tp, tn, fp, fn = self.get_confussion_matrix(y_predict, threshold)
        accuracy = (tp + tn) / (tp + tn + fp + fn) # Exactitud
        precision = tp / (tp + fp) # Precisión
        recall = tp / (tp + fn) # Exhaustividad
        f1 = 2 * precision * recall / (precision + recall) # F1

        if as_table:
            table = f"""
            Metrics:

            - Exactitud:      {accuracy*100:.2f}%
            - Precision:      {precision*100:.2f}%
            - Exhaustividad:  {recall*100:.2f}%
            - F1-Score:       {f1*100:.2f}%
            """
            return table

        return accuracy, precision, recall, f1

    def get_w_derivate(self, i: int, y_predict: list = None):
        """
        Computes the derivative of the weight parameter w_i for gradient descent.

        Parameters:
            i (int): Index of the weight parameter.
            y_predict (list, optional): List of predicted values. Defaults to self.get_prediction_list().

        Returns:
            float: The computed derivative of w_i.
        """
        if y_predict is None:
            y_predict = self.get_prediction_list()
        y_real = self.get_real_values()
        d = self.data.drop(columns=[self.y])
        d = d.values.tolist()
        if len(y_real) != len(y_predict) or len(y_real) != len(d):
            raise ValueError("y_real, y_predict and d must have the same length")
        return 2 * sum([(y_predict[j] - y_real[j]) * d[j][i] for j in range(len(y_real))]) / len(y_real)

    def get_b_derivate(self, y_predict: list = None):
        """
        Computes the derivative of the bias term b for gradient descent.

        Parameters:
            y_predict (list, optional): List of predicted values. Defaults to self.get_prediction_list().

        Returns:
            float: The computed derivative of b.
        """
        if y_predict is None:
            y_predict = self.get_prediction_list()
        y_real = self.get_real_values()
        if len(y_real) != len(y_predict):
            raise ValueError("y_real and y_predict must have the same length")
        return 2 * sum([y_predict[j] - y_real[j] for j in range(len(y_real))]) / len(y_real)

    def gradient_descent(self, alpha: float, iterations: int):
        """
        Performs gradient descent to optimize the weights and bias.

        Parameters:
            alpha (float): Learning rate.
            iterations (int): Number of iterations.

        Returns:
            tuple: (Updated weights, Updated bias, Updated model instance)
        """
        w = copy.deepcopy(self.w)
        b = copy.deepcopy(self.b)
        for _ in range(iterations):
            pred = self.get_prediction_list(w, b)
            w = [w[i] - alpha * self.get_w_derivate(i, pred) for i in range(len(w))]
            b = b - alpha * self.get_b_derivate(pred)
        logreg = copy.deepcopy(self)
        return [float(x) for x in w], b, logreg.with_b(b).with_w(w)