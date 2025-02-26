import copy
import pandas as pd

class LinearRegression:
    """
    Linear regression model

    Attributes:
        - data: pandas.DataFrame
        - w: list
        - b: float
        - y: str
    """
    def __init__(self, path: str, y: str):
        """
        Constructor

        Parameters:
            - path: str
            -----
            - y: str
            -----
        """

        self.data = pd.read_csv(path)
        self.w: list = [1 for _ in range(len(self.data.columns) - 1)]
        self.b: float = 0
        self.y: str = y

    def with_w(self, w: list):
        self.w = w
        return self

    def with_b(self, b: float):
        self.b = b
        return self

    def with_fields(self, fields: list):
        if self.y not in fields:
            fields.append(self.y)
        self.data = self.data[fields]
        return self

    def get_real_values(self):
        return self.data[self.y].values

    def get_prediction(self, params: list, w, b):
        if len(params) != len(w):
            raise ValueError("params length must be equal to w length")
        return sum([params[i] * w[i] for i in range(len(params))]) + b

    def get_prediction_list(self, w = None, b = None):
        if w is None:
            w = self.w
        if b is None:
            b = self.b
        d = self.data.drop(columns=[self.y])
        d = d.values.tolist()
        return [self.get_prediction(row, w, b) for row in d]

    def get_mae(self):
        y_real = self.get_real_values()
        y_pred = self.get_prediction_list()
        return sum([abs(y_real[i] - y_pred[i]) for i in range(len(y_real))]) / len(y_real)

    def get_mse(self):
        y_real = self.get_real_values()
        y_pred = self.get_prediction_list()
        return sum([(y_real[i] - y_pred[i]) ** 2 for i in range(len(y_real))]) / len(y_real)

    def get_rmse(self):
        return self.get_mse() ** 0.5

    def get_w_derivate(self, i: int, y_predict):
        y_real = self.get_real_values()
        d = self.data.drop(columns=[self.y])
        d = d.values.tolist()
        if len(y_real) != len(y_predict) or len(y_real) != len(d):
            raise ValueError("y_real, y_predict and d must have the same length")
        return 2 * sum([(y_predict[j] - y_real[j]) * d[j][i] for j in range(len(y_real))]) / len(y_real)

    def get_b_derivate(self, y_predict):
        y_real = self.get_real_values()
        if len(y_real) != len(y_predict):
            raise ValueError("y_real and y_predict must have the same length")
        return 2 * sum([y_predict[j] - y_real[j] for j in range(len(y_real))]) / len(y_real)

    def gradient_descent(self, alpha: float, iterations: int):
        w = copy.deepcopy(self.w)
        b = copy.deepcopy(self.b)
        for _ in range(iterations):
            pred = self.get_prediction_list(w, b)
            w = [w[i] - alpha * self.get_w_derivate(i, pred) for i in range(len(w))]
            b = b - alpha * self.get_b_derivate(pred)
        return w, b

lr1 = LinearRegression("test.csv", "y").with_b(0).with_w([1, 1, 1, 1]).with_fields(["x1", "x2", "x3", "x4"])

# print(lr1.get_real_values())
# print(lr1.get_prediction_list())
# print(lr1.get_mse())
# print(lr1.get_mae())
# print(lr1.get_rmse())
res1 = lr1.gradient_descent(0.01, 40)

lista_w = [float(x) for x in res1[0]]
b = float(res1[1])

lr2 = LinearRegression("test.csv", "y").with_b(b).with_w(lista_w).with_fields(["x1", "x2", "x3", "x4"])
print(lr2.get_real_values())
print(lr2.get_prediction_list())