from models.linear_regression import LinearRegression
from models.logistic_regression import LogisticRegression
from tools.regularization import Regularization
from tools.normailzation import Normalization

lr1 = LinearRegression("test.csv", "y").with_b(0).with_w([-3, 4, 5, 0]).with_fields(["x1", "x2", "x3", "x4"])
lista_w, b, lr2 = lr1.gradient_descent(0.01, 40)

print(lr2.get_real_values())
print(lr2.get_prediction_list())

print("-------------------")
print("-------------------")

logreg1 = LogisticRegression("test.csv", "y").with_b(0).with_w([-3, 4, 5, 0]).with_fields(["x1", "x2", "x3", "x4"])
lista_w, b, logreg2 = logreg1.gradient_descent(0.01, 100)

print(logreg1.get_real_values())
print(logreg1.get_prediction_list())
print(logreg1.get_confussion_matrix())
print("-------------------")
print(logreg2.get_real_values())
print(logreg2.get_prediction_list())
print(logreg2.get_metrics(as_table=True))

print(lr1.data["x1"].values)

# print(Regularization(lr1, 1))

print("################")
print("################")
print(Normalization(lr1).scale('x4').scale_log('x1').z_score('x2'))