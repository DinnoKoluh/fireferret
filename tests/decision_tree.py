import numpy as np

from fireferret.trees import DecisionTree
from fireferret.utils.dataset import split_dataset

dt = DecisionTree()
print(dt.healthcheck())

X = [1, 2, 3, 4, 5, 6, 7, 8, 9, 10]
y = ["a", "b", "c", "d", "e", "f", "g", "h", "i", "j"]

X_train, y_train, X_test, y_test = split_dataset(np.array(X), np.array(y, dtype=object), 0.77)

print(X_train)
print(X_test)
print(y_train)
print(y_test)
