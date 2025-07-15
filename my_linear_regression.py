import numpy as np


class MyLinearRegression:
    def __init__(self, learning_rate=0.01, num_iters=1000):
        self.lr = learning_rate
        self.num_iters = num_iters
        self.weights = None
        self.bias = None
        self.cost_history = []

    def hypothesis(self, X):
        return np.dot(X, self.weights) + self.bias

    def compute_cost(self, X, y):
        m, n = X.shape
        y_hat = self.hypothesis(X)
        error = y_hat - y
        cost = np.sum(error ** 2) / (2 * m)

        return cost

    def compute_gradients(self, X, y):
        m, n = X.shape
        y_hat = self.hypothesis(X)
        error = y_hat - y

        dw = (1 / m) * np.dot(X.T, error)
        db = (1 / m) * np.sum(error)

        return dw, db

    def fit(self, X, y):
        m, n = X.shape

        # Initialize weights and bias
        self.weights = np.zeros((n, 1))
        self.bias = 0.0

        for _ in range(self.num_iters):
            dw, db = self.compute_gradients(X, y)
            self.weights -= self.lr * dw
            self.bias -= self.lr * db

            self.cost_history.append(self.compute_cost(X, y))

        return self

    def predict(self, X):
        return self.hypothesis(X)

    def score(self, X, y):
        return self.compute_cost(X, y)
