import numpy as np

class LinearRegression:

    def __init__(self, X, Y):
        ones = np.ones((X.shape[0], 1))
        self.X = np.hstack((ones, X))
        self.Y = Y.reshape(-1, 1)

        self.n = self.X.shape[0]
        self.d = self.X.shape[1] - 1

        self.beta = None
        self.y_hat = None
        self.SSE = None
        self.sigma2 = None

    def fit(self):
        XT = self.X.T
        self.beta = np.dot(np.linalg.inv(np.dot(XT, self.X)), np.dot(XT, self.Y))
        self.y_hat = np.dot(self.X, self.beta)

    def variance(self):
        self.SSE = np.sum((self.Y - self.y_hat) ** 2)
        self.sigma2 = self.SSE / (self.n - self.d - 1)
        return self.sigma2

    def std(self):
        return np.sqrt(self.variance())

    def rmse(self):
        mse = np.mean((self.Y - self.y_hat) ** 2)
        return np.sqrt(mse)