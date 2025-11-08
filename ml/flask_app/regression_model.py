import numpy as np

class MyRidgeRegression:

    def __init__(self, learning_rate=0.01, n_iterations=1000, lambda_val=0.1):
        self.learning_rate=learning_rate
        self.n_iterations=n_iterations
        self.lambda_val=lambda_val 
        self.weights =None
        self.bias=None

    def fit(self, X, y):
        n_samples, n_features = X.shape
        cost_history = []
        self.weights = np.zeros(n_features)
        self.bias = 0

        for _ in range(self.n_iterations):
            y_pred = np.dot(X, self.weights) + self.bias

            mse = (1 / n_samples) * np.sum((y_pred - y) ** 2)
            l2_penalty = (self.lambda_val / n_samples) * np.sum(np.square(self.weights))
            cost = mse + l2_penalty
            cost_history.append(cost)
            dw = (1 / n_samples) * np.dot(X.T, (y_pred - y))
            db = (1 / n_samples) * np.sum(y_pred - y)
            dw_reg = (self.lambda_val / n_samples) * (2 * self.weights)
            dw += dw_reg
            self.weights-= self.learning_rate * dw
            self.bias -=self.learning_rate * db

        return cost_history

    def predict(self, X):
        return np.dot(X,self.weights)+self.bias


def mean_squared_error(y_true, y_pred):
    return np.mean((y_true-y_pred)** 2)

def r2_score(y_true, y_pred):
    tss = np.sum((y_true-np.mean(y_true))**2)
    rss = np.sum((y_true-y_pred)**2)
    r2 =1-(rss/tss)
    return r2