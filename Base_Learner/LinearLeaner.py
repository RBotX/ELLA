import numpy as np
from sklearn.linear_model import LinearRegression
from sklearn.metrics import explained_variance_score


class LinearModel:

    def __init__(self): pass

    def fit(self, X, y):
        learner = LinearRegression(fit_intercept=False).fit(X, y)
        theta = np.mat(learner.coef_).T
        X = np.mat(X)
        hessian = X.T * X / X.shape[0] / 2.0
        return theta, hessian

    def predict(self, X, theta):
        return X * theta

    def score(self, y_pred, y):
        return explained_variance_score(y_pred, y)
