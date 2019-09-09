import numpy as np
from sklearn.linear_model import LogisticRegression
from sklearn.metrics import accuracy_score, roc_auc_score


class LogisticModel:

    def __init__(self):
        self.learner = LogisticRegression()

    def fit(self, X, y):
        self.learner.fit(X, y)
        theta = np.mat(self.learner.coef_).T

        cnt1 = 0
        for _ in y: cnt1 = cnt1 + (1 if _ else 0)
        setattr(self, "bias", 1.0 - 1.0 * cnt1 / y.shape[0])

        hessian = np.mat(np.zeros((X.shape[1], X.shape[1])))
        for x in X:
            x = np.mat(x)
            sigma = 1.0 / (1.0 + np.exp(-x * theta))
            sigma = sigma[0, 0]
            hessian = hessian + x.T * x * sigma * (1.0 - sigma)
        hessian = hessian / 2.0 / X.shape[0]
        return theta, hessian

    def predict_prob(self, X, theta):
        return 1.0 / (1.0 + np.exp(-(np.mat(X) * theta)))

    def predict(self, X, theta):
        return np.array(self.predict_prob(X, theta)) > self.bias

    def score(self, y_pred, y):
        return accuracy_score(y_pred=y_pred, y_true=y)
