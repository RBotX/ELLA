import numpy as np
from scipy.linalg import sqrtm
from sklearn.linear_model import LinearRegression, Lasso
from sklearn.metrics import accuracy_score, explained_variance_score



class ELLA:

    def __init__(self, K, dim, lam, mu):
        # Init Parameters
        self.dim = dim  # Scale of Input
        self.lam, self.mu = lam, mu  # Parameters of Object Function

        # Share Knowledge Base
        self.K = K
        self.L = np.mat(np.zeros((dim, K)))
        self.L_free = [i for i in range(K)]
        # self.reinitialize_knowledge()

        # Task Parameters
        self.T = 0
        self.task_dict = {}
        self.S = np.mat(np.zeros((K, 0)))
        self.A = np.mat(np.zeros((dim * K, dim * K)))
        self.b = np.mat(np.zeros((dim * K, 1)))
        self.task_leaner = {}
        self.task_param = {}

    def fit(self, X, y, task_id, learner = None, X_test = None, y_test = None):
        new_task_flag = False
        if task_id not in self.task_dict: # Get a new task
            self.task_dict[task_id] = self.T
            task_id = self.T
            self.T = self.T + 1
            new_task_flag = True
        else:
            task_id = self.task_dict[task_id]
            (theta, hessian) = self.task_param[task_id]
            s = self.S[:, task_id]
            self.A = self.A - np.kron(s * s.T, hessian)
            self.b = self.b - np.kron(s.T, (theta.T * hessian)).T

        self.task_leaner[task_id] = learner

        theta, hessian = self.task_leaner[task_id].fit(X, y)
        if new_task_flag and len(self.L_free) > 0:
            s = np.zeros((self.K, 1))
            s[self.L_free.pop(0)] = 1.0
        else:
            s = self.encode_s(hessian, theta, 1.0 / X.shape[0] / 2.0)

        self.task_param[task_id] = (theta, hessian)

        self.S = np.hstack((self.S, s))
        self.A = self.A + np.kron(s*s.T, hessian)
        self.b = self.b + np.kron(s.T, (theta.T * hessian)).T

        self.L = (self.A / self.T + self.lam * np.eye(self.dim * self.K, self.dim * self.K)).I * (self.b / self.T)
        self.L = self.L.reshape(self.K, self.dim).T

        self.reinitialize_knowledge()


    def encode_s(self, hessian, theta, ratio):
        hessian_sqrt = sqrtm(hessian)

        weight = hessian_sqrt * self.L
        target = hessian_sqrt * theta

        lasso = Lasso(fit_intercept=False, alpha=self.mu * ratio, max_iter=10000).fit(weight, target)

        return np.mat(lasso.coef_).T

    def reinitialize_knowledge(self):
        eps = 1e-8
        for i in range(self.dim):
            if i in self.L_free: continue
            flag = True
            for j in range(self.K):
                if abs(self.L[i, j]) > eps:
                    flag = False
                    break
            if flag : self.L_free.append(i)

    def predict(self, X, task_id):
        if task_id not in self.task_dict:
            print("Error : This task is not trained before...")
            return np.zeros((X.shape[0], 1))

        task_id = self.task_dict[task_id]
        theta = self.L * self.S[:, task_id]
        return self.task_leaner[task_id].predict(X, theta)

    def score(self, X, y, task_id):
        if task_id not in self.task_dict:
            print("Error : This task is not trained before...")
            return 0

        y_pred = self.predict(X, task_id)

        return self.task_leaner[self.task_dict[task_id]].score(y_pred, y)













