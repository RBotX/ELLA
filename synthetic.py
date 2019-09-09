from ELLA import ELLA
import numpy as np
from sklearn.model_selection import train_test_split
from sklearn.linear_model import LinearRegression, LogisticRegression
from sklearn.metrics import explained_variance_score, accuracy_score
import warnings
warnings.filterwarnings('ignore')
from Base_Learner.LinearLeaner import LinearModel
from Base_Learner.LogisticLeaner import LogisticModel

def to_bin(y): return y > 0

T = 100 # Tasks
d = 13 # Features
nt = 100 # Size of samples
k = 6 # Size of mem
typ = 3 # Type of different w
noise_var = .5 # Noise

w_true = np.mat(np.random.randn(typ, d))
X_train_comb, y_train_comb = np.zeros((0, d)), np.zeros((0, 1))
X_test_pool, y_test_pool = [], []
X_train_pool, y_train_pool = [], []

for t in range(T):
    X_set = np.mat(np.hstack((np.random.randn(nt, d-1), np.ones((nt, 1)))))
    w = w_true[t%typ]
    y_set = X_set * w.T + noise_var * np.mat(np.random.randn(nt, 1))
    X_train, X_test, y_train, y_test = train_test_split(X_set, y_set, test_size=0.1, random_state=19260817)

    X_test_pool.append(X_test)
    y_test_pool.append(y_test)
    X_train_pool.append(X_train)
    y_train_pool.append(y_train)

    X_train_comb = np.vstack((X_train_comb, X_train))
    y_train_comb = np.vstack((y_train_comb, y_train))

ella_linear = ELLA(K=k, dim=d, mu=np.exp(-12), lam=np.exp(-10))
for t in range(T):
    ella_linear.fit(X_train_pool[t], y_train_pool[t], t, LinearModel())
print("ELLA Linear", np.mean([ella_linear.score(X_test_pool[i], y_test_pool[i], i) for i in range(T)]))

linear_regression = LinearRegression().fit(X_train_comb, y_train_comb)
print("Put all data into LinearR", np.mean([explained_variance_score(linear_regression.predict(X_test_pool[i]), y_test_pool[i]) for i in range(T)]))

lr_pool = []
for t in range(T):
    lr_pool.append(LinearRegression().fit(X_train_pool[t], y_train_pool[t]))
print("Put data into LinearR respectively", np.mean([explained_variance_score(lr_pool[i].predict(X_test_pool[i]), y_test_pool[i]) for i in range(T)]))

ella_logitic = ELLA(K=k, dim=d, mu=np.exp(-12), lam=np.exp(-10))
for t in range(T):
    ella_logitic.fit(X_train_pool[t], to_bin(y_train_pool[t]), t, LogisticModel())
print("ELLA Logit", np.mean([ella_logitic.score(X_test_pool[i], to_bin(y_test_pool[i]), i) for i in range(T)]))

logistic_regression = LogisticRegression().fit(X_train_comb, to_bin(y_train_comb))
print("Put all data into LogitR", np.mean([accuracy_score(logistic_regression.predict(X_test_pool[i]), to_bin(y_test_pool[i])) for i in range(T)]))

lr_pool = []
for t in range(T):
    lr_pool.append(LogisticRegression().fit(X_train_pool[t], to_bin(y_train_pool[t])))
print("Put data into LogitR respectively", np.mean([accuracy_score(lr_pool[i].predict(X_test_pool[i]), to_bin(y_test_pool[i])) for i in range(T)]))
