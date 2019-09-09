import warnings
warnings.filterwarnings('ignore')
from ELLA import ELLA
import numpy as np
from scipy.io import loadmat
from sklearn.model_selection import train_test_split
from Base_Learner.LogisticLeaner import LogisticModel, LogisticRegression
from sklearn.metrics import accuracy_score, roc_auc_score

data = loadmat('landminedata.mat')

Ts = 0
T = data["label"].shape[1]
k = 2
d = data["feature"][0, 0].shape[1] + 1

X_train_comb, y_train_comb = np.zeros((0, d)), np.zeros((0, 1))
X_test_pool, y_test_pool = [], []
X_train_pool, y_train_pool = [], []

for t in np.random.permutation(range(T)):
    X_ori = data["feature"][0, t]
    X, y = np.hstack((X_ori, np.ones((X_ori.shape[0], 1)))), (data["label"][0, t] > 0.99)

    X_set, y_set = [], []
    X_drop, y_drop = [], []
    cnt0, cnt1 = 0, 0
    for i in range(len(X_ori)):
        if y[i]:
            X_set.append(X[i])
            y_set.append(y[i])
            cnt1 = cnt1 + 1
        if not y[i] and cnt0<=cnt1:
            X_set.append(X[i])
            y_set.append(y[i])
            cnt0 = cnt0 + 1
        else:
            X_drop.append(X[i])
            y_drop.append(y[i])
    X_set, y_set = np.array(X_set), np.array(y_set)

    X_train, X_test, y_train, y_test = train_test_split(X_set, y_set, test_size=0.5, random_state=19260817)

    X_train_pool.append(X_train)
    y_train_pool.append(y_train)
    X_test_pool.append(np.vstack((X_test, X_drop)))
    y_test_pool.append(np.vstack((y_test, y_drop)))
    # X_test_pool.append(X_test)
    # y_test_pool.append(y_test)

    X_train_comb = np.vstack((X_train_comb, X_train))
    y_train_comb = np.vstack((y_train_comb, y_train))


ella = ELLA(K=k, dim=d, mu=np.exp(-10), lam=np.exp(-12))
for t in range(T):
    ella.fit(X_train_pool[t], y_train_pool[t], t, LogisticModel())
    if t>Ts: print("Task 1's ACC = ", ella.score(X_test_pool[Ts], y_test_pool[Ts], Ts))
    # print(ella.score(X_test_pool[t], y_test_pool[t], t))

print("ELLA Linear", np.mean([ella.score(X_test_pool[i], y_test_pool[i], i) for i in range(T)]))

logistic_regression = LogisticRegression().fit(X_train_comb, y_train_comb)
print("Put all data into LogitR", np.mean([accuracy_score(y_pred=logistic_regression.predict(X_test_pool[i]), y_true=y_test_pool[i]) for i in range(T)]))

lr_pool = []
for t in range(T):
    lr_pool.append(LogisticRegression().fit(X_train_pool[t], y_train_pool[t]))
print("Put data into LogitR respectively", np.mean([accuracy_score(y_pred=lr_pool[i].predict(X_test_pool[i]), y_true=y_test_pool[i]) for i in range(T)]))