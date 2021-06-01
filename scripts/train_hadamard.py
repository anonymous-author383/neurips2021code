"""Usage:
    train_hadamard.py GRAPHPATH EMBEDDINGPATH OUTPUTFILE
"""
from sys import argv
import sklearn
import emBench
import numpy as np
from sklearn.multiclass import OneVsRestClassifier
import pickle as pkl
import matplotlib.pyplot as plt

from sklearn.model_selection import ShuffleSplit
from sklearn.multiclass import OneVsRestClassifier
from sklearn.linear_model import LogisticRegression

title = argv[3]

def histofy(ps, num_bins):
    bins = np.zeros(num_bins, dtype=np.float32)
    for p in ps:
        bins[int( np.floor( p * (num_bins - 1)) )] += 1
    accumulator = 0
    for i in range(num_bins):
        accumulator += bins[-(i + 1)]
        bins[-(i + 1)] = accumulator / len(ps)
    print(bins)
    return bins


def predict_cv(X, y, train_ratio=0.2, n_splits=10, random_state=0, C=1.):
    micro, macro, precision = [], [], []
    shuffle = ShuffleSplit(n_splits=n_splits, test_size=1-train_ratio,
            random_state=random_state)
    for train_index, test_index in shuffle.split(X):
        p = []
        print(train_index.shape, test_index.shape)
        assert len(set(train_index) & set(test_index)) == 0
        assert len(train_index) + len(test_index) == X.shape[0]
        X_train, X_test = X[train_index], X[test_index]
        y_train, y_test = y[train_index], y[test_index]
        clf = OneVsRestClassifier(
                LogisticRegression(
                    C=C,
                    solver="liblinear",
                    multi_class="ovr"),
                n_jobs=-1)
        clf.fit(X_train, y_train)
        y_score = clf.predict_proba(X_test)
        y_pred = np.zeros(y_test.shape, dtype=bool)
        for v in range(len(y_test)):
            k = np.sum(y_test[v])
            pred_idxs = np.argsort(-y_score[v])
            y_pred[v, pred_idxs[:k]] = 1
            p.append(np.sum(y_pred[v] & y_test[v]) / k)
        mi = sklearn.metrics.f1_score(y_test, y_pred, average="micro")
        ma = sklearn.metrics.f1_score(y_test, y_pred, average="macro")
        micro.append(mi)
        macro.append(ma)
        precision.append(p)
    return micro, macro, precision

g = emBench.load(argv[1])
X = np.load(argv[2])
label_counts = np.sum(g._communities, axis=1)
test_verts = label_counts > 0
Y = g._communities[test_verts]
micro, macro, precision = predict_cv(X[test_verts], Y, train_ratio=0.8, n_splits=1)
print(micro)
print(macro)

num_bins = 10
fig, ax = plt.subplots()
for p in precision:
    bins = histofy(p, num_bins)
    ax.plot(np.arange(num_bins), bins)
ax.set_title('%s micro reliability curve' % (title))
fig.savefig('figures/%s_micro-reliability-curve.png' % (title))


