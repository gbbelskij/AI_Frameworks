import numpy as np
from sklearn.base import BaseEstimator, ClassifierMixin, RegressorMixin


def _pairwise_distances(A, B, metric, B_sq=None):
    if metric == "euclidean":
        A_sq = np.einsum("ij,ij->i", A, A)
        if B_sq is None:
            B_sq = np.einsum("ij,ij->i", B, B)
        sq = A_sq[:, None] - 2.0 * (A @ B.T) + B_sq[None, :]
        return np.sqrt(np.maximum(sq, 0.0))
    if metric == "manhattan":
        out = np.zeros((A.shape[0], B.shape[0]))
        for j in range(A.shape[1]):
            out += np.abs(A[:, j, None] - B[None, :, j])
        return out
    if metric == "cosine":
        A_norm = A / np.maximum(np.linalg.norm(A, axis=1, keepdims=True), 1e-12)
        B_norm = B / np.maximum(np.linalg.norm(B, axis=1, keepdims=True), 1e-12)
        return np.maximum(1.0 - A_norm @ B_norm.T, 0.0)
    raise ValueError(f"unknown metric: {metric}")


class _KNNBase(BaseEstimator):
    def __init__(self, n_neighbors=5, weights="uniform", metric="euclidean", batch_size=512):
        self.n_neighbors = n_neighbors
        self.weights = weights
        self.metric = metric
        self.batch_size = batch_size

    def _store(self, X, y):
        self.X_ = np.asarray(X, dtype=float)
        self.y_ = np.asarray(y)
        self.X_sq_ = np.einsum("ij,ij->i", self.X_, self.X_)
        self.n_features_in_ = self.X_.shape[1]

    def kneighbors(self, X):
        X = np.asarray(X, dtype=float)
        k = min(self.n_neighbors, len(self.X_))
        dist = np.empty((len(X), k))
        idx = np.empty((len(X), k), dtype=int)
        for start in range(0, len(X), self.batch_size):
            stop = start + self.batch_size
            D = _pairwise_distances(X[start:stop], self.X_, self.metric, self.X_sq_)
            part = np.argpartition(D, k - 1, axis=1)[:, :k]
            part_d = np.take_along_axis(D, part, axis=1)
            order = np.argsort(part_d, axis=1, kind="stable")
            idx[start:stop] = np.take_along_axis(part, order, axis=1)
            dist[start:stop] = np.take_along_axis(part_d, order, axis=1)
        return dist, idx

    def _neighbor_weights(self, dist):
        if self.weights == "uniform":
            return np.ones_like(dist)
        if self.weights == "distance":
            zero = dist <= 1e-12
            with np.errstate(divide="ignore"):
                w = 1.0 / dist
            has_zero = zero.any(axis=1)
            w[has_zero] = zero[has_zero].astype(float)
            return w
        raise ValueError(f"unknown weights: {self.weights}")


class KNNClassifier(_KNNBase, ClassifierMixin):
    def fit(self, X, y):
        self.classes_, encoded = np.unique(y, return_inverse=True)
        self._store(X, encoded)
        return self

    def predict_proba(self, X):
        dist, idx = self.kneighbors(X)
        w = self._neighbor_weights(dist)
        labels = self.y_[idx]
        proba = np.zeros((len(idx), len(self.classes_)))
        for c in range(len(self.classes_)):
            proba[:, c] = (w * (labels == c)).sum(axis=1)
        return proba / proba.sum(axis=1, keepdims=True)

    def predict(self, X):
        return self.classes_[np.argmax(self.predict_proba(X), axis=1)]


class KNNRegressor(_KNNBase, RegressorMixin):
    def fit(self, X, y):
        self._store(X, np.asarray(y, dtype=float))
        return self

    def predict(self, X):
        dist, idx = self.kneighbors(X)
        w = self._neighbor_weights(dist)
        return (w * self.y_[idx]).sum(axis=1) / w.sum(axis=1)
