import numpy as np
from sklearn.base import BaseEstimator, ClassifierMixin, RegressorMixin

from .tree import Binner, DecisionTreeRegressor


def _sigmoid(z):
    return 0.5 * (1.0 + np.tanh(0.5 * z))


class _BaseGradientBoosting(BaseEstimator):
    def __init__(self, n_estimators=100, learning_rate=0.1, max_depth=3, min_samples_split=2,
                 min_samples_leaf=1, subsample=1.0, max_features=None, max_bins=255,
                 random_state=None):
        self.n_estimators = n_estimators
        self.learning_rate = learning_rate
        self.max_depth = max_depth
        self.min_samples_split = min_samples_split
        self.min_samples_leaf = min_samples_leaf
        self.subsample = subsample
        self.max_features = max_features
        self.max_bins = max_bins
        self.random_state = random_state

    def _init_score(self, y):
        raise NotImplementedError

    def _negative_gradient(self, y, raw):
        raise NotImplementedError

    def _loss(self, y, raw):
        raise NotImplementedError

    def _update_leaves(self, tree, leaves, y, raw, residual):
        pass

    def _boost(self, X, y):
        binner = Binner(self.max_bins).fit(X)
        Xg = binner.transform(X)
        rng = np.random.default_rng(self.random_state)
        n = len(y)
        self.init_ = self._init_score(y)
        raw = np.full(n, self.init_)
        self.estimators_ = []
        self.train_loss_ = []
        for _ in range(self.n_estimators):
            residual = self._negative_gradient(y, raw)
            sample_idx = None
            if self.subsample < 1.0:
                sample_idx = np.sort(rng.choice(n, max(1, int(self.subsample * n)), replace=False))
            tree = DecisionTreeRegressor(max_depth=self.max_depth,
                                         min_samples_split=self.min_samples_split,
                                         min_samples_leaf=self.min_samples_leaf,
                                         max_features=self.max_features, max_bins=self.max_bins,
                                         random_state=int(rng.integers(0, 2 ** 31 - 1)))
            tree._fit_binned(Xg, residual, binner, sample_idx)
            leaves = tree._apply_binned(Xg)
            fit_idx = np.arange(n) if sample_idx is None else sample_idx
            self._update_leaves(tree, leaves[fit_idx], y[fit_idx], raw[fit_idx], residual[fit_idx])
            raw += self.learning_rate * tree.value_[leaves]
            self.estimators_.append(tree)
            self.train_loss_.append(self._loss(y, raw))
        self.n_features_in_ = Xg.shape[1]
        importances = np.mean([t.feature_importances_ for t in self.estimators_], axis=0)
        self.feature_importances_ = importances / max(importances.sum(), 1e-12)
        return self

    def staged_raw_predict(self, X):
        X = np.asarray(X, dtype=float)
        raw = np.full(len(X), self.init_)
        for tree in self.estimators_:
            raw = raw + self.learning_rate * tree.predict(X)
            yield raw

    def _raw_predict(self, X):
        raw = None
        for raw in self.staged_raw_predict(X):
            pass
        return raw if raw is not None else np.full(len(X), self.init_)


class GradientBoostingRegressor(_BaseGradientBoosting, RegressorMixin):
    def _init_score(self, y):
        return float(np.mean(y))

    def _negative_gradient(self, y, raw):
        return y - raw

    def _loss(self, y, raw):
        return float(np.mean((y - raw) ** 2))

    def fit(self, X, y):
        return self._boost(X, np.asarray(y, dtype=float))

    def staged_predict(self, X):
        yield from self.staged_raw_predict(X)

    def predict(self, X):
        return self._raw_predict(X)


class GradientBoostingClassifier(_BaseGradientBoosting, ClassifierMixin):
    def _init_score(self, y):
        p = np.clip(np.mean(y), 1e-12, 1 - 1e-12)
        return float(np.log(p / (1 - p)))

    def _negative_gradient(self, y, raw):
        return y - _sigmoid(raw)

    def _loss(self, y, raw):
        return float(np.mean(np.logaddexp(0.0, raw) - y * raw))

    def _update_leaves(self, tree, leaves, y, raw, residual):
        p = _sigmoid(raw)
        size = len(tree.value_)
        numerator = np.bincount(leaves, weights=residual, minlength=size)
        denominator = np.bincount(leaves, weights=p * (1 - p), minlength=size)
        is_leaf = tree.feature_ < 0
        gamma = np.where(denominator > 1e-12, numerator / np.maximum(denominator, 1e-12), 0.0)
        tree.value_ = np.where(is_leaf, gamma, tree.value_)

    def fit(self, X, y):
        self.classes_, encoded = np.unique(y, return_inverse=True)
        if len(self.classes_) != 2:
            raise ValueError("only binary classification is supported")
        return self._boost(X, encoded.astype(float))

    def decision_function(self, X):
        return self._raw_predict(X)

    def staged_predict_proba(self, X):
        for raw in self.staged_raw_predict(X):
            p = _sigmoid(raw)
            yield np.column_stack([1 - p, p])

    def predict_proba(self, X):
        p = _sigmoid(self.decision_function(X))
        return np.column_stack([1 - p, p])

    def predict(self, X):
        return self.classes_[(self.decision_function(X) > 0).astype(int)]
