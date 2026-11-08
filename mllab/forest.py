import numpy as np
from joblib import Parallel, delayed
from sklearn.base import BaseEstimator, ClassifierMixin, RegressorMixin

from .tree import Binner, DecisionTreeClassifier, DecisionTreeRegressor


def _grow(tree, Xg, target, binner, seed, bootstrap, max_samples, classes):
    rng = np.random.default_rng(seed)
    n = len(Xg)
    size = n if max_samples is None else max(1, int(max_samples * n))
    sample_idx = rng.integers(0, n, size) if bootstrap else None
    if classes is not None:
        tree.classes_, tree.n_classes_ = classes, len(classes)
    return tree._fit_binned(Xg, target, binner, sample_idx)


class _BaseForest(BaseEstimator):
    _tree_class = None

    def __init__(self, n_estimators=100, criterion=None, max_depth=None, min_samples_split=2,
                 min_samples_leaf=1, max_features="sqrt", bootstrap=True, max_samples=None,
                 max_bins=255, n_jobs=None, random_state=None):
        self.n_estimators = n_estimators
        self.criterion = criterion
        self.max_depth = max_depth
        self.min_samples_split = min_samples_split
        self.min_samples_leaf = min_samples_leaf
        self.max_features = max_features
        self.bootstrap = bootstrap
        self.max_samples = max_samples
        self.max_bins = max_bins
        self.n_jobs = n_jobs
        self.random_state = random_state

    def _make_tree(self, seed):
        params = dict(max_depth=self.max_depth, min_samples_split=self.min_samples_split,
                      min_samples_leaf=self.min_samples_leaf, max_features=self.max_features,
                      max_bins=self.max_bins, random_state=seed)
        if self.criterion is not None:
            params["criterion"] = self.criterion
        return self._tree_class(**params)

    def _fit_forest(self, X, target, classes=None):
        binner = Binner(self.max_bins).fit(X)
        Xg = binner.transform(X)
        seeds = np.random.default_rng(self.random_state).integers(0, 2 ** 31 - 1, self.n_estimators)
        self.estimators_ = Parallel(n_jobs=self.n_jobs)(
            delayed(_grow)(self._make_tree(int(s)), Xg, target, binner, int(s), self.bootstrap,
                           self.max_samples, classes)
            for s in seeds
        )
        self.n_features_in_ = Xg.shape[1]
        self.feature_importances_ = np.mean([t.feature_importances_ for t in self.estimators_], axis=0)
        return self


class RandomForestClassifier(_BaseForest, ClassifierMixin):
    _tree_class = DecisionTreeClassifier

    def fit(self, X, y):
        self.classes_, encoded = np.unique(y, return_inverse=True)
        self.n_classes_ = len(self.classes_)
        return self._fit_forest(X, encoded.astype(np.intp), self.classes_)

    def predict_proba(self, X):
        X = np.asarray(X, dtype=float)
        return np.mean([tree.predict_proba(X) for tree in self.estimators_], axis=0)

    def predict(self, X):
        return self.classes_[np.argmax(self.predict_proba(X), axis=1)]


class RandomForestRegressor(_BaseForest, RegressorMixin):
    _tree_class = DecisionTreeRegressor

    def __init__(self, n_estimators=100, criterion=None, max_depth=None, min_samples_split=2,
                 min_samples_leaf=1, max_features=1.0, bootstrap=True, max_samples=None,
                 max_bins=255, n_jobs=None, random_state=None):
        super().__init__(n_estimators, criterion, max_depth, min_samples_split, min_samples_leaf,
                         max_features, bootstrap, max_samples, max_bins, n_jobs, random_state)

    def fit(self, X, y):
        return self._fit_forest(X, np.asarray(y, dtype=float))

    def predict(self, X):
        X = np.asarray(X, dtype=float)
        return np.mean([tree.predict(X) for tree in self.estimators_], axis=0)
