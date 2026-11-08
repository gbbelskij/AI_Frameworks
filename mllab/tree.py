import numpy as np
from sklearn.base import BaseEstimator, ClassifierMixin, RegressorMixin


class Binner:
    def __init__(self, max_bins=255):
        self.max_bins = max_bins

    def fit(self, X):
        X = np.asarray(X, dtype=float)
        self.thresholds_ = []
        for j in range(X.shape[1]):
            values = np.unique(X[:, j])
            if len(values) <= self.max_bins:
                thresholds = (values[:-1] + values[1:]) / 2
            else:
                quantiles = np.linspace(0, 1, self.max_bins + 1)[1:-1]
                thresholds = np.unique(np.quantile(X[:, j], quantiles))
            self.thresholds_.append(thresholds)
        n_bins = np.array([len(t) + 1 for t in self.thresholds_])
        self.offsets_ = np.concatenate([[0], np.cumsum(n_bins)[:-1]])
        self.total_bins_ = int(n_bins.sum())
        self.bin_feature_ = np.repeat(np.arange(len(n_bins)), n_bins)
        self.is_last_bin_ = np.zeros(self.total_bins_, dtype=bool)
        self.is_last_bin_[self.offsets_ + n_bins - 1] = True
        return self

    def transform(self, X):
        X = np.asarray(X, dtype=float)
        out = np.empty(X.shape, dtype=np.intp)
        for j, thresholds in enumerate(self.thresholds_):
            out[:, j] = np.searchsorted(thresholds, X[:, j], side="left") + self.offsets_[j]
        return out

    def threshold(self, global_bin):
        feature = self.bin_feature_[global_bin]
        return feature, self.thresholds_[feature][global_bin - self.offsets_[feature]]


def _resolve_max_features(max_features, n_features):
    if max_features is None:
        return n_features
    if max_features == "sqrt":
        return max(1, int(np.sqrt(n_features)))
    if max_features == "log2":
        return max(1, int(np.log2(n_features)))
    if isinstance(max_features, float):
        return max(1, int(max_features * n_features))
    return min(int(max_features), n_features)


def _xlogx(a):
    return a * np.log2(np.where(a > 0, a, 1.0))


class _BaseDecisionTree(BaseEstimator):
    _is_classifier = False

    def __init__(self, criterion, max_depth=None, min_samples_split=2, min_samples_leaf=1,
                 max_features=None, min_impurity_decrease=0.0, max_bins=255, random_state=None):
        self.criterion = criterion
        self.max_depth = max_depth
        self.min_samples_split = min_samples_split
        self.min_samples_leaf = min_samples_leaf
        self.max_features = max_features
        self.min_impurity_decrease = min_impurity_decrease
        self.max_bins = max_bins
        self.random_state = random_state

    def _cost(self, stats, counts):
        safe = np.maximum(counts, 1e-12)
        if self.criterion == "gini":
            return counts - (stats ** 2).sum(axis=-1) / safe
        if self.criterion == "entropy":
            return _xlogx(counts) - _xlogx(stats).sum(axis=-1)
        if self.criterion == "squared_error":
            return -(stats[..., 0] ** 2) / safe
        raise ValueError(f"unknown criterion: {self.criterion}")

    def _histogram(self, Xg, idx, features):
        block = Xg[idx] if features is None else Xg[np.ix_(idx, features)]
        width = block.shape[1]
        flat = block.ravel()
        total = self._binner.total_bins_
        if self._is_classifier:
            k = self.n_classes_
            codes = flat * k + np.repeat(self._target[idx], width)
            hist = np.bincount(codes, minlength=total * k).reshape(total, k).astype(float)
            counts = hist.sum(axis=1)
        else:
            sums = np.bincount(flat, weights=np.repeat(self._target[idx], width), minlength=total)
            counts = np.bincount(flat, minlength=total).astype(float)
            hist = sums[:, None]
        return hist, counts

    def _node_stats(self, idx):
        if self._is_classifier:
            stats = np.bincount(self._target[idx], minlength=self.n_classes_).astype(float)
        else:
            stats = np.array([self._target[idx].sum()])
        return stats, float(len(idx))

    def _best_split(self, Xg, idx, parent_stats, parent_n, features):
        hist, counts = self._histogram(Xg, idx, features)
        b = self._binner
        cum = np.cumsum(hist, axis=0)
        cum_n = np.cumsum(counts)
        base = np.vstack([np.zeros((1, hist.shape[1])), cum])[b.offsets_]
        base_n = np.concatenate([[0.0], cum_n])[b.offsets_]
        left = cum - base[b.bin_feature_]
        left_n = cum_n - base_n[b.bin_feature_]
        right = parent_stats[None, :] - left
        right_n = parent_n - left_n
        valid = (~b.is_last_bin_) & (left_n >= self.min_samples_leaf) & (right_n >= self.min_samples_leaf)
        if not valid.any():
            return None
        gain = self._cost(parent_stats, parent_n) - self._cost(left, left_n) - self._cost(right, right_n)
        gain = np.where(valid, gain, -np.inf)
        best = int(np.argmax(gain))
        return best, gain[best]

    def _leaf_value(self, stats, n):
        if self._is_classifier:
            return stats / n
        return stats[0] / n

    def _is_pure(self, idx):
        values = self._target[idx]
        return values.min() == values.max()

    def _fit_binned(self, Xg, target, binner, sample_idx=None):
        self._binner = binner
        self._target = target
        n_features = Xg.shape[1]
        self.n_features_in_ = n_features
        k_features = _resolve_max_features(self.max_features, n_features)
        rng = np.random.default_rng(self.random_state)
        max_depth = np.inf if self.max_depth is None else self.max_depth
        root_idx = np.arange(len(Xg)) if sample_idx is None else np.asarray(sample_idx)
        n_root = len(root_idx)

        feature, threshold, bin_threshold, left, right, value, n_node = [], [], [], [], [], [], []
        importances = np.zeros(n_features)

        def new_node(stats, n):
            feature.append(-1)
            threshold.append(np.nan)
            bin_threshold.append(-1)
            left.append(-1)
            right.append(-1)
            value.append(self._leaf_value(stats, n))
            n_node.append(n)
            return len(feature) - 1

        stats, n = self._node_stats(root_idx)
        stack = [(new_node(stats, n), root_idx, stats, n, 0)]
        depth_reached = 0
        while stack:
            node, idx, stats, n, depth = stack.pop()
            depth_reached = max(depth_reached, depth)
            if (depth >= max_depth or n < self.min_samples_split or n < 2 * self.min_samples_leaf
                    or self._is_pure(idx)):
                continue
            features = None
            if k_features < n_features:
                features = np.sort(rng.choice(n_features, k_features, replace=False))
            split = self._best_split(Xg, idx, stats, n, features)
            if split is None:
                continue
            global_bin, gain = split
            if gain <= 1e-12 or gain / n_root < self.min_impurity_decrease:
                continue
            f, thr = binner.threshold(global_bin)
            go_left = Xg[idx, f] <= global_bin
            left_idx, right_idx = idx[go_left], idx[~go_left]
            left_stats, left_n = self._node_stats(left_idx)
            right_stats, right_n = self._node_stats(right_idx)
            feature[node], threshold[node], bin_threshold[node] = f, thr, global_bin
            importances[f] += gain
            left[node] = new_node(left_stats, left_n)
            right[node] = new_node(right_stats, right_n)
            stack.append((right[node], right_idx, right_stats, right_n, depth + 1))
            stack.append((left[node], left_idx, left_stats, left_n, depth + 1))

        self.feature_ = np.array(feature, dtype=np.intp)
        self.threshold_ = np.array(threshold, dtype=float)
        self.bin_threshold_ = np.array(bin_threshold, dtype=np.intp)
        self.left_ = np.array(left, dtype=np.intp)
        self.right_ = np.array(right, dtype=np.intp)
        self.value_ = np.array(value, dtype=float)
        self.n_node_samples_ = np.array(n_node, dtype=float)
        self.depth_ = depth_reached
        self.n_leaves_ = int((self.feature_ < 0).sum())
        total = importances.sum()
        self.feature_importances_ = importances / total if total > 0 else importances
        del self._binner, self._target
        return self

    def _traverse(self, X, thresholds):
        node = np.zeros(len(X), dtype=np.intp)
        active = np.arange(len(X))
        while active.size:
            current = node[active]
            f = self.feature_[current]
            internal = f >= 0
            active, current, f = active[internal], current[internal], f[internal]
            if not active.size:
                break
            go_left = X[active, f] <= thresholds[current]
            node[active] = np.where(go_left, self.left_[current], self.right_[current])
        return node

    def apply(self, X):
        return self._traverse(np.asarray(X, dtype=float), self.threshold_)

    def _apply_binned(self, Xg):
        return self._traverse(Xg, self.bin_threshold_)


class DecisionTreeClassifier(_BaseDecisionTree, ClassifierMixin):
    _is_classifier = True

    def __init__(self, criterion="gini", max_depth=None, min_samples_split=2, min_samples_leaf=1,
                 max_features=None, min_impurity_decrease=0.0, max_bins=255, random_state=None):
        super().__init__(criterion, max_depth, min_samples_split, min_samples_leaf, max_features,
                         min_impurity_decrease, max_bins, random_state)

    def _prepare(self, y):
        self.classes_, encoded = np.unique(y, return_inverse=True)
        self.n_classes_ = len(self.classes_)
        return encoded.astype(np.intp)

    def fit(self, X, y):
        binner = Binner(self.max_bins).fit(X)
        return self._fit_binned(binner.transform(X), self._prepare(y), binner)

    def predict_proba(self, X):
        return self.value_[self.apply(X)]

    def predict(self, X):
        return self.classes_[np.argmax(self.predict_proba(X), axis=1)]


class DecisionTreeRegressor(_BaseDecisionTree, RegressorMixin):
    def __init__(self, criterion="squared_error", max_depth=None, min_samples_split=2,
                 min_samples_leaf=1, max_features=None, min_impurity_decrease=0.0, max_bins=255,
                 random_state=None):
        super().__init__(criterion, max_depth, min_samples_split, min_samples_leaf, max_features,
                         min_impurity_decrease, max_bins, random_state)

    def fit(self, X, y):
        binner = Binner(self.max_bins).fit(X)
        return self._fit_binned(binner.transform(X), np.asarray(y, dtype=float), binner)

    def predict(self, X):
        return self.value_[self.apply(X)]
