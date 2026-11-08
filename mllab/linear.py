import numpy as np
from sklearn.base import BaseEstimator, ClassifierMixin, RegressorMixin, clone


def _sigmoid(z):
    return 0.5 * (1.0 + np.tanh(0.5 * z))


class LinearRegression(BaseEstimator, RegressorMixin):
    def __init__(self, alpha=0.0, fit_intercept=True):
        self.alpha = alpha
        self.fit_intercept = fit_intercept

    def fit(self, X, y):
        X = np.asarray(X, dtype=float)
        y = np.asarray(y, dtype=float)
        if self.fit_intercept:
            x_mean, y_mean = X.mean(axis=0), y.mean()
        else:
            x_mean, y_mean = np.zeros(X.shape[1]), 0.0
        Xc, yc = X - x_mean, y - y_mean
        gram = Xc.T @ Xc
        rhs = Xc.T @ yc
        if self.alpha > 0:
            self.coef_ = np.linalg.solve(gram + self.alpha * np.eye(X.shape[1]), rhs)
        else:
            self.coef_ = np.linalg.pinv(gram, hermitian=True) @ rhs
        self.intercept_ = y_mean - x_mean @ self.coef_
        self.n_features_in_ = X.shape[1]
        return self

    def predict(self, X):
        return np.asarray(X, dtype=float) @ self.coef_ + self.intercept_


class LogisticRegression(BaseEstimator, ClassifierMixin):
    def __init__(self, C=1.0, penalty="l2", fit_intercept=True, max_iter=100, tol=1e-6):
        self.C = C
        self.penalty = penalty
        self.fit_intercept = fit_intercept
        self.max_iter = max_iter
        self.tol = tol

    def _design(self, X):
        X = np.asarray(X, dtype=float)
        if self.fit_intercept:
            return np.hstack([np.ones((len(X), 1)), X])
        return X

    def _reg_mask(self, n_params):
        mask = np.ones(n_params)
        if self.fit_intercept:
            mask[0] = 0.0
        return mask

    def _objective(self, A, y, w, lam, mask):
        z = A @ w
        loss = np.mean(np.logaddexp(0.0, z) - y * z)
        if self.penalty == "l2":
            loss += 0.5 * lam * np.sum(mask * w ** 2)
        elif self.penalty == "l1":
            loss += lam * np.sum(mask * np.abs(w))
        return loss

    def _fit_newton(self, A, y, lam, mask):
        n, d = A.shape
        w = np.zeros(d)
        reg = lam * mask if self.penalty == "l2" else np.zeros(d)
        self.loss_curve_ = [self._objective(A, y, w, lam, mask)]
        for _ in range(self.max_iter):
            p = _sigmoid(A @ w)
            grad = A.T @ (p - y) / n + reg * w
            hess = (A * (p * (1 - p))[:, None]).T @ A / n + np.diag(reg + 1e-10)
            step = np.linalg.solve(hess, grad)
            t, current = 1.0, self.loss_curve_[-1]
            while t > 1e-8:
                candidate = w - t * step
                value = self._objective(A, y, candidate, lam, mask)
                if value <= current - 1e-4 * t * grad @ step:
                    break
                t *= 0.5
            w = candidate
            self.loss_curve_.append(value)
            if np.max(np.abs(t * step)) < self.tol:
                break
        return w

    def _fit_proximal(self, A, y, lam, mask):
        n, d = A.shape
        lipschitz = np.linalg.norm(A, 2) ** 2 / (4 * n)
        step = 1.0 / lipschitz
        w = np.zeros(d)
        v, t = w.copy(), 1.0
        self.loss_curve_ = [self._objective(A, y, w, lam, mask)]
        for _ in range(self.max_iter):
            grad = A.T @ (_sigmoid(A @ v) - y) / n
            u = v - step * grad
            w_new = np.where(mask > 0, np.sign(u) * np.maximum(np.abs(u) - step * lam, 0.0), u)
            t_new = (1 + np.sqrt(1 + 4 * t * t)) / 2
            v = w_new + (t - 1) / t_new * (w_new - w)
            converged = np.max(np.abs(w_new - w)) < self.tol
            w, t = w_new, t_new
            self.loss_curve_.append(self._objective(A, y, w, lam, mask))
            if converged:
                break
        return w

    def fit(self, X, y):
        self.classes_, encoded = np.unique(y, return_inverse=True)
        if len(self.classes_) != 2:
            raise ValueError("only binary classification is supported")
        A = self._design(X)
        y = encoded.astype(float)
        mask = self._reg_mask(A.shape[1])
        lam = 0.0 if self.penalty is None else 1.0 / (self.C * len(y))
        if self.penalty == "l1":
            w = self._fit_proximal(A, y, lam, mask)
        else:
            w = self._fit_newton(A, y, lam, mask)
        self.intercept_ = w[0] if self.fit_intercept else 0.0
        self.coef_ = w[1:] if self.fit_intercept else w
        self.n_features_in_ = len(self.coef_)
        return self

    def decision_function(self, X):
        return np.asarray(X, dtype=float) @ self.coef_ + self.intercept_

    def predict_proba(self, X):
        p = _sigmoid(self.decision_function(X))
        return np.column_stack([1 - p, p])

    def predict(self, X):
        return self.classes_[(self.decision_function(X) > 0).astype(int)]


class ClippedRegressor(BaseEstimator, RegressorMixin):
    def __init__(self, estimator, low=0.0, high=1.0):
        self.estimator = estimator
        self.low = low
        self.high = high

    def fit(self, X, y):
        self.estimator_ = clone(self.estimator).fit(X, y)
        return self

    def predict(self, X):
        return np.clip(self.estimator_.predict(X), self.low, self.high)
