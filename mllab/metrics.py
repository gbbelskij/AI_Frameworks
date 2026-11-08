import json
import time

import numpy as np
import pandas as pd
from sklearn.metrics import (accuracy_score, f1_score, mean_absolute_error, mean_squared_error,
                             r2_score, roc_auc_score)
from sklearn.model_selection import cross_validate

from .data import RESULTS_DIR

CLASSIFICATION_METRICS = ["roc_auc", "accuracy", "f1"]
REGRESSION_METRICS = ["r2", "rmse", "mae"]


def classification_metrics(y_true, proba, threshold=0.5):
    pred = (proba >= threshold).astype(int)
    return {
        "roc_auc": roc_auc_score(y_true, proba),
        "accuracy": accuracy_score(y_true, pred),
        "f1": f1_score(y_true, pred),
    }


def regression_metrics(y_true, pred):
    return {
        "r2": r2_score(y_true, pred),
        "rmse": float(np.sqrt(mean_squared_error(y_true, pred))),
        "mae": mean_absolute_error(y_true, pred),
    }


def evaluate(model, X_train, y_train, X_test, y_test, task):
    start = time.perf_counter()
    model.fit(X_train, y_train)
    fit_time = time.perf_counter() - start
    start = time.perf_counter()
    if task == "classification":
        scores = classification_metrics(y_test, model.predict_proba(X_test)[:, 1])
    else:
        scores = regression_metrics(y_test, model.predict(X_test))
    predict_time = time.perf_counter() - start
    return {**scores, "fit_time": fit_time, "predict_time": predict_time}


class ResultsLog:
    def __init__(self, lab, algorithm):
        self.lab = lab
        self.algorithm = algorithm
        self.rows = []

    def add(self, task, variant, scores):
        row = {"lab": self.lab, "algorithm": self.algorithm, "task": task, "variant": variant}
        row.update({k: float(v) for k, v in scores.items()})
        self.rows = [r for r in self.rows if not (r["task"] == task and r["variant"] == variant)]
        self.rows.append(row)
        return row

    def table(self, task):
        cols = CLASSIFICATION_METRICS if task == "classification" else REGRESSION_METRICS
        frame = pd.DataFrame([r for r in self.rows if r["task"] == task])
        return frame.set_index("variant")[cols + ["fit_time", "predict_time"]]

    def plot(self):
        import matplotlib.pyplot as plt
        fig, axes = plt.subplots(1, 2, figsize=(13, 3.5))
        for ax, (task, metric) in zip(axes, [("classification", "roc_auc"), ("regression", "r2")]):
            values = self.table(task)[metric]
            values.plot.barh(ax=ax, color=["#9db4d6", "#3b6fb6", "#f2b880", "#d9772b"][:len(values)])
            ax.set_xlim(max(0.0, values.min() - 0.05), min(1.0, values.max() + 0.02))
            ax.set_title(f"{self.algorithm}: {task}, {metric} на test")
            ax.invert_yaxis()
            for i, v in enumerate(values):
                ax.text(v, i, f" {v:.4f}", va="center", fontsize=9)
        plt.tight_layout()
        return fig

    def save(self):
        RESULTS_DIR.mkdir(exist_ok=True)
        path = RESULTS_DIR / f"lab{self.lab}.json"
        path.write_text(json.dumps(self.rows, ensure_ascii=False, indent=2))
        return path


def load_all_results():
    rows = []
    for path in sorted(RESULTS_DIR.glob("lab*.json")):
        rows += json.loads(path.read_text())
    return pd.DataFrame(rows)


def cv_report(search, top=10):
    frame = pd.DataFrame(search.cv_results_)
    params = [c for c in frame.columns if c.startswith("param_")]
    frame = frame[params + ["mean_test_score", "std_test_score", "rank_test_score"]]
    frame.columns = [c.replace("param_", "") for c in frame.columns]
    return frame.sort_values("rank_test_score").head(top).reset_index(drop=True)


def compare_cv(candidates, X, y, cv, scoring, n_jobs=-1):
    rows = []
    for name, model in candidates.items():
        res = cross_validate(model, X, y, cv=cv, scoring=scoring, n_jobs=n_jobs, return_train_score=True)
        rows.append({
            "candidate": name,
            "cv_mean": res["test_score"].mean(),
            "cv_std": res["test_score"].std(),
            "train_mean": res["train_score"].mean(),
            "fit_time": res["fit_time"].mean(),
        })
    return pd.DataFrame(rows).set_index("candidate")
