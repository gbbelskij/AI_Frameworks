import numpy as np
import pandas as pd
from sklearn.base import BaseEstimator, TransformerMixin, clone
from sklearn.ensemble import RandomForestRegressor
from sklearn.model_selection import KFold

from .data import CHESS_FEATURES, DOTA_HEROES, DOTA_META, RANDOM_STATE


def _hero_stats(team1, team2, y, smoothing):
    y = y.astype(float)
    games = team1.sum(0) + team2.sum(0)
    wins = team1.T @ y + team2.T @ (1 - y)
    hero_wr = (wins + 0.5 * smoothing) / (games + smoothing)

    pair_games = team1.T @ team1 + team2.T @ team2
    pair_wins = (team1 * y[:, None]).T @ team1 + (team2 * (1 - y)[:, None]).T @ team2
    pair_wr = (pair_wins + 0.5 * smoothing) / (pair_games + smoothing)
    expected_pair = (hero_wr[:, None] + hero_wr[None, :]) / 2
    synergy = pair_wr - expected_pair
    np.fill_diagonal(synergy, 0.0)

    vs_games = team1.T @ team2 + team2.T @ team1
    vs_wins = (team1 * y[:, None]).T @ team2 + (team2 * (1 - y)[:, None]).T @ team1
    vs_wr = (vs_wins + 0.5 * smoothing) / (vs_games + smoothing)
    counter = vs_wr - (0.5 + (hero_wr[:, None] - hero_wr[None, :]) / 2)
    np.fill_diagonal(counter, 0.0)
    return hero_wr, synergy, counter


def _hero_scores(team1, team2, stats, synergy_features):
    hero_wr, synergy, counter = stats
    centered = hero_wr - 0.5
    wr1 = team1 @ centered
    wr2 = team2 @ centered
    cols = [wr1, wr2, wr1 - wr2]
    if synergy_features:
        syn1 = ((team1 @ synergy) * team1).sum(1) / 2
        syn2 = ((team2 @ synergy) * team2).sum(1) / 2
        cnt = ((team1 @ counter) * team2).sum(1)
        cols += [syn1, syn2, syn1 - syn2, cnt]
    return np.column_stack(cols)


class DotaFeatures(BaseEstimator, TransformerMixin):
    def __init__(self, meta="raw", heroes="pm1", drop_constant=False, winrate=False,
                 synergy=False, smoothing=20.0, n_folds=5, random_state=RANDOM_STATE):
        self.meta = meta
        self.heroes = heroes
        self.drop_constant = drop_constant
        self.winrate = winrate
        self.synergy = synergy
        self.smoothing = smoothing
        self.n_folds = n_folds
        self.random_state = random_state

    @staticmethod
    def _frame(X):
        return X if isinstance(X, pd.DataFrame) else pd.DataFrame(X, columns=DOTA_META + DOTA_HEROES)

    def _split(self, X):
        return X[DOTA_META], X[self.hero_columns_].to_numpy(dtype=float)

    @staticmethod
    def _teams(X):
        heroes = X[DOTA_HEROES].to_numpy(dtype=float)
        return (heroes == 1).astype(float), (heroes == -1).astype(float)

    def fit(self, X, y=None):
        X = self._frame(X)
        if self.drop_constant:
            self.hero_columns_ = [c for c in DOTA_HEROES if X[c].nunique() > 1]
        else:
            self.hero_columns_ = list(DOTA_HEROES)
        if self.meta == "onehot":
            self.categories_ = {c: sorted(X[c].unique()) for c in DOTA_META}
        elif self.meta == "region":
            self.categories_ = {
                "region": sorted((X["cluster_id"] // 10).unique()),
                "game_mode": sorted(X["game_mode"].unique()),
                "game_type": sorted(X["game_type"].unique()),
            }
        if self.winrate or self.synergy:
            if y is None:
                raise ValueError("winrate/synergy features require y")
            t1, t2 = self._teams(X)
            self.stats_ = _hero_stats(t1, t2, np.asarray(y), self.smoothing)
        return self

    def _meta_block(self, meta):
        if self.meta == "drop":
            return np.empty((len(meta), 0)), []
        if self.meta == "raw":
            return meta.to_numpy(dtype=float), list(DOTA_META)
        frame = meta
        if self.meta == "region":
            frame = pd.DataFrame({
                "region": meta["cluster_id"] // 10,
                "game_mode": meta["game_mode"],
                "game_type": meta["game_type"],
            })
        blocks, names = [], []
        for col, cats in self.categories_.items():
            values = frame[col].to_numpy()
            blocks.append(np.column_stack([(values == c).astype(float) for c in cats]))
            names += [f"{col}={c}" for c in cats]
        return np.hstack(blocks), names

    def _hero_block(self, hero):
        if self.heroes == "none":
            return np.empty((len(hero), 0)), []
        if self.heroes == "pm1":
            return hero, list(self.hero_columns_)
        t1 = (hero == 1).astype(float)
        t2 = (hero == -1).astype(float)
        names = [f"t1_{c}" for c in self.hero_columns_] + [f"t2_{c}" for c in self.hero_columns_]
        return np.hstack([t1, t2]), names

    def _score_names(self):
        names = ["wr_team1", "wr_team2", "wr_diff"]
        if self.synergy:
            names += ["syn_team1", "syn_team2", "syn_diff", "counter"]
        return names

    def _assemble(self, X, scores):
        meta, hero = self._split(X)
        meta_block, meta_names = self._meta_block(meta)
        hero_block, hero_names = self._hero_block(hero)
        blocks = [meta_block, hero_block]
        self.feature_names_ = meta_names + hero_names
        if scores is not None:
            blocks.append(scores)
            self.feature_names_ = self.feature_names_ + self._score_names()
        return np.hstack(blocks)

    def transform(self, X):
        X = self._frame(X)
        scores = None
        if self.winrate or self.synergy:
            t1, t2 = self._teams(X)
            scores = _hero_scores(t1, t2, self.stats_, self.synergy)
        return self._assemble(X, scores)

    def fit_transform(self, X, y=None, **fit_params):
        X = self._frame(X)
        self.fit(X, y)
        if not (self.winrate or self.synergy):
            return self._assemble(X, None)
        y = np.asarray(y)
        t1, t2 = self._teams(X)
        width = 7 if self.synergy else 3
        scores = np.zeros((len(X), width))
        folds = KFold(self.n_folds, shuffle=True, random_state=self.random_state)
        for fit_idx, oof_idx in folds.split(X):
            stats = _hero_stats(t1[fit_idx], t2[fit_idx], y[fit_idx], self.smoothing)
            scores[oof_idx] = _hero_scores(t1[oof_idx], t2[oof_idx], stats, self.synergy)
        return self._assemble(X, scores)

    def get_feature_names_out(self, input_features=None):
        return np.array(self.feature_names_)


class ChessFeatures(BaseEstimator, TransformerMixin):
    def __init__(self, columns=None, interactions=False):
        self.columns = columns
        self.interactions = interactions

    def fit(self, X, y=None):
        X = X if isinstance(X, pd.DataFrame) else pd.DataFrame(X, columns=CHESS_FEATURES)
        self.levels_ = {c: sorted(X[c].unique()) for c in X.columns}
        self.selected_ = list(self.columns) if self.columns is not None else list(X.columns)
        return self

    def transform(self, X):
        X = X if isinstance(X, pd.DataFrame) else pd.DataFrame(X, columns=CHESS_FEATURES)
        blocks, names = [], []
        for col in self.selected_:
            levels = self.levels_[col]
            values = X[col].to_numpy()
            if len(levels) == 2:
                blocks.append((values == levels[1]).astype(float)[:, None])
                names.append(col)
            else:
                blocks.append(np.column_stack([(values == v).astype(float) for v in levels]))
                names += [f"{col}={v}" for v in levels]
        out = np.hstack(blocks)
        if self.interactions:
            i, j = np.triu_indices(out.shape[1], k=1)
            out = np.hstack([out, out[:, i] * out[:, j]])
            names = names + [f"{names[a]}*{names[b]}" for a, b in zip(i, j)]
        self.feature_names_ = names
        return out

    def get_feature_names_out(self, input_features=None):
        return np.array(self.feature_names_)


class ImportanceWeighter(BaseEstimator, TransformerMixin):
    def __init__(self, estimator=None, power=1.0):
        self.estimator = estimator
        self.power = power

    def fit(self, X, y):
        if self.estimator is None:
            estimator = RandomForestRegressor(n_estimators=200, random_state=RANDOM_STATE, n_jobs=-1)
        else:
            estimator = clone(self.estimator)
        importances = np.asarray(estimator.fit(X, y).feature_importances_, dtype=float)
        self.weights_ = (importances / importances.max()) ** self.power
        return self

    def transform(self, X):
        return np.asarray(X, dtype=float) * self.weights_
