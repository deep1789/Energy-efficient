"""Leakage-aware evaluation on the UCI Energy Efficiency (ENB2012) dataset (v2).

Protocols (all nested: the inner tuning split matches the outer protocol):
  random   - repeated random 12-fold CV (the protocol used by most prior work)
  loso     - leave-one-building-shape-out (12 shapes -> 12 folds)
  shuffle  - 20 random splits holding out 3 whole shapes (train on 9)
Models include physics-informed engineered features and a Ridge->GBM residual hybrid.
Outputs go to results/ ; figures are made by src/make_figures.py. Fixed seed.
"""
import json, sys, time, warnings
import numpy as np, pandas as pd
from sklearn.base import BaseEstimator, RegressorMixin, clone
from sklearn.ensemble import GradientBoostingRegressor, RandomForestRegressor
from sklearn.linear_model import Ridge
from sklearn.metrics import mean_absolute_error, mean_squared_error, r2_score
from sklearn.model_selection import (GridSearchCV, GroupKFold, GroupShuffleSplit,
                                     KFold, LeaveOneGroupOut, RepeatedKFold)
from sklearn.neighbors import KNeighborsRegressor, NearestNeighbors
from sklearn.pipeline import make_pipeline
from sklearn.preprocessing import FunctionTransformer, StandardScaler
from sklearn.svm import SVR

warnings.filterwarnings("ignore")
SEED = 42
FEATS = ["X1", "X2", "X3", "X4", "X5", "X6", "X7", "X8"]
TARGETS = {"Y1": "Heating load", "Y2": "Cooling load"}

df = pd.read_csv("data/ENB2012_data.csv")
X = df[FEATS].values
shape_key = df[["X1", "X2", "X3", "X4", "X5"]].astype(str).agg("|".join, axis=1)
groups = pd.factorize(shape_key)[0]
G = len(set(groups))
print("rows", len(df), "shapes", G, "rows/shape", np.bincount(groups).tolist(), flush=True)


def phys(A):
    """Physics-informed features. Roof area ~ floor area; X7 = glazing/floor area."""
    x1, x2, x3, x4, x5, x6, x7, x8 = A.T
    vol = x4 * x5                      # volume proxy (floor area x height)
    glaz = x7 * x4                     # absolute glazing area
    return np.column_stack([x1, x2, x3, x4, x5, x7, x8, vol, glaz,
                            x2 / vol, x3 / vol, glaz / x3, glaz / x2,
                            *(x6 == k for k in (2, 3, 4, 5))])


class Hybrid(BaseEstimator, RegressorMixin):
    """Ridge captures the global (extrapolable) trend; GBM models the residual."""
    def __init__(self, alpha=1.0, n_estimators=200, learning_rate=0.05, max_depth=2):
        self.alpha, self.n_estimators, self.learning_rate, self.max_depth = alpha, n_estimators, learning_rate, max_depth
    def fit(self, X, y):
        self.sc_ = StandardScaler().fit(X)
        Z = self.sc_.transform(X)
        self.lin_ = Ridge(alpha=self.alpha).fit(Z, y)
        self.gbm_ = GradientBoostingRegressor(n_estimators=self.n_estimators, learning_rate=self.learning_rate,
                                              max_depth=self.max_depth, random_state=SEED).fit(X, y - self.lin_.predict(Z))
        return self
    def predict(self, X):
        return self.lin_.predict(self.sc_.transform(X)) + self.gbm_.predict(X)


PH = lambda: FunctionTransformer(phys)
MODELS = {
    "Ridge": (make_pipeline(StandardScaler(), Ridge()), {"ridge__alpha": [0.1, 1, 10, 100]}),
    "kNN": (make_pipeline(StandardScaler(), KNeighborsRegressor()), {"kneighborsregressor__n_neighbors": [3, 9, 15]}),
    "SVR (RBF)": (make_pipeline(StandardScaler(), SVR()), {"svr__C": [1, 30], "svr__gamma": ["scale", 0.02]}),
    "Random forest": (RandomForestRegressor(n_estimators=100, random_state=SEED, n_jobs=4),
                      {"max_features": [0.5, 1.0], "min_samples_leaf": [1, 5]}),
    "Gradient boosting": (GradientBoostingRegressor(random_state=SEED, n_estimators=200),
                          {"learning_rate": [0.05, 0.1], "max_depth": [2, 3]}),
    "Ridge + phys": (make_pipeline(PH(), StandardScaler(), Ridge()), {"ridge__alpha": [0.1, 1, 10, 100]}),
    "GBM + phys": (make_pipeline(PH(), GradientBoostingRegressor(random_state=SEED, n_estimators=200)),
                   {"gradientboostingregressor__learning_rate": [0.05, 0.1], "gradientboostingregressor__max_depth": [2, 3]}),
    "Hybrid (Ridge->GBM) + phys": (make_pipeline(PH(), Hybrid()),
                                   {"hybrid__alpha": [1, 10, 100], "hybrid__max_depth": [1, 2]}),
}


def make_splits(protocol):
    if protocol == "random":
        return list(RepeatedKFold(n_splits=12, n_repeats=2, random_state=SEED).split(X)), False
    if protocol == "loso":
        return list(LeaveOneGroupOut().split(X, groups=groups)), True
    if protocol == "shuffle":
        return list(GroupShuffleSplit(n_splits=20, test_size=3, random_state=SEED).split(X, groups=groups)), True


def inner_cv(tr, grouped):
    if not grouped:
        return KFold(3, shuffle=True, random_state=SEED)
    g = groups[tr]                      # inner folds also hold out whole shapes
    return list(GroupKFold(n_splits=3).split(tr, groups=g))


def metrics(y, p):
    return dict(RMSE=mean_squared_error(y, p) ** 0.5, MAE=mean_absolute_error(y, p), R2=r2_score(y, p))


rows, oof_rows = [], []
scaler_all = StandardScaler().fit(X)
Z = scaler_all.transform(X)
diag = []
for protocol in ("random", "loso", "shuffle"):
    splits, grouped = make_splits(protocol)
    # leakage diagnostics: distance from each test row to nearest training row
    for k, (tr, te) in enumerate(splits):
        d, _ = NearestNeighbors(n_neighbors=1).fit(Z[tr]).kneighbors(Z[te])
        same = np.isin(groups[te], groups[tr]).mean()
        diag.append(dict(protocol=protocol, fold=k, nn_dist_mean=d.mean(), nn_dist_median=float(np.median(d)),
                         frac_test_shape_in_train=same, n_train=len(tr), n_test=len(te)))
    for t in TARGETS:
        y = df[t].values
        for name, (est, grid) in MODELS.items():
            t0 = time.time()
            for k, (tr, te) in enumerate(splits):
                gs = GridSearchCV(clone(est), grid, cv=inner_cv(tr, grouped), scoring="neg_root_mean_squared_error", n_jobs=1)
                gs.fit(X[tr], y[tr])
                p = gs.predict(X[te])
                rows.append(dict(protocol=protocol, target=t, model=name, fold=k, **metrics(y[te], p)))
                if protocol == "loso":
                    oof_rows.append(pd.DataFrame(dict(target=t, model=name, idx=te, shape=groups[te], y=y[te], pred=p)))
            print(f"{protocol:8s} {t} {name:28s} {time.time()-t0:5.0f}s", flush=True)
        pd.DataFrame(rows).to_csv("results/folds_all.csv", index=False)   # incremental save
pd.DataFrame(diag).to_csv("results/leakage_diagnostics.csv", index=False)
pd.concat(oof_rows).to_csv("results/loso_oof_predictions.csv", index=False)
r = pd.DataFrame(rows); r.to_csv("results/folds_all.csv", index=False)
s = r.groupby(["target", "protocol", "model"])[["RMSE", "MAE", "R2"]].agg(["mean", "std"])
s.columns = [f"{a}_{b}" for a, b in s.columns]; s.to_csv("results/summary_all.csv")
print("done", flush=True)
