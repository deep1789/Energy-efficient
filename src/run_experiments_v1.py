"""Experiments on the UCI Energy Efficiency (ENB2012) dataset.

Random-split nested CV and building-shape-grouped CV for heating (Y1) and cooling (Y2) load.
Writes CSV/JSON results to results/ and figures to figures/. Seed is fixed.
"""
import json, warnings
import numpy as np, pandas as pd
import matplotlib; matplotlib.use("Agg")
import matplotlib.pyplot as plt
from scipy.stats import wilcoxon
from sklearn.base import clone
from sklearn.ensemble import GradientBoostingRegressor, RandomForestRegressor
from sklearn.inspection import permutation_importance
from sklearn.linear_model import Ridge
from sklearn.metrics import mean_absolute_error, mean_squared_error, r2_score
from sklearn.model_selection import (GridSearchCV, GroupKFold, KFold,
                                     RepeatedKFold)
from sklearn.neighbors import KNeighborsRegressor
from sklearn.pipeline import make_pipeline
from sklearn.preprocessing import StandardScaler
from sklearn.svm import SVR
from sklearn.tree import DecisionTreeRegressor

warnings.filterwarnings("ignore")
SEED = 42
FEATS = ["X1", "X2", "X3", "X4", "X5", "X6", "X7", "X8"]
NAMES = {"X1": "Relative compactness", "X2": "Surface area", "X3": "Wall area",
         "X4": "Roof area", "X5": "Overall height", "X6": "Orientation",
         "X7": "Glazing area", "X8": "Glazing distribution"}
TARGETS = {"Y1": "Heating load", "Y2": "Cooling load"}

df = pd.read_csv("data/ENB2012_data.csv")
X = df[FEATS].values
# building shape = combination of the geometry attributes X1-X5 (12 distinct shapes)
groups = pd.factorize(df[["X1", "X2", "X3", "X4", "X5"]].astype(str).agg("|".join, axis=1))[0]
print("rows", len(df), "shapes", len(set(groups)))

# name -> (estimator, param grid)
MODELS = {
    "Ridge": (make_pipeline(StandardScaler(), Ridge()), {"ridge__alpha": [0.01, 0.1, 1, 10]}),
    "kNN": (make_pipeline(StandardScaler(), KNeighborsRegressor()), {"kneighborsregressor__n_neighbors": [3, 5, 9]}),
    "SVR (RBF)": (make_pipeline(StandardScaler(), SVR()), {"svr__C": [1, 10, 100], "svr__gamma": ["scale", 0.05]}),
    "Decision tree": (DecisionTreeRegressor(random_state=SEED), {"max_depth": [4, 8, None]}),
    "Random forest": (RandomForestRegressor(n_estimators=200, random_state=SEED, n_jobs=-1),
                      {"max_features": [0.5, 1.0], "min_samples_leaf": [1, 3]}),
    "Gradient boosting": (GradientBoostingRegressor(random_state=SEED),
                          {"n_estimators": [200, 500], "learning_rate": [0.05, 0.1], "max_depth": [2, 3]}),
}

def metrics(y, p):
    return dict(RMSE=mean_squared_error(y, p) ** 0.5, MAE=mean_absolute_error(y, p), R2=r2_score(y, p))

def evaluate(splitter, y, split_groups=None):
    """Nested CV: inner 3-fold grid search on each outer training fold."""
    rows = []
    for name, (est, grid) in MODELS.items():
        for k, (tr, te) in enumerate(splitter.split(X, y, split_groups)):
            gs = GridSearchCV(clone(est), grid, cv=KFold(3, shuffle=True, random_state=SEED),
                              scoring="neg_root_mean_squared_error")
            gs.fit(X[tr], y[tr])
            rows.append(dict(model=name, fold=k, **metrics(y[te], gs.predict(X[te]))))
    return pd.DataFrame(rows)

def summarize(r):
    g = r.groupby("model")[["RMSE", "MAE", "R2"]].agg(["mean", "std"])
    g.columns = [f"{a}_{b}" for a, b in g.columns]
    return g.sort_values("RMSE_mean")

out, tests = {}, []
for t, label in TARGETS.items():
    y = df[t].values
    # 1) random 10-fold, repeated 3x (nested) — the protocol used in most prior work
    rand = evaluate(RepeatedKFold(n_splits=10, n_repeats=3, random_state=SEED), y)
    # 2) shape-grouped CV: whole building shapes held out (12 shapes -> 6 folds)
    grp = evaluate(GroupKFold(n_splits=6), y, groups)
    for tag, r in (("random", rand), ("grouped", grp)):
        r.to_csv(f"results/folds_{t}_{tag}.csv", index=False)
        s = summarize(r); s.to_csv(f"results/summary_{t}_{tag}.csv")
        out[f"{t}_{tag}"] = s.round(4).to_dict("index")
        print(f"\n=== {label} — {tag} CV ===\n", s.round(3))
    # paired Wilcoxon: best vs. others on random CV RMSE
    best = summarize(rand).index[0]
    b = rand[rand.model == best].sort_values("fold").RMSE.values
    for m in MODELS:
        if m != best:
            o = rand[rand.model == m].sort_values("fold").RMSE.values
            tests.append(dict(target=t, best=best, other=m, p=wilcoxon(b, o).pvalue))

pd.DataFrame(tests).to_csv("results/wilcoxon_random_cv.csv", index=False)
json.dump(out, open("results/summary_all.json", "w"), indent=1)

# ---- permutation importance (gradient boosting, held-out 25% by shape) & figures ----
fig, ax = plt.subplots(1, 2, figsize=(10, 4), sharey=True)
imp_rows = []
for a, (t, label) in zip(ax, TARGETS.items()):
    y = df[t].values
    te_shapes = set(np.random.RandomState(SEED).permutation(sorted(set(groups)))[:3])
    te = np.array([g in te_shapes for g in groups])
    m = clone(MODELS["Gradient boosting"][0]).set_params(n_estimators=500, learning_rate=0.05, max_depth=3).fit(X[~te], y[~te])
    pi = permutation_importance(m, X[te], y[te], n_repeats=30, random_state=SEED, scoring="neg_root_mean_squared_error")
    order = np.argsort(pi.importances_mean)
    a.barh([NAMES[FEATS[i]] for i in order], pi.importances_mean[order], xerr=pi.importances_std[order], color="#3b6ea5")
    a.set_title(label); a.set_xlabel("Increase in RMSE when permuted (kWh/m²)")
    for i in range(len(FEATS)):
        imp_rows.append(dict(target=t, feature=NAMES[FEATS[i]], mean=pi.importances_mean[i], std=pi.importances_std[i]))
plt.tight_layout(); plt.savefig("figures/permutation_importance.png", dpi=200); plt.close()
pd.DataFrame(imp_rows).to_csv("results/permutation_importance.csv", index=False)

# correlation heatmap
c = df.corr(); fig, a = plt.subplots(figsize=(6, 5))
im = a.imshow(c, cmap="RdBu_r", vmin=-1, vmax=1); a.set_xticks(range(10)); a.set_yticks(range(10))
a.set_xticklabels(c.columns); a.set_yticklabels(c.columns); plt.colorbar(im)
for i in range(10):
    for j in range(10):
        a.text(j, i, f"{c.values[i, j]:.1f}", ha="center", va="center", fontsize=6)
plt.tight_layout(); plt.savefig("figures/correlation.png", dpi=200); plt.close()

# random vs grouped comparison (RMSE)
fig, ax = plt.subplots(1, 2, figsize=(10, 4), sharey=False)
for a, t in zip(ax, TARGETS):
    r = pd.read_csv(f"results/summary_{t}_random.csv", index_col=0)
    g = pd.read_csv(f"results/summary_{t}_grouped.csv", index_col=0).loc[r.index]
    x = np.arange(len(r)); a.bar(x - .2, r.RMSE_mean, .4, yerr=r.RMSE_std, label="Random 10-fold", color="#3b6ea5")
    a.bar(x + .2, g.RMSE_mean, .4, yerr=g.RMSE_std, label="Held-out building shapes", color="#d98c3f")
    a.set_xticks(x); a.set_xticklabels(r.index, rotation=30, ha="right"); a.set_title(TARGETS[t]); a.set_ylabel("RMSE (kWh/m²)")
ax[0].legend(); plt.tight_layout(); plt.savefig("figures/cv_comparison.png", dpi=200); plt.close()
print("done")
