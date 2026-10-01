"""Same leakage-aware protocol on two further datasets with natural groups.

concrete   - UCI Concrete Compressive Strength; group = identical 7 ingredient quantities (a mix tested at several ages)
parkinsons - UCI Parkinson's Telemonitoring; group = patient (subject#)
Protocols: random K-fold; grouped K-fold (inner tuning also grouped); grouped K-fold with RANDOM inner tuning (ablation).
Usage: python src/run_extra.py concrete|parkinsons
"""
import sys, time, warnings
import numpy as np, pandas as pd
from sklearn.base import clone
from sklearn.ensemble import GradientBoostingRegressor, RandomForestRegressor
from sklearn.linear_model import Ridge
from sklearn.metrics import mean_absolute_error, mean_squared_error, r2_score
from sklearn.model_selection import GridSearchCV, GroupKFold, KFold
from sklearn.neighbors import KNeighborsRegressor
from sklearn.pipeline import make_pipeline
from sklearn.preprocessing import StandardScaler
from sklearn.svm import SVR

warnings.filterwarnings("ignore")
SEED = 42
name = sys.argv[1]
ONLY_PROTOCOL = sys.argv[2] if len(sys.argv) > 2 else None     # optional: run a single protocol / target in its own process
ONLY_TARGET = sys.argv[3] if len(sys.argv) > 3 else None

if name == "concrete":
    d = pd.read_csv("data/concrete_data.csv"); d.columns = [c.strip() for c in d.columns]
    for c in d.columns: d[c] = pd.to_numeric(d[c].astype(str).str.strip())
    feats = [c for c in d.columns if c != "concrete_compressive_strength"]
    targets = ["concrete_compressive_strength"]
    groups = pd.factorize(d[feats[:7]].astype(str).agg("|".join, axis=1))[0]
    K, REP = 10, 2
else:
    d = pd.read_csv("data/parkinsons_updrs.data")
    targets = ["total_UPDRS", "motor_UPDRS"]
    feats = [c for c in d.columns if c not in ("subject#", "total_UPDRS", "motor_UPDRS")]
    groups = d["subject#"].values
    K, REP = 10, 1
X = d[feats].values
print(name, X.shape, "groups", len(set(groups)), flush=True)

MODELS = {
    "Ridge": (make_pipeline(StandardScaler(), Ridge()), {"ridge__alpha": [0.1, 1, 10, 100]}),
    "kNN": (make_pipeline(StandardScaler(), KNeighborsRegressor()), {"kneighborsregressor__n_neighbors": [3, 9, 15]}),
    "SVR (RBF)": (make_pipeline(StandardScaler(), SVR()), {"svr__C": [1, 30], "svr__gamma": ["scale", 0.02]}),
    "Random forest": (RandomForestRegressor(n_estimators=100, random_state=SEED, n_jobs=1),
                      {"max_features": [0.5, 1.0], "min_samples_leaf": [1, 5]}),
    "Gradient boosting": (GradientBoostingRegressor(random_state=SEED, n_estimators=200),
                          {"learning_rate": [0.05, 0.1], "max_depth": [2, 3]}),
}
import os as _os
ABLATE = list(MODELS) if _os.environ.get("ABLATE_ALL") else ["Ridge", "SVR (RBF)", "Gradient boosting"]

def splits(protocol, rep):
    rs = np.random.RandomState(SEED + rep)
    if protocol == "random":
        return list(KFold(K, shuffle=True, random_state=SEED + rep).split(X))
    ug = np.unique(groups); perm = dict(zip(ug, rs.permutation(len(ug))))   # shuffled group ids -> different folds per repeat
    return list(GroupKFold(K).split(X, groups=np.array([perm[g] for g in groups])))

def inner(tr, grouped):
    if grouped:
        return list(GroupKFold(3).split(tr, groups=groups[tr]))
    return KFold(3, shuffle=True, random_state=SEED)

import os
OUT = f"results/extra/{name}_folds.csv" if not ONLY_PROTOCOL else f"results/extra/parts/{name}__{ONLY_PROTOCOL}__{ONLY_TARGET}.csv"
rows = pd.read_csv(OUT).to_dict("records") if os.path.exists(OUT) else []     # resume after a restart
done = {(r["protocol"], r["rep"], r["target"], r["model"]) for r in rows}
diag = []
for protocol, grouped_inner, models in (("random", False, MODELS), ("grouped", True, MODELS), ("grouped_randominner", False, {m: MODELS[m] for m in ABLATE})):
    if ONLY_PROTOCOL and protocol != ONLY_PROTOCOL:
        continue
    outer = "random" if protocol == "random" else "grouped"
    for rep in range(REP):
        sp = splits(outer, rep)
        if protocol != "grouped_randominner":
            for k, (tr, te) in enumerate(sp):
                diag.append(dict(protocol=protocol, rep=rep, fold=k, frac_test_group_in_train=np.isin(groups[te], groups[tr]).mean(), n_test=len(te)))
        for t in targets:
            if ONLY_TARGET and t != ONLY_TARGET:
                continue
            y = d[t].values
            for m, (est, grid) in models.items():
                if (protocol, rep, t, m) in done:
                    continue
                t0 = time.time()
                for k, (tr, te) in enumerate(sp):
                    gs = GridSearchCV(clone(est), grid, cv=inner(tr, grouped_inner), scoring="neg_root_mean_squared_error").fit(X[tr], y[tr])
                    p = gs.predict(X[te])
                    rows.append(dict(dataset=name, protocol=protocol, target=t, model=m, rep=rep, fold=k,
                                     n_test=len(te), sse=float(((y[te] - p) ** 2).sum()), RMSE=mean_squared_error(y[te], p) ** .5,
                                     MAE=mean_absolute_error(y[te], p), R2=r2_score(y[te], p) if len(te) > 1 else np.nan))
                print(f"{protocol:20s} rep{rep} {t:30s} {m:18s} {time.time()-t0:5.0f}s", flush=True)
                pd.DataFrame(rows).to_csv(OUT, index=False)   # save after every model
if not ONLY_PROTOCOL:
    pd.DataFrame(diag).to_csv(f"results/extra/{name}_diagnostics.csv", index=False)
print("done", flush=True)
