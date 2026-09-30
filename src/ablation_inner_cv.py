"""Ablation: does hyper-parameter tuning with a *random* inner split leak under leave-one-shape-out outer CV?"""
import sys; sys.argv = ["x"]
import numpy as np, pandas as pd, warnings
warnings.filterwarnings("ignore")
import importlib.util
# reuse model definitions without re-running experiments
src = open("src/run_experiments.py").read().split("rows, oof_rows = [], []")[0]
ns = {}; exec(src, ns)
X, df, groups, MODELS, TARGETS, SEED = (ns[k] for k in ("X", "df", "groups", "MODELS", "TARGETS", "SEED"))
from sklearn.base import clone
from sklearn.model_selection import GridSearchCV, GroupKFold, KFold, LeaveOneGroupOut
from sklearn.metrics import mean_squared_error
rows = []
for t in TARGETS:
    y = df[t].values
    for name in ["Ridge", "SVR (RBF)", "Random forest", "Gradient boosting"]:
        est, grid = MODELS[name]
        for inner in ("random", "grouped"):
            errs = []
            for tr, te in LeaveOneGroupOut().split(X, groups=groups):
                cv = KFold(3, shuffle=True, random_state=SEED) if inner == "random" else list(GroupKFold(3).split(tr, groups=groups[tr]))
                gs = GridSearchCV(clone(est), grid, cv=cv, scoring="neg_root_mean_squared_error").fit(X[tr], y[tr])
                errs.append(mean_squared_error(y[te], gs.predict(X[te])) ** .5)
            rows.append(dict(target=t, model=name, inner_cv=inner, loso_rmse=np.mean(errs)))
            print(rows[-1], flush=True)
pd.DataFrame(rows).to_csv("results/ablation_inner_cv.csv", index=False)
