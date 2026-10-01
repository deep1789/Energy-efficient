"""Mechanism of tuning leakage on ENB2012 (LOSO): inner-CV error versus true held-out error as a function of capacity
(SVR kernel width gamma; kNN k), for random and group-matched inner folds."""
import numpy as np, pandas as pd
from sklearn.base import clone
from sklearn.model_selection import KFold, GroupKFold, LeaveOneGroupOut, cross_val_predict
from sklearn.neighbors import KNeighborsRegressor
from sklearn.pipeline import make_pipeline
from sklearn.preprocessing import StandardScaler
from sklearn.svm import SVR
from common import *
X, ys, g = load_enb(); y = ys["Y1"]
GAM = [0.003, 0.01, 0.03, 0.1, 0.3, 1.0]; KS = [1, 3, 5, 9, 15, 25, 41, 65, 129]
rows = []
for tr, te in LeaveOneGroupOut().split(X, groups=g):
    inner = {"random": KFold(3, shuffle=True, random_state=SEED), "grouped": list(GroupKFold(3).split(tr, groups=g[tr]))}
    for fam, grid in (("SVR gamma", GAM), ("kNN k", KS)):
        for v in grid:
            mdl = make_pipeline(StandardScaler(), SVR(C=30.0, gamma=v)) if fam == "SVR gamma" else make_pipeline(StandardScaler(), KNeighborsRegressor(n_neighbors=v))
            outer = rmse(y[te], clone(mdl).fit(X[tr], y[tr]).predict(X[te]))
            rec = dict(family=fam, value=v, heldout_geometry=int(g[te][0]), outer_mse=outer ** 2)
            for name, cv in inner.items():
                p = cross_val_predict(clone(mdl), X[tr], y[tr], cv=cv); rec["inner_" + name + "_mse"] = float(np.mean((y[tr] - p) ** 2))
            rows.append(rec)
pd.DataFrame(rows).to_csv("results/new/tuning_paths.csv", index=False)
d = pd.DataFrame(rows).groupby(["family", "value"])[["outer_mse", "inner_random_mse", "inner_grouped_mse"]].mean() ** .5
print(d.round(2)); print("done")
