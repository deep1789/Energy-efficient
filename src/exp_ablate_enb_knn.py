"""ENB2012 LOSO tuning ablation for kNN (random vs group-matched inner folds)."""
import numpy as np, pandas as pd, warnings
warnings.filterwarnings("ignore")
from sklearn.base import clone
from sklearn.model_selection import GridSearchCV, GroupKFold, KFold, LeaveOneGroupOut
from sklearn.neighbors import KNeighborsRegressor
from sklearn.pipeline import make_pipeline
from sklearn.preprocessing import StandardScaler
from common import *
X, ys, g = load_enb(); rows = []
for t, y in ys.items():
    for inner in ("random", "grouped"):
        errs = []
        for tr, te in LeaveOneGroupOut().split(X, groups=g):
            cv = KFold(3, shuffle=True, random_state=SEED) if inner == "random" else list(GroupKFold(3).split(tr, groups=g[tr]))
            gs = GridSearchCV(make_pipeline(StandardScaler(), KNeighborsRegressor()), {"kneighborsregressor__n_neighbors": [3, 9, 15]}, cv=cv, scoring="neg_root_mean_squared_error").fit(X[tr], y[tr])
            errs.append(rmse(y[te], gs.predict(X[te])))
        rows.append(dict(target=t, model="kNN", inner_cv=inner, loso_rmse=float(np.mean(errs))))
pd.DataFrame(rows).to_csv("results/new/ablation_enb_knn.csv", index=False); print(rows)
