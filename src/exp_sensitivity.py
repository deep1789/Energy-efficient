"""Sensitivity of the random-vs-grouped gap to the number of folds K and to the fold assignment seed (fixed hyper-parameters)."""
import numpy as np, pandas as pd
from sklearn.base import clone
from sklearn.model_selection import KFold, GroupKFold
from common import *
rows = []
cfg = [("ENB2012", "Y1", load_enb(), [3, 4, 6, 12], ["Ridge", "GBM"]), ("Concrete", "strength", load_concrete(), [5, 10, 20], ["Ridge", "GBM"]),
       ("Parkinson", "total_UPDRS", load_pk("all")[:3], [5, 10, 20], ["Ridge", "GBM"])]
for ds, tg, (X, ys, g), Ks, models in cfg:
    y = ys[tg]; ug = np.unique(g)
    for K in Ks:
        for seed in range(3):
            perm = dict(zip(ug, np.random.RandomState(seed).permutation(len(ug)))); gp = np.array([perm[x] for x in g])
            for m in models:
                for proto in ("random", "grouped"):
                    sp = KFold(K, shuffle=True, random_state=seed).split(X) if proto == "random" else GroupKFold(K).split(X, groups=gp)
                    se = 0.0
                    for tr, te in sp: se += ((y[te] - clone(fixed_model(m)).fit(X[tr], y[tr]).predict(X[te])) ** 2).sum()
                    rows.append(dict(dataset=ds, K=K, seed=seed, model=m, protocol=proto, rmse=(se / len(y)) ** .5))
        print(ds, K, flush=True)
    pd.DataFrame(rows).to_csv("results/new/sensitivity_K.csv", index=False)
print("done")
