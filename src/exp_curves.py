"""Learning curves versus number of training groups (group-held-out) with a matched random-split comparator;
plus concrete: random-split error stratified by the number of other rows of the same mix."""
import numpy as np, pandas as pd
from sklearn.base import clone
from sklearn.model_selection import KFold, GroupKFold
from common import *

rows = []
cfg = [("ENB2012", "Y1", load_enb(), 2, [2, 3, 4, 6, 8, 10], 25, ["Ridge", "SVR", "RF", "GBM"]),
       ("ENB2012", "Y2", load_enb(), 2, [2, 3, 4, 6, 8, 10], 25, ["Ridge", "SVR", "RF", "GBM"]),
       ("Concrete", "strength", load_concrete(), 40, [20, 50, 100, 200, 300, 380], 8, ["Ridge", "SVR", "RF", "GBM"]),
       ("Parkinson", "total_UPDRS", load_pk("all")[:3], 6, [4, 8, 12, 20, 28, 36], 6, ["Ridge", "SVR", "GBM"])]
for ds, tg, (X, ys, g), t_groups, ks, R, models in cfg:
    y = ys[tg]; ug = np.unique(g)
    for k in ks:
        for rep in range(R):
            rs = np.random.RandomState(100 * rep + k)
            perm = rs.permutation(ug); te_g, tr_g = perm[:t_groups], perm[t_groups:t_groups + k]
            te = np.isin(g, te_g); tr = np.isin(g, tr_g)
            # random-split comparator: same number of training rows and same number of test rows, drawn at random from all rows
            idx = rs.permutation(len(y)); te_r = idx[:te.sum()]; tr_r = idx[te.sum():te.sum() + tr.sum()]
            for m in models:
                mdl = fixed_model(m)
                rows.append(dict(dataset=ds, target=tg, model=m, k_groups=k, rep=rep, n_train=int(tr.sum()), protocol="grouped",
                                 rmse=rmse(y[te], clone(mdl).fit(X[tr], y[tr]).predict(X[te]))))
                rows.append(dict(dataset=ds, target=tg, model=m, k_groups=k, rep=rep, n_train=int(tr.sum()), protocol="random",
                                 rmse=rmse(y[te_r], clone(mdl).fit(X[tr_r], y[tr_r]).predict(X[te_r]))))
        print(ds, tg, "k", k, flush=True)
    pd.DataFrame(rows).to_csv("results/new/learning_curves.csv", index=False)

# ---- concrete: stratify random-split error by leaked group-mate count ----
X, ys, g = load_concrete(); y = ys["strength"]; n_i = np.bincount(g); mates = n_i[g] - 1
res = []
for m in ("GBM", "RF", "Ridge"):
    for proto in ("random", "grouped"):
        oof = np.zeros(len(y)); cnt = np.zeros(len(y))
        for rep in range(3):
            if proto == "random": sp = KFold(10, shuffle=True, random_state=rep).split(X)
            else:
                perm = dict(zip(np.unique(g), np.random.RandomState(rep).permutation(len(n_i)))); sp = GroupKFold(10).split(X, groups=np.array([perm[x] for x in g]))
            for tr, te in sp:
                oof[te] += clone(fixed_model(m)).fit(X[tr], y[tr]).predict(X[te]); cnt[te] += 1
        e = y - oof / cnt
        bins = pd.cut(mates, [-1, 0, 2, 5, 10, 100], labels=["0", "1-2", "3-5", "6-10", ">10"])
        for b in bins.categories:
            s = bins == b; res.append(dict(model=m, protocol=proto, mates=str(b), n=int(s.sum()), rmse=float(np.sqrt((e[s] ** 2).mean()))))
pd.DataFrame(res).to_csv("results/new/concrete_stratified.csv", index=False); print("done")
