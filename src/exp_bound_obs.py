"""Observed inflation of fixed-hyper-parameter GBM and Ridge (same models as in icc_bound.csv) for the bound comparison."""
import numpy as np, pandas as pd
from sklearn.base import clone
from sklearn.model_selection import KFold, GroupKFold, LeaveOneGroupOut
from common import *
pk = load_pk("all"); rows = []
for ds, tg, (X, ys, g), key, Kd in [("ENB2012", "heating", load_enb(), "Y1", 12), ("ENB2012", "cooling", load_enb(), "Y2", 12), ("Concrete", "strength", load_concrete(), "strength", 10),
        ("Parkinson", "total UPDRS", pk[:3], "total_UPDRS", 10), ("Parkinson", "motor UPDRS", pk[:3], "motor_UPDRS", 10)]:
    y = ys[key]
    for m in ("GBM", "Ridge"):
        res = {}
        for proto in ("random", "grouped"):
            sp = KFold(Kd, shuffle=True, random_state=SEED).split(X) if proto == "random" else (LeaveOneGroupOut().split(X, groups=g) if ds == "ENB2012" else GroupKFold(Kd).split(X, groups=g))
            se = 0.0
            for tr, te in sp: se += ((y[te] - clone(fixed_model(m)).fit(X[tr], y[tr]).predict(X[te])) ** 2).sum()
            res[proto] = (se / len(y)) ** .5
        rows.append(dict(dataset=ds, target=tg, model=m, random=res["random"], grouped=res["grouped"], rho_obs=res["grouped"] / res["random"])); print(rows[-1], flush=True)
pd.DataFrame(rows).to_csv("results/new/bound_observed.csv", index=False); print("done")
