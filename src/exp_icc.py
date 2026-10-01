"""Residual ICC of group-held-out predictions of flexible models, and the implied upper bound on the inflation ratio."""
import numpy as np, pandas as pd
from sklearn.base import clone
from sklearn.model_selection import GroupKFold, LeaveOneGroupOut
from common import *

def icc_oneway(e, g):
    ug, inv = np.unique(g, return_inverse=True); n_i = np.bincount(inv); N = len(e); k = len(ug)
    means = np.bincount(inv, weights=e) / n_i; grand = e.mean()
    msb = (n_i * (means - grand) ** 2).sum() / (k - 1); msw = ((e - means[inv]) ** 2).sum() / (N - k)
    n0 = (N - (n_i ** 2).sum() / N) / (k - 1); su = max((msb - msw) / n0, 0.0)
    return su / (su + msw), su, msw, n_i

pk = load_pk("all"); out = []
for ds, tg, (X, ys, g), key, Kd in [("ENB2012", "heating", load_enb(), "Y1", 12), ("ENB2012", "cooling", load_enb(), "Y2", 12),
        ("Concrete", "strength", load_concrete(), "strength", 10), ("Parkinson", "total UPDRS", pk[:3], "total_UPDRS", 10), ("Parkinson", "motor UPDRS", pk[:3], "motor_UPDRS", 10)]:
    y = ys[key]; sp = list(LeaveOneGroupOut().split(X, groups=g)) if ds == "ENB2012" else list(GroupKFold(10).split(X, groups=g))
    for m in ("Ridge", "GBM"):
        e = np.zeros(len(y))
        for tr, te in sp: e[te] = y[te] - clone(fixed_model(m)).fit(X[tr], y[tr]).predict(X[te])
        r, su, sw, n_i = icc_oneway(e, g)
        tot = (e ** 2).mean()                         # group-held-out MSE
        n_tr = np.average(n_i, weights=n_i) * (Kd - 1) / Kd
        # decompose held-out MSE: group-offset part and within-group part (uncentred, using group means of the residual)
        inv = pd.factorize(g)[0]; gm = np.bincount(inv, weights=e) / np.bincount(inv)
        between = (gm[inv] ** 2).mean(); within = tot - between
        sigma2_u, sigma2_w = between, within                               # offset energy and within-group energy
        V = 1.0 / (1.0 / max(sigma2_u, 1e-12) + n_tr / sigma2_w)             # posterior variance of a group offset
        rho_bound = np.sqrt(tot / (sigma2_w + V))
        out.append(dict(dataset=ds, target=tg, model=m, groups=len(n_i), rowwt_m_train=n_tr, grouped_rmse=tot ** .5, icc_resid=r,
                        offset_share=between / tot, rho_bound=rho_bound, rho_bound_inf=np.sqrt(tot / sigma2_w)))
        print(out[-1], flush=True)
pd.DataFrame(out).to_csv("results/new/icc_bound.csv", index=False); print("done")
