"""(A) Synthetic validation of the random-effects inflation formula; (B) ICC of real datasets and predicted vs observed inflation;
(C) leakage-fraction formula versus measured values."""
import itertools, json, numpy as np, pandas as pd
from sklearn.model_selection import KFold, GroupKFold
from sklearn.base import clone
from common import *

def v_post(r, m_tr):  # posterior variance of a group effect given m_tr rows of the group (unit total variance)
    return 1.0 / (1.0 / r + m_tr / (1 - r))
def rho_theory(r, m_tr, shrink=True):
    return float(np.sqrt(1.0 / ((1 - r) + (v_post(r, m_tr) if shrink else (1 - r) / m_tr))))

# ---------------- A: simulation ----------------
rows = []; G = 50; K = 10
for r, m in itertools.product([0.05, 0.2, 0.4, 0.6, 0.8], [2, 5, 20, 64]):
    for rep in range(3):
        rs = np.random.RandomState(1000 * rep + m + int(100 * r))
        z = rs.randn(G, 3); u = rs.randn(G) * np.sqrt(r)
        gid = np.repeat(np.arange(G), m); w = rs.randn(G * m, 2)
        f = z[gid, 0] + 0.5 * z[gid, 1] - 0.5 * w[:, 0] + 0.3 * np.sin(2 * w[:, 1])
        f = f / 1.0
        y = f + u[gid] + rs.randn(G * m) * np.sqrt(1 - r)
        X = np.column_stack([z[gid], w])
        res = {}
        for name in ("Ridge", "GBM"):
            mdl = fixed_model(name)
            if name == "GBM": mdl.set_params(n_estimators=150)
            for proto, sp in (("random", KFold(K, shuffle=True, random_state=rep).split(X)), ("grouped", GroupKFold(K).split(X, groups=gid))):
                se = 0.0
                for tr, te in sp:
                    p = clone(mdl).fit(X[tr], y[tr]).predict(X[te]); se += ((y[te] - p) ** 2).sum()
                res[(name, proto)] = (se / len(y)) ** .5
        # oracle noise floors: RMSE relative to each protocol's irreducible error
        rows.append(dict(icc=r, m=m, rep=rep, ridge_rand=res[("Ridge", "random")], ridge_grp=res[("Ridge", "grouped")],
                         gbm_rand=res[("GBM", "random")], gbm_grp=res[("GBM", "grouped")],
                         rho_gbm=res[("GBM", "grouped")] / res[("GBM", "random")], rho_ridge=res[("Ridge", "grouped")] / res[("Ridge", "random")],
                         rho_theory=rho_theory(r, m * (K - 1) / K), rho_theory_max=1 / np.sqrt(1 - r)))
    print("sim", r, m, flush=True)
pd.DataFrame(rows).to_csv("results/new/sim_theory.csv", index=False)

# ---------------- B: ICC of real data ----------------
def icc_oneway(e, g):
    ug, inv = np.unique(g, return_inverse=True); n_i = np.bincount(inv); N = len(e); k = len(ug)
    means = np.bincount(inv, weights=e) / n_i; grand = e.mean()
    msb = (n_i * (means - grand) ** 2).sum() / (k - 1); msw = ((e - means[inv]) ** 2).sum() / (N - k)
    n0 = (N - (n_i ** 2).sum() / N) / (k - 1)
    su = max((msb - msw) / n0, 0.0); return su / (su + msw), su, msw, n_i
out = []
pk = load_pk("all"); datasets = [("ENB2012", "heating", load_enb(), "Y1", 12), ("ENB2012", "cooling", load_enb(), "Y2", 12),
    ("Concrete", "strength", load_concrete(), "strength", 10), ("Parkinson", "total UPDRS", pk[:3], "total_UPDRS", 10), ("Parkinson", "motor UPDRS", pk[:3], "motor_UPDRS", 10)]
for ds, tg, (X, ys, g), key, Kd in datasets:
    y = ys[key]
    icc_y, *_ = icc_oneway(y, g)
    # residuals of a grouped-CV ridge (no group memory): share of residual variance that is a group offset
    pred = np.zeros(len(y))
    for tr, te in GroupKFold(10).split(X, groups=g): pred[te] = fixed_model("Ridge").fit(X[tr], y[tr]).predict(X[te])
    icc_r, su, sw, n_i = icc_oneway(y - pred, g)
    m_tr = float(np.average(n_i, weights=n_i)) * (Kd - 1) / Kd        # row-weighted mean group size in a training set
    out.append(dict(dataset=ds, target=tg, groups=len(n_i), mean_m=n_i.mean(), rowwt_m=np.average(n_i, weights=n_i), icc_y=icc_y, icc_resid=icc_r,
                    rho_theory=rho_theory(icc_r, m_tr), rho_theory_max=1 / np.sqrt(1 - icc_r)))
pd.DataFrame(out).to_csv("results/new/icc_real.csv", index=False); print(pd.DataFrame(out).round(3).to_string())

# ---------------- C: leakage fraction ----------------
rows = []
for ds, (X, ys, g), Kd in (("ENB2012", load_enb(), 12), ("Concrete", load_concrete(), 10), ("Parkinson", pk[:3], 10)):
    n_i = np.bincount(g)
    for K2 in (2, 5, 10, 20):
        frac = float(((n_i) * (1 - (1.0 / K2) ** (n_i - 1))).sum() / n_i.sum())
        rows.append(dict(dataset=ds, K=K2, predicted_frac=frac))
    # measured with actual folds
    meas = np.mean([np.isin(g[te], g[tr]).mean() for tr, te in KFold(Kd, shuffle=True, random_state=SEED).split(X)])
    rows.append(dict(dataset=ds, K=Kd, measured_frac=meas))
pd.DataFrame(rows).to_csv("results/new/leak_fraction.csv", index=False); print(pd.DataFrame(rows).round(3).to_string()); print("done")
