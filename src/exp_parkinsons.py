"""Parkinson's: (1) feature-set ablation, random vs patient-held-out; (2) per-patient skill vs no-skill baseline;
(3) personalisation curve: how many recordings of a new patient restore accuracy."""
import numpy as np, pandas as pd
from sklearn.base import clone
from sklearn.model_selection import KFold, GroupKFold
from common import *

# ---------- (1) feature-set ablation ----------
rows = []
for fs in ("voice", "voice+demo", "voice+time", "all", "demo+time"):
    X, ys, g, d = load_pk(fs)
    for tg in ("total_UPDRS",):
        y = ys[tg]
        for m in ("Ridge", "RF", "GBM"):
            for proto in ("random", "grouped"):
                sp = KFold(10, shuffle=True, random_state=SEED).split(X) if proto == "random" else GroupKFold(10).split(X, groups=g)
                se = 0.0
                for tr, te in sp: se += ((y[te] - clone(fixed_model(m)).fit(X[tr], y[tr]).predict(X[te])) ** 2).sum()
                rows.append(dict(features=fs, n_features=X.shape[1], target=tg, model=m, protocol=proto, rmse=(se / len(y)) ** .5))
        print("ablation", fs, flush=True)
    pd.DataFrame(rows).to_csv("results/new/pk_feature_ablation.csv", index=False)

# ---------- (2) per-patient skill, all features, grouped 10-fold OOF ----------
X, ys, g, d = load_pk("all"); y = ys["total_UPDRS"]
oof = {m: np.zeros(len(y)) for m in ("Ridge", "RF", "GBM")}; base = np.zeros(len(y)); fold_of = np.zeros(len(y), int)
for f, (tr, te) in enumerate(GroupKFold(10).split(X, groups=g)):
    fold_of[te] = f; base[te] = y[tr].mean()
    for m in oof: oof[m][te] = clone(fixed_model(m)).fit(X[tr], y[tr]).predict(X[te])
pp = []
for s in np.unique(g):
    i = g == s; rec = dict(subject=int(s), n=int(i.sum()), mean_updrs=float(y[i].mean()), baseline_rmse=rmse(y[i], base[i]))
    for m in oof: rec[m + "_rmse"] = rmse(y[i], oof[m][i]); rec[m + "_bias"] = float((oof[m][i] - y[i]).mean())
    pp.append(rec)
pd.DataFrame(pp).to_csv("results/new/pk_per_patient.csv", index=False)
pd.DataFrame(dict(subject=g, y=y, base=base, **{m: oof[m] for m in oof})).to_csv("results/new/pk_oof.csv", index=False)
print("per patient done", flush=True)

# ---------- (3) personalisation curve ----------
# held-out patient s; model trained on the other patients of its fold; first n recordings (by test_time) of s are available;
# evaluate on recordings 41+ of s (fixed evaluation set). Variants: offset correction; GBM retrained with the n recordings (weight 20).
rows = []; NS = [0, 5, 10, 20, 40]
for f in range(10):
    te_idx = np.where(fold_of == f)[0]; tr_idx = np.where(fold_of != f)[0]
    base_models = {m: clone(fixed_model(m)).fit(X[tr_idx], y[tr_idx]) for m in ("Ridge", "GBM")}
    for s in np.unique(g[te_idx]):
        i = np.where(g == s)[0]; i = i[np.argsort(d.test_time.values[i])]; ev = i[40:]
        for m, mdl in base_models.items():
            p_ev = mdl.predict(X[ev]); p_first = mdl.predict(X[i[:40]])
            for n in NS:
                off = (y[i[:n]] - p_first[:n]).mean() if n > 0 else 0.0
                rows.append(dict(subject=int(s), model=m, variant="offset", n=n, se=float(((y[ev] - (p_ev + off)) ** 2).sum()), cnt=len(ev)))
        for n in NS[1:]:
            Xa = np.vstack([X[tr_idx], X[i[:n]]]); ya = np.concatenate([y[tr_idx], y[i[:n]]])
            w = np.concatenate([np.ones(len(tr_idx)), np.full(n, 20.0)])
            mdl = fixed_model("GBM").fit(Xa, ya, sample_weight=w)
            rows.append(dict(subject=int(s), model="GBM", variant="retrain_w20", n=n, se=float(((y[ev] - mdl.predict(X[ev])) ** 2).sum()), cnt=len(ev)))
        base_se = float(((y[ev] - base[ev]) ** 2).sum()); rows.append(dict(subject=int(s), model="mean", variant="baseline", n=0, se=base_se, cnt=len(ev)))
    print("personalisation fold", f, flush=True)
    pd.DataFrame(rows).to_csv("results/new/pk_personalisation.csv", index=False)
print("done")
