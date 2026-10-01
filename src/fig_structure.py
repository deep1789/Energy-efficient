"""Figure 2: group structure of the three datasets."""
import numpy as np, pandas as pd
import matplotlib; matplotlib.use("Agg"); import matplotlib.pyplot as plt
from common import *
plt.rcParams.update({"font.size": 8, "axes.spines.top": False, "axes.spines.right": False, "font.family": "serif"})
C = ["#1f4e79", "#c0504d", "#4f8f4f", "#7f6000"]
Xe, ye, ge = load_enb(); Xc, yc, gc = load_concrete(); Xp, yp, gp, dp = load_pk("all")
fig, ax = plt.subplots(2, 2, figsize=(7.2, 5.6))
# (a) ENB: heating load per geometry
a = ax[0, 0]; y = ye["Y1"]
for k in range(12):
    v = y[ge == k]; a.scatter(np.full(len(v), k + 1) + np.random.RandomState(k).uniform(-.25, .25, len(v)), v, s=4, color=C[0], alpha=.5)
    a.plot([k + .6, k + 1.4], [v.mean()] * 2, color="k", lw=1)
a.set_xlabel("Building geometry"); a.set_ylabel("Heating load (kWh/m²)"); a.set_title("(a) ENB2012: 12 geometries × 64 rows", loc="left", fontsize=8)
# (b) Concrete: group size histogram
a = ax[0, 1]; n_i = np.bincount(gc); vals, cnt = np.unique(n_i, return_counts=True)
a.bar(vals, cnt, color=C[2]); a.set_yscale("log"); a.set_xlabel("Rows per mix"); a.set_ylabel("Number of mixes (log)")
a.set_title(f"(b) Concrete: {len(n_i)} mixes, {int((n_i==1).sum())} singletons", loc="left", fontsize=8)
# (c) Parkinson's: per-patient distribution of total UPDRS
a = ax[1, 0]; order = np.argsort([yp["total_UPDRS"][gp == s].mean() for s in np.unique(gp)])
data = [yp["total_UPDRS"][gp == np.unique(gp)[o]] for o in order]
bp = a.boxplot(data, widths=.7, showfliers=False, patch_artist=True, medianprops=dict(color="k", lw=.8))
for b in bp["boxes"]: b.set_facecolor("#c9d9ec"); b.set_edgecolor(C[0]); b.set_linewidth(.6)
a.set_xticks([]); a.set_xlabel("42 patients (sorted by mean)"); a.set_ylabel("Total UPDRS"); a.set_title("(c) Parkinson's: 42 patients, 101–168 rows each", loc="left", fontsize=8)
# (d) ICC of the target
a = ax[1, 1]
def icc1(e, g):
    ug, inv = np.unique(g, return_inverse=True); n = np.bincount(inv); N = len(e); k = len(ug); m = np.bincount(inv, weights=e) / n; gr = e.mean()
    msb = (n * (m - gr) ** 2).sum() / (k - 1); msw = ((e - m[inv]) ** 2).sum() / (N - k); n0 = (N - (n ** 2).sum() / N) / (k - 1); su = max((msb - msw) / n0, 0); return su / (su + msw)
lab = ["ENB\nheating", "ENB\ncooling", "Concrete\nstrength", "PD total\nUPDRS", "PD motor\nUPDRS"]
vals = [icc1(ye["Y1"], ge), icc1(ye["Y2"], ge), icc1(yc["strength"], gc), icc1(yp["total_UPDRS"], gp), icc1(yp["motor_UPDRS"], gp)]
a.bar(range(5), vals, color=[C[0], C[0], C[2], C[1], C[1]]); a.set_xticks(range(5)); a.set_xticklabels(lab, fontsize=7); a.set_ylim(0, 1); a.set_ylabel("ICC of the target")
for i, v in enumerate(vals): a.text(i, v + .02, f"{v:.2f}", ha="center", fontsize=7)
a.set_title("(d) Share of target variance between groups", loc="left", fontsize=8)
plt.tight_layout(); plt.savefig("figures/paper/fig02_structure.png", dpi=300); print([round(v, 3) for v in vals])
# descriptive stats table
rows = []
for ds, X, ys, g in (("ENB2012", Xe, ye, ge), ("Concrete", Xc, yc, gc), ("Parkinson", Xp, yp, gp)):
    n_i = np.bincount(g); n_i = n_i[n_i > 0]
    for k, v in ys.items():
        rows.append(dict(dataset=ds, target=k, N=len(v), p=X.shape[1], groups=len(n_i), min_m=n_i.min(), median_m=float(np.median(n_i)), max_m=n_i.max(), mean=v.mean(), sd=v.std(ddof=1), lo=v.min(), hi=v.max()))
pd.DataFrame(rows).round(2).to_csv("results/new/dataset_descriptives.csv", index=False); print(pd.DataFrame(rows).round(2).to_string())
