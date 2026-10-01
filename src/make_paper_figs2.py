"""Figures 6 (learning curves), 7 (extrapolation distance, ENB2012)."""
import numpy as np, pandas as pd
import matplotlib; matplotlib.use("Agg"); import matplotlib.pyplot as plt, matplotlib.ticker
from scipy.stats import spearmanr
from common import *
plt.rcParams.update({"font.size": 8, "axes.spines.top": False, "axes.spines.right": False, "font.family": "serif"})
BLUE, ORG, GRN, RED, GREY = "#1f4e79", "#d98c3f", "#4f8f4f", "#c0504d", "#8a8a8a"
MC = {"Ridge": GREY, "SVR": ORG, "RF": GRN, "GBM": BLUE}
# ---- Fig 6 ----
d = pd.read_csv("results/new/learning_curves.csv").groupby(["dataset", "target", "model", "protocol", "k_groups"]).rmse.agg(["mean", "std"]).reset_index()
sd = {"Y1": 10.09, "Y2": 9.51, "strength": 16.71, "total_UPDRS": 10.70}
panels = [("ENB2012", "Y1", "(a) ENB2012 heating"), ("ENB2012", "Y2", "(b) ENB2012 cooling"), ("Concrete", "strength", "(c) Concrete strength"), ("Parkinson", "total_UPDRS", "(d) Parkinson's total UPDRS")]
fig, ax = plt.subplots(2, 2, figsize=(7.2, 5.6)); ax = ax.ravel()
for a, (ds, tg, title) in zip(ax, panels):
    s = d[(d.dataset == ds) & (d.target == tg)]
    for m in s.model.unique():
        for proto, ls in (("grouped", "-"), ("random", "--")):
            q = s[(s.model == m) & (s.protocol == proto)].sort_values("k_groups")
            a.plot(q.k_groups, q["mean"], ls, marker="o", ms=3, color=MC[m], lw=1.4, label=m if proto == "grouped" else None)
    a.axhline(sd[tg], color="k", ls=":", lw=1); a.set_title(title, loc="left", fontsize=8.5); a.set_xlabel("Number of training groups"); a.set_ylabel("RMSE")
    if ds == "Concrete":
        a.set_xscale("log"); a.set_xticks([20, 50, 100, 200, 380]); a.xaxis.set_major_formatter(matplotlib.ticker.ScalarFormatter()); a.minorticks_off()
ax[0].plot([], [], "k-", label="held-out groups"); ax[0].plot([], [], "k--", label="random split (same rows)"); ax[0].plot([], [], "k:", label="target SD (no skill)")
ax[0].legend(fontsize=6.5, frameon=False, ncol=2, loc="upper right"); plt.tight_layout(); plt.savefig("figures/paper/fig06_learning_curves.png", dpi=300); plt.close()

# ---- Fig 7: extrapolation distance in ENB2012 ----
X, ys, g = load_enb(); oof = pd.read_csv("results/loso_oof_predictions.csv"); oof = oof[oof.target == "Y1"]
geo = np.array([X[g == k][0, :5] for k in range(12)]); z = (geo - geo.mean(0)) / geo.std(0)
dist = np.array([np.min([np.linalg.norm(z[k] - z[j]) for j in range(12) if j != k]) for k in range(12)])
oof["se"] = (oof.y - oof.pred) ** 2; per = oof.groupby(["model", "shape"]).se.mean().pow(.5).unstack()
fig, ax = plt.subplots(1, 2, figsize=(7.2, 3.0))
lab = {"Ridge": ("Ridge", GREY), "SVR (RBF)": ("SVR", ORG), "Gradient boosting": ("Gradient boosting", BLUE), "Random forest": ("Random forest", GRN)}
rows = []
for m, (nm, c) in lab.items():
    v = per.loc[m].values; ax[0].scatter(dist, v, color=c, s=22, label=nm); rho, p = spearmanr(dist, v); rows.append(dict(model=m, spearman=rho, p=p))
for k in range(12): ax[0].annotate(str(k + 1), (dist[k], per.loc["Gradient boosting"].values[k]), fontsize=5.5, xytext=(2, 2), textcoords="offset points")
ax[0].set_xlabel("Distance to nearest training geometry (standardised $X_1$–$X_5$)"); ax[0].set_ylabel("LOSO RMSE (kWh/m²)"); ax[0].legend(fontsize=6.5, frameon=False); ax[0].set_title("(a) Error versus extrapolation distance", loc="left", fontsize=8.5)
pd.DataFrame(rows).to_csv("results/new/enb_distance_spearman.csv", index=False)
edge = np.isin(np.arange(12), [0, 5, 6, 11]); w = .35
for i, (m, (nm, c)) in enumerate(lab.items()):
    ax[1].bar(i - w / 2, per.loc[m].values[~edge].mean(), w, color=c, alpha=.55); ax[1].bar(i + w / 2, per.loc[m].values[edge].mean(), w, color=c)
ax[1].set_xticks(range(4)); ax[1].set_xticklabels([v[0] for v in lab.values()], fontsize=7); ax[1].set_ylabel("Mean per-geometry RMSE (kWh/m²)")
ax[1].bar([], [], color="k", alpha=.55, label="interior geometries (2–5, 8–11)"); ax[1].bar([], [], color="k", label="end-of-grid geometries (1, 6, 7, 12)"); ax[1].legend(fontsize=6.5, frameon=False, loc="upper left")
ax[1].set_title("(b) Interior versus end-of-grid", loc="left", fontsize=8.5); plt.tight_layout(); plt.savefig("figures/paper/fig07_extrapolation.png", dpi=300); plt.close()
print(pd.DataFrame(rows).round(2)); print(per.round(2).loc[list(lab)].assign(**{}).T.assign(dist=dist.round(2)).round(2).to_string())
