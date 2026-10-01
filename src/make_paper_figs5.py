"""Figure 13: sensitivity to K and fold seed; precision versus number of groups."""
import numpy as np, pandas as pd
from scipy.stats import t as tdist
import matplotlib; matplotlib.use("Agg"); import matplotlib.pyplot as plt
plt.rcParams.update({"font.size": 8, "axes.spines.top": False, "axes.spines.right": False, "font.family": "serif"})
BLUE, ORG, GRN, RED, GREY = "#1f4e79", "#d98c3f", "#4f8f4f", "#c0504d", "#8a8a8a"
s = pd.read_csv("results/new/sensitivity_K.csv"); fig, ax = plt.subplots(2, 2, figsize=(7.2, 5.4)); ax = ax.ravel()
for a, ds, title in zip(ax[:3], ["ENB2012", "Concrete", "Parkinson"], ["(a) ENB2012 heating", "(b) Concrete strength", "(c) Parkinson's total UPDRS"]):
    q = s[s.dataset == ds]
    for m, c in (("GBM", BLUE), ("Ridge", GREY)):
        for proto, ls, mk in (("random", "--", "o"), ("grouped", "-", "s")):
            z = q[(q.model == m) & (q.protocol == proto)].groupby("K").rmse.agg(["mean", "std"]).reset_index()
            a.errorbar(z.K, z["mean"], yerr=z["std"], fmt=ls, marker=mk, ms=3.5, color=c, capsize=2, lw=1.3, label=f"{'gradient boosting' if m=='GBM' else 'ridge'}, {'held-out groups' if proto=='grouped' else 'random'}")
    a.set_xlabel("Number of folds $K$"); a.set_ylabel("RMSE"); a.set_title(title, loc="left", fontsize=8.5); a.set_xticks(sorted(q.K.unique()))
h, l = ax[0].get_legend_handles_labels(); fig.legend(h, l, loc="lower center", ncol=4, frameon=False, fontsize=6.5)
G = np.arange(3, 500); a = ax[3]; a.plot(G, tdist.ppf(.975, G - 1) / np.sqrt(G), color=BLUE, lw=1.6); a.set_xscale("log"); a.set_yscale("log")
for g_, lab in ((12, "ENB2012\n$G=12$"), (42, "Parkinson's\n$G=42$"), (427, "Concrete\n$G=427$")):
    v = tdist.ppf(.975, g_ - 1) / np.sqrt(g_); a.scatter(g_, v, color=RED, zorder=4, s=25); a.annotate(f"{lab}\n{v:.2f}", (g_, v), xytext=(6, 6), textcoords="offset points", fontsize=6.5)
a.set_xlabel("Number of groups $G$"); a.set_ylabel("$t_{G-1,0.975}/\\sqrt{G}$ (relative half-width $\\div$ CV)"); a.set_title("(d) Precision of a group-level estimate", loc="left", fontsize=8.5)
plt.tight_layout(rect=(0, .05, 1, 1)); plt.savefig("figures/paper/fig13_sensitivity.png", dpi=300); plt.close()
q = s[s.model == "GBM"].groupby(["dataset", "K", "protocol"]).rmse.mean().unstack(); q["rho"] = q.grouped / q.random; print(q.round(2))
