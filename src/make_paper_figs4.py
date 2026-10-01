"""Figure 11 (theory vs simulation and vs real data) and the numbers for Section 6.5."""
import numpy as np, pandas as pd
import matplotlib; matplotlib.use("Agg"); import matplotlib.pyplot as plt
plt.rcParams.update({"font.size": 8, "axes.spines.top": False, "axes.spines.right": False, "font.family": "serif"})
BLUE, ORG, GRN, RED, GREY = "#1f4e79", "#d98c3f", "#4f8f4f", "#c0504d", "#8a8a8a"
s = pd.read_csv("results/new/sim_theory.csv").groupby(["icc", "m"])[["rho_gbm", "rho_ridge", "rho_theory", "rho_theory_max"]].mean().reset_index()
fig, ax = plt.subplots(1, 2, figsize=(7.2, 3.2), gridspec_kw={"width_ratios": [1, 1.15]})
cm = {2: "#9ecae1", 5: "#6baed6", 20: "#3182bd", 64: "#08306b"}
for m, c in cm.items():
    q = s[s.m == m]; ax[0].scatter(q.rho_theory, q.rho_gbm, s=26, color=c, label=f"$m={m}$"); ax[0].scatter(q.rho_theory, q.rho_ridge, s=12, marker="x", color=GREY)
ax[0].plot([.9, 2.6], [.9, 2.6], "k--", lw=.8); ax[0].set_xlabel("Predicted $\\rho$, Eq. (6)"); ax[0].set_ylabel("Observed $\\rho$ (simulation)")
ax[0].scatter([], [], marker="x", color=GREY, s=12, label="ridge"); ax[0].legend(frameon=False, fontsize=6.5, loc="upper left"); ax[0].set_title("(a) Synthetic random-effects data", loc="left", fontsize=8.5)
r = np.corrcoef(s.rho_gbm, s.rho_theory)[0, 1]; ax[0].text(2.6, 1.0, f"$r={r:.2f}$", ha="right", fontsize=7)
b = pd.read_csv("results/new/bound_observed.csv").merge(pd.read_csv("results/new/icc_bound.csv"), on=["dataset", "target", "model"], suffixes=("_obs", "_res"))
lab = {"heating": "ENB\nheating", "cooling": "ENB\ncooling", "strength": "Concrete", "total UPDRS": "PD total", "motor UPDRS": "PD motor"}
g = b[b.model == "GBM"].set_index("target").loc[list(lab)]; rg = b[b.model == "Ridge"].set_index("target").loc[list(lab)]
x = np.arange(5); w = .27
ax[1].bar(x - w, g.rho_obs, w, color=BLUE, label="GBM observed"); ax[1].bar(x, g.rho_bound, w, color=ORG, label="GBM intercept bound, Eq. (6)"); ax[1].bar(x + w, rg.rho_obs, w, color=GREY, label="ridge observed")
for xx, v in zip(x - w, g.rho_obs): ax[1].text(xx, v + .2, f"{v:.1f}", ha="center", fontsize=6)
for xx, v in zip(x, g.rho_bound): ax[1].text(xx, v + .2, f"{v:.1f}", ha="center", fontsize=6)
ax[1].set_xticks(x); ax[1].set_xticklabels(list(lab.values()), fontsize=7); ax[1].set_ylabel("Inflation ratio $\\rho$"); ax[1].legend(frameon=False, fontsize=6.5, loc="upper right"); ax[1].set_title("(b) Real data", loc="left", fontsize=8.5)
plt.tight_layout(); plt.savefig("figures/paper/fig11_theory_validation.png", dpi=300); plt.close()
# markdown table for simulation
t = pd.read_csv("results/new/sim_theory.csv").groupby(["icc", "m"])[["rho_gbm", "rho_theory", "rho_ridge"]].mean().round(2)
rows = ["| ICC $r$ | $m=2$ | $m=5$ | $m=20$ | $m=64$ | Ridge (all $m$) |", "|---:|:---|:---|:---|:---|:---|"]
for icc in sorted(set(i for i, _ in t.index)):
    cells = [f"{t.loc[(icc, m), 'rho_gbm']:.2f} ({t.loc[(icc, m), 'rho_theory']:.2f})" for m in (2, 5, 20, 64)]
    rr = [t.loc[(icc, m), 'rho_ridge'] for m in (2, 5, 20, 64)]
    rows.append(f"| {icc:.2f} | " + " | ".join(cells) + f" | {min(rr):.2f}–{max(rr):.2f} |")
open("paper/build/_sim_table.md", "w").write("\n".join(rows)); print("\n".join(rows))
print("corr", r, "MAE", (s.rho_gbm - s.rho_theory).abs().mean(), "ridge max", s.rho_ridge.max())
