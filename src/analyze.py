"""Tables, statistics and figures from results/folds_all.csv (written by run_experiments.py)."""
import numpy as np, pandas as pd
import matplotlib; matplotlib.use("Agg")
import matplotlib.pyplot as plt
from scipy.stats import wilcoxon

rng = np.random.RandomState(42)
r = pd.read_csv("results/folds_all.csv")
diag = pd.read_csv("results/leakage_diagnostics.csv")
oof = pd.read_csv("results/loso_oof_predictions.csv")
T = {"Y1": "Heating load", "Y2": "Cooling load"}
ORDER = ["Ridge", "kNN", "SVR (RBF)", "Random forest", "Gradient boosting",
         "Ridge + phys", "GBM + phys", "Hybrid (Ridge->GBM) + phys"]

def boot(x, n=5000):
    x = np.asarray(x); m = [x[rng.randint(0, len(x), len(x))].mean() for _ in range(n)]
    return np.percentile(m, [2.5, 97.5])

# ---- main table: RMSE mean [95% bootstrap CI over folds] per protocol ----
lines = []
for t in T:
    lines += [f"\n### {T[t]} - RMSE (kWh/m2), mean [95% CI over folds]; pooled out-of-fold R2 (LOSO) in last column\n",
              "| Model | Random 12-fold | Leave-one-shape-out | 3-shape hold-out | Gap (LOSO/random) | LOSO R2 |", "|---|---|---|---|---|---|"]
    for m in ORDER:
        cells, mean = [], {}
        for p in ("random", "loso", "shuffle"):
            v = r[(r.target == t) & (r.protocol == p) & (r.model == m)].RMSE.values
            lo, hi = boot(v); mean[p] = v.mean(); cells.append(f"{v.mean():.2f} [{lo:.2f}, {hi:.2f}]")
        o = oof[(oof.target == t) & (oof.model == m)]
        r2 = 1 - ((o.y - o.pred) ** 2).sum() / ((o.y - o.y.mean()) ** 2).sum()   # pooled out-of-fold R2
        lines.append(f"| {m} | " + " | ".join(cells) + f" | {mean['loso']/mean['random']:.1f}x | {r2:.2f} |")
open("results/table_main.md", "w").write("\n".join(lines))

# ---- paired comparisons over the 12 shapes (LOSO) ----
pair = []
for t in T:
    L = r[(r.target == t) & (r.protocol == "loso")].pivot(index="fold", columns="model", values="RMSE")
    for a, b in [("Hybrid (Ridge->GBM) + phys", "Ridge"), ("Hybrid (Ridge->GBM) + phys", "Gradient boosting"),
                 ("Hybrid (Ridge->GBM) + phys", "Ridge + phys"), ("Ridge + phys", "Ridge"),
                 ("GBM + phys", "Gradient boosting"), ("Gradient boosting", "Ridge")]:
        d = L[a] - L[b]
        pair.append(dict(target=t, a=a, b=b, mean_diff=d.mean(), a_better_in=int((d < 0).sum()), of=len(d),
                         wilcoxon_p=wilcoxon(a_ := L[a], L[b]).pvalue))
pd.DataFrame(pair).to_csv("results/paired_loso.csv", index=False)

# ---- leakage diagnostics ----
g = diag.groupby("protocol")[["nn_dist_mean", "nn_dist_median", "frac_test_shape_in_train"]].mean()
g.to_csv("results/leakage_diagnostics_summary.csv"); print(g)

# ---- figures ----
plt.rcParams.update({"font.size": 9})
# 1) optimism gap
fig, ax = plt.subplots(1, 2, figsize=(10, 3.8))
for a, t in zip(ax, T):
    s = r[r.target == t].groupby(["protocol", "model"]).RMSE.mean().unstack(0).loc[ORDER]
    x = np.arange(len(ORDER))
    a.bar(x - .27, s.random, .27, label="Random 12-fold", color="#3b6ea5")
    a.bar(x, s.loso, .27, label="Leave-one-shape-out", color="#d98c3f")
    a.bar(x + .27, s.shuffle, .27, label="3-shape hold-out", color="#8a8a8a")
    a.set_yscale("log"); a.set_xticks(x); a.set_xticklabels([m.replace(" (Ridge->GBM)", "\n(Ridge->GBM)") for m in ORDER], rotation=40, ha="right", fontsize=7)
    a.set_ylabel("RMSE (kWh/m$^2$, log scale)"); a.set_title(T[t])
ax[1].legend(fontsize=7, loc="upper left", bbox_to_anchor=(0.0, 1.0), frameon=True); ax[0].set_ylim(0.3, 8); ax[1].set_ylim(0.9, 9); plt.tight_layout(); plt.savefig("figures/fig_gap.png", dpi=220); plt.close()

# 2) per-shape error heatmap (LOSO)
fig, ax = plt.subplots(1, 2, figsize=(11, 3.6))
for a, t in zip(ax, T):
    o = oof[oof.target == t].assign(se=lambda d: (d.y - d.pred) ** 2)
    h = o.groupby(["model", "shape"]).se.mean().pow(.5).unstack().loc[ORDER]
    im = a.imshow(h.values, cmap="viridis_r", aspect="auto"); a.set_yticks(range(len(ORDER))); a.set_yticklabels(ORDER, fontsize=7)
    a.set_xticks(range(h.shape[1])); a.set_xticklabels(range(1, h.shape[1] + 1)); a.set_xlabel("Held-out building shape"); a.set_title(T[t])
    for i in range(h.shape[0]):
        for j in range(h.shape[1]):
            a.text(j, i, f"{h.values[i, j]:.1f}", ha="center", va="center", fontsize=5.5, color="w")
    plt.colorbar(im, ax=a, label="RMSE")
plt.tight_layout(); plt.savefig("figures/fig_per_shape.png", dpi=220); plt.close()

# 3) leakage diagnostic
fig, ax = plt.subplots(figsize=(4.6, 3.4))
for p, c, lab in (("random", "#3b6ea5", "Random 12-fold"), ("loso", "#d98c3f", "Leave-one-shape-out")):
    ax.hist(diag[diag.protocol == p].nn_dist_mean, bins=12, alpha=.7, color=c, label=lab)
ax.set_xlabel("Mean nearest-training-row distance of test fold\n(standardised features)"); ax.set_ylabel("Folds"); ax.legend(fontsize=7)
plt.tight_layout(); plt.savefig("figures/fig_leakage_distance.png", dpi=220); plt.close()

# 4) parity, LOSO: GBM vs hybrid
fig, ax = plt.subplots(2, 2, figsize=(7, 6))
for i, t in enumerate(T):
    for j, m in enumerate(["Gradient boosting", "Hybrid (Ridge->GBM) + phys"]):
        o = oof[(oof.target == t) & (oof.model == m)]
        ax[i, j].scatter(o.y, o.pred, s=4, c=o['shape'], cmap="tab20"); lim = [o.y.min() - 2, o.y.max() + 2]
        ax[i, j].plot(lim, lim, "k--", lw=.7); ax[i, j].set_title(f"{T[t]}: {m}", fontsize=8)
        ax[i, j].set_xlabel("Simulated (kWh/m$^2$)"); ax[i, j].set_ylabel("Predicted (leave-one-shape-out)")
plt.tight_layout(); plt.savefig("figures/fig_parity.png", dpi=220); plt.close()

# 5) correlation
d = pd.read_csv("data/ENB2012_data.csv").corr(); fig, a = plt.subplots(figsize=(5.2, 4.4))
im = a.imshow(d, cmap="RdBu_r", vmin=-1, vmax=1); a.set_xticks(range(10)); a.set_yticks(range(10)); a.set_xticklabels(d.columns); a.set_yticklabels(d.columns)
for i in range(10):
    for j in range(10): a.text(j, i, f"{d.values[i, j]:.1f}", ha="center", va="center", fontsize=6)
plt.colorbar(im); plt.tight_layout(); plt.savefig("figures/fig_correlation.png", dpi=220); plt.close()
print("analysis done")
