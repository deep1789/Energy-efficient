"""Paper figures that depend only on finished results: 1 (concept), 3 (leakage fraction), 4 (inflation by model), 11 (tuning paths), 12 (rank reversal)."""
import numpy as np, pandas as pd
import matplotlib; matplotlib.use("Agg"); import matplotlib.pyplot as plt
from matplotlib.patches import Rectangle, FancyBboxPatch
from common import *
plt.rcParams.update({"font.size": 8, "axes.spines.top": False, "axes.spines.right": False, "font.family": "serif"})
BLUE, ORG, GRN, RED, GREY = "#1f4e79", "#d98c3f", "#4f8f4f", "#c0504d", "#8a8a8a"

# ---------- Fig 1: concept ----------
fig, ax = plt.subplots(1, 2, figsize=(7.2, 3.1))
rs = np.random.RandomState(3); G, m = 6, 8
cols = plt.cm.tab10(np.arange(G))
for k, (a, title) in enumerate(zip(ax, ["(a) Random split", "(b) Group-held-out split"])):
    a.set_xlim(-.5, G - .5); a.set_ylim(-2.2, m + 1.2); a.axis("off"); a.set_title(title, loc="left", fontsize=9)
    test_groups = {1, 4}
    for gi in range(G):
        for j in range(m):
            if k == 0: is_test = rs.rand() < .25
            else: is_test = gi in test_groups
            a.scatter(gi, j, s=70, color=cols[gi], marker="s" if is_test else "o", edgecolor="k" if is_test else "none", linewidth=1.4, zorder=3)
        a.text(gi, m + .35, f"group {gi+1}", ha="center", fontsize=7)
    a.scatter([], [], marker="o", color=GREY, s=40, label="training row"); a.scatter([], [], marker="s", color="white", edgecolor="k", s=40, label="test row")
    a.legend(loc="lower center", bbox_to_anchor=(.5, -.06), ncol=2, frameon=False, fontsize=7)
ax[0].text(2.5, -1.6, "every test row has training rows from its own group", ha="center", fontsize=7, color=RED)
ax[1].text(2.5, -1.6, "test groups are absent from training", ha="center", fontsize=7, color=GRN)
plt.tight_layout(); plt.savefig("figures/paper/fig01_concept.png", dpi=300); plt.close()

# ---------- Fig 3: leakage fraction vs K ----------
Xe, ye, ge = load_enb(); Xc, yc, gc = load_concrete(); Xp, yp, gp, _ = load_pk("all")
fig, a = plt.subplots(figsize=(4.4, 3.2)); Ks = np.arange(2, 21)
for lab, g, c in (("ENB2012", ge, BLUE), ("Concrete", gc, GRN), ("Parkinson's", gp, RED)):
    n = np.bincount(g); n = n[n > 0]
    a.plot(Ks, [(n * (1 - (1.0 / K) ** (n - 1))).sum() / n.sum() for K in Ks], color=c, lw=1.6, label=lab + " (Eq. 12)")
leak = pd.read_csv("results/new/leak_fraction.csv").dropna(subset=["measured_frac"])
for _, r in leak.iterrows():
    a.scatter(r.K, r.measured_frac, marker="x", color=dict(ENB2012=BLUE, Concrete=GRN, Parkinson=RED)[r.dataset], s=40, zorder=4)
a.scatter([], [], marker="x", color="k", label="measured"); a.set_xlabel("Number of folds $K$"); a.set_ylabel("Fraction of test rows with a\ntraining group-mate"); a.set_ylim(.6, 1.02)
a.legend(frameon=False, fontsize=7, loc="lower right"); plt.tight_layout(); plt.savefig("figures/paper/fig03_leakage_fraction.png", dpi=300); plt.close()

# ---------- Fig 4: inflation by model ----------
e = pd.read_csv("results/summary_all.csv"); e = e[e.protocol.isin(["random", "loso"])].pivot_table(index=["target", "model"], columns="protocol", values="RMSE_mean").reset_index()
e["gap"] = e.loso / e.random; x = pd.read_csv("results/extra/summary_extra.csv")
M = ["Ridge", "kNN", "SVR (RBF)", "Random forest", "Gradient boosting"]
P = [("ENB2012 heating", e[e.target == "Y1"].set_index("model").gap, BLUE), ("ENB2012 cooling", e[e.target == "Y2"].set_index("model").gap, "#6fa0d0"),
     ("Concrete strength", x[x.dataset == "concrete"].set_index("model").gap, GRN), ("Parkinson's total UPDRS", x[(x.dataset == "parkinsons") & (x.target == "total_UPDRS")].set_index("model").gap, RED),
     ("Parkinson's motor UPDRS", x[(x.dataset == "parkinsons") & (x.target == "motor_UPDRS")].set_index("model").gap, "#e39a97")]
fig, a = plt.subplots(figsize=(7.2, 3.3)); w = .8 / len(P)
for i, (lab, gser, c) in enumerate(P):
    v = [gser.get(m, np.nan) for m in M]; b = a.bar(np.arange(5) + i * w, v, w, color=c, label=lab)
    for xx, vv in zip(np.arange(5) + i * w, v): a.text(xx, vv * 1.05, f"{vv:.1f}", ha="center", fontsize=5.5)
a.axhline(1, color="k", lw=.7, ls="--"); a.set_yscale("log"); a.set_ylim(.8, 14); a.set_xticks(np.arange(5) + .4 - w / 2); a.set_xticklabels(M)
a.set_ylabel("Inflation ratio $\\rho$ (RMSE held-out groups / random split)"); a.legend(fontsize=6.5, ncol=3, frameon=False, loc="upper left")
plt.tight_layout(); plt.savefig("figures/paper/fig04_inflation.png", dpi=300); plt.close()

# ---------- Fig 11: tuning paths ----------
t = pd.read_csv("results/new/tuning_paths.csv").groupby(["family", "value"])[["outer_mse", "inner_random_mse", "inner_grouped_mse"]].mean() ** .5
fig, ax = plt.subplots(1, 2, figsize=(7.2, 3.0))
for a, fam, xl in zip(ax, ["SVR gamma", "kNN k"], ["Kernel parameter $\\gamma$ (SVR, $C=30$)", "Number of neighbours $k$ (kNN)"]):
    d = t.loc[fam]; a.plot(d.index, d.inner_random_mse, "o-", color=RED, label="Inner CV, random folds"); a.plot(d.index, d.inner_grouped_mse, "s-", color=ORG, label="Inner CV, group-pure folds")
    a.plot(d.index, d.outer_mse, "^-", color=BLUE, lw=2, label="True held-out geometry"); a.set_xscale("log"); a.set_xlabel(xl); a.set_ylabel("RMSE (kWh/m²)")
    for col, c in (("inner_random_mse", RED), ("inner_grouped_mse", ORG), ("outer_mse", BLUE)): a.scatter(d[col].idxmin(), d[col].min(), s=90, facecolor="none", edgecolor=c, linewidth=1.6, zorder=5)
ax[0].legend(fontsize=6.5, frameon=False, loc="upper right", bbox_to_anchor=(1.0, 0.98)); ax[0].set_title("(a) SVR", loc="left", fontsize=9); ax[1].set_title("(b) kNN", loc="left", fontsize=9)
plt.tight_layout(); plt.savefig("figures/paper/fig12_tuning_paths.png", dpi=300); plt.close()

# ---------- Fig 12: rank reversal ----------
rows = []
for tg, lab in (("Y1", "ENB heating"), ("Y2", "ENB cooling")):
    s = pd.read_csv("results/summary_all.csv"); s = s[(s.target == tg) & s.model.isin(M)]
    for p in ("random", "loso"): rows += [dict(block=lab, protocol="random" if p == "random" else "grouped", model=m, rmse=v) for m, v in s[s.protocol == p].set_index("model").RMSE_mean.items()]
for (ds, tg), lab in ((("concrete", "concrete_compressive_strength"), "Concrete"), (("parkinsons", "total_UPDRS"), "PD total"), (("parkinsons", "motor_UPDRS"), "PD motor")):
    s = x[(x.dataset == ds) & (x.target == tg)].set_index("model")
    for p in ("random", "grouped"): rows += [dict(block=lab, protocol=p, model=m, rmse=s.loc[m, p]) for m in M]
R = pd.DataFrame(rows); R["rank"] = R.groupby(["block", "protocol"]).rmse.rank(); R.to_csv("results/new/rank_table.csv", index=False)
fig, ax = plt.subplots(1, 5, figsize=(7.2, 2.9), sharey=True); mc = dict(zip(M, [GREY, "#9467bd", ORG, GRN, BLUE]))
for a, b in zip(ax, ["ENB heating", "ENB cooling", "Concrete", "PD total", "PD motor"]):
    d = R[R.block == b]
    for m in M:
        r1, r2 = d[(d.model == m) & (d.protocol == "random")]["rank"].iloc[0], d[(d.model == m) & (d.protocol == "grouped")]["rank"].iloc[0]
        a.plot([0, 1], [r1, r2], "o-", color=mc[m], lw=1.6, label=m)
    a.set_xticks([0, 1]); a.set_xticklabels(["random", "grouped"], fontsize=6.5); a.set_title(b, fontsize=7.5); a.set_xlim(-.3, 1.3)
ax[0].set_ylim(5.4, .6); ax[0].set_ylabel("Rank (1 = lowest RMSE)"); ax[0].set_yticks(range(1, 6)); h, l = ax[0].get_legend_handles_labels(); fig.legend(h, l, loc="lower center", ncol=5, frameon=False, fontsize=7); fig.subplots_adjust(bottom=.22)
plt.savefig("figures/paper/fig14_rank_reversal.png", dpi=300); plt.close()
import shutil; shutil.copy("figures/fig_per_shape.png", "figures/paper/fig05_per_geometry.png"); print("ok")
