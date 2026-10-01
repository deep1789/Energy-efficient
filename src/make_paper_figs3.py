"""Figures 8 (Parkinson's skill), 10 (Concrete strata), 15 (flowchart)."""
import numpy as np, pandas as pd
import matplotlib; matplotlib.use("Agg"); import matplotlib.pyplot as plt
from matplotlib.patches import FancyBboxPatch, FancyArrowPatch
plt.rcParams.update({"font.size": 8, "axes.spines.top": False, "axes.spines.right": False, "font.family": "serif"})
BLUE, ORG, GRN, RED, GREY = "#1f4e79", "#d98c3f", "#4f8f4f", "#c0504d", "#8a8a8a"
# ---------- Fig 8 ----------
a = pd.read_csv("results/new/pk_feature_ablation.csv"); p = pd.read_csv("results/new/pk_per_patient.csv")
fig, ax = plt.subplots(1, 3, figsize=(7.4, 3.0), gridspec_kw={"width_ratios": [1.25, 1, .8]})
order = ["demo+time", "voice+demo", "all", "voice+time", "voice"]; lab = ["age, sex,\ntime", "voice,\nage, sex", "all 19", "voice,\ntime", "voice\nonly"]
r = a[a.model == "RF"].pivot(index="features", columns="protocol", values="rmse").loc[order]; x = np.arange(5); w = .38
ax[0].bar(x - w / 2, r.random, w, color=BLUE, label="random split"); ax[0].bar(x + w / 2, r.grouped, w, color=ORG, label="held-out patients")
ax[0].axhline(10.89, color="k", ls=":", lw=1); ax[0].text(4.45, 11.2, "training mean", ha="right", fontsize=6.5)
for xx, v in zip(x - w / 2, r.random): ax[0].text(xx, v + .25, f"{v:.1f}", ha="center", fontsize=6)
ax[0].set_ylim(0, 16.5); ax[0].set_xticks(x); ax[0].set_xticklabels(lab, fontsize=6.5); ax[0].set_ylabel("RMSE, total UPDRS (random forest)"); ax[0].legend(frameon=False, fontsize=6.5, loc="upper left", bbox_to_anchor=(0, 1.02)); ax[0].set_title("(a) Feature sets", loc="left", fontsize=8.5)
for m, c, mk in (("Ridge", GREY, "o"), ("GBM", BLUE, "s")): ax[1].scatter(p.baseline_rmse, p[m + "_rmse"], s=14, color=c, marker=mk, label=m, alpha=.8)
lim = [0, 31]; ax[1].plot(lim, lim, "k--", lw=.8); ax[1].set_xlim(lim); ax[1].set_ylim(lim); ax[1].set_xlabel("RMSE of training-mean predictor"); ax[1].set_ylabel("RMSE of model"); ax[1].legend(frameon=False, fontsize=6.5, loc="upper left")
ax[1].set_title("(b) Per patient", loc="left", fontsize=8.5); ax[1].text(29, 2, "model better\nbelow line", ha="right", fontsize=6)
sh = []
for m in ("Ridge", "RF", "GBM"): sh.append(((p[m + "_bias"] ** 2 * p.n).sum()) / ((p[m + "_rmse"] ** 2 * p.n).sum()))
ax[2].bar(range(3), sh, color=[GREY, "#4f8f4f", BLUE]); ax[2].set_xticks(range(3)); ax[2].set_xticklabels(["Ridge", "RF", "GBM"]); ax[2].set_ylim(0, 1.05); ax[2].set_ylabel("Share of squared error\nthat is patient-level bias")
for i, v in enumerate(sh): ax[2].text(i, v + .02, f"{v:.2f}", ha="center", fontsize=7)
ax[2].set_title("(c) Error decomposition", loc="left", fontsize=8.5); plt.tight_layout(); plt.savefig("figures/paper/fig08_parkinsons_skill.png", dpi=300); plt.close()
# ---------- Fig 10 ----------
c = pd.read_csv("results/new/concrete_stratified.csv"); bins = ["0", "1-2", "3-5", "6-10", ">10"]; n = [246, 106, 594, 37, 47]
fig, ax = plt.subplots(1, 2, figsize=(7.0, 2.9), sharey=True)
for a_, m, t in zip(ax, ["GBM", "RF"], ["(a) Gradient boosting", "(b) Random forest"]):
    q = c[c.model == m].pivot(index="mates", columns="protocol", values="rmse").loc[bins]; x = np.arange(5)
    a_.bar(x - .2, q.random, .4, color=BLUE, label="random split"); a_.bar(x + .2, q.grouped, .4, color=ORG, label="group-held-out")
    a_.set_xticks(x); a_.set_xticklabels([f"{b}\n(n={k})" for b, k in zip(bins, n)], fontsize=6.5); a_.set_xlabel("Other rows of the same mix"); a_.set_title(t, loc="left", fontsize=8.5)
ax[0].set_ylabel("RMSE (MPa)"); ax[0].legend(frameon=False, fontsize=6.5); plt.tight_layout(); plt.savefig("figures/paper/fig10_concrete_stratified.png", dpi=300); plt.close()
# ---------- Fig 15: flowchart ----------
fig, ax = plt.subplots(figsize=(7.0, 4.6)); ax.axis("off"); ax.set_xlim(0, 10); ax.set_ylim(0, 10)
def box(x, y, w, h, t, fc="#eaf1f8", ec=BLUE):
    ax.add_patch(FancyBboxPatch((x - w / 2, y - h / 2), w, h, boxstyle="round,pad=0.02,rounding_size=0.12", fc=fc, ec=ec, lw=1.1)); ax.text(x, y, t, ha="center", va="center", fontsize=7)
def arr(x1, y1, x2, y2, t=None):
    ax.add_patch(FancyArrowPatch((x1, y1), (x2, y2), arrowstyle="-|>", mutation_scale=8, lw=1, color="k"))
    if t: ax.text((x1 + x2) / 2 + .1, (y1 + y2) / 2 + .12, t, fontsize=6.5, color=RED)
box(5, 9.2, 5.2, .9, "Can rows be grouped?\n(design point, subject, site, mix, batch)")
box(2.0, 7.4, 3.2, .9, "No identifiable groups:\nrandom splits, but report\nthe assumption", fc="#f3f3f3", ec=GREY)
box(7.3, 7.4, 4.0, .9, "Estimate ICC, group sizes,\nleaked fraction (Eq. 12)", fc="#eaf1f8")
box(7.3, 5.8, 4.0, .9, "Will the model predict NEW groups?")
box(2.0, 5.8, 3.2, .9, "Only new rows of known\ngroups: random splits\nare appropriate", fc="#f3f3f3", ec=GREY)
box(7.3, 4.2, 4.2, 1.0, "Group-pure outer AND inner folds\n(nested tuning); report number of groups $G$", fc="#e6f2e6", ec=GRN)
box(7.3, 2.6, 4.2, 1.0, "Report training-mean baseline and skill score;\nbootstrap over groups; learning curve", fc="#e6f2e6", ec=GRN)
box(7.3, 1.0, 4.2, .9, "New group can be calibrated?\nTest offset correction / retraining", fc="#fdf0e0", ec=ORG)
arr(3.9, 9.0, 2.6, 7.85, "no"); arr(6.0, 8.75, 7.0, 7.85, "yes"); arr(7.3, 6.95, 7.3, 6.25); arr(5.3, 5.8, 3.6, 5.8, "no"); arr(7.3, 5.35, 7.3, 4.7, "yes"); arr(7.3, 3.7, 7.3, 3.1); arr(7.3, 2.1, 7.3, 1.45)
plt.tight_layout(); plt.savefig("figures/paper/fig15_flowchart.png", dpi=300); plt.close(); print("ok")
