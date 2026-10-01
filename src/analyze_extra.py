"""Summaries for the two extra datasets (concrete, parkinsons) and a cross-dataset gap figure."""
import glob, numpy as np, pandas as pd
import matplotlib; matplotlib.use("Agg"); import matplotlib.pyplot as plt
rng = np.random.RandomState(42)
SD = {"concrete_compressive_strength": pd.read_csv("data/concrete_data.csv").iloc[:, -1].astype(float).std(),
      "total_UPDRS": pd.read_csv("data/parkinsons_updrs.data").total_UPDRS.std(),
      "motor_UPDRS": pd.read_csv("data/parkinsons_updrs.data").motor_UPDRS.std()}
parts = [pd.read_csv("results/extra/concrete_folds.csv")] + [pd.read_csv(f) for f in sorted(glob.glob("results/extra/parts/parkinsons__*.csv"))]
r = pd.concat(parts); r.to_csv("results/extra/all_folds.csv", index=False)
def pooled(d): return (d.sse.sum() / d.n_test.sum()) ** .5
def boot_ci(d, n=2000):
    f = [g for _, g in d.groupby(["rep", "fold"])]
    v = [pooled(pd.concat([f[i] for i in rng.randint(0, len(f), len(f))])) for _ in range(n)]
    return np.percentile(v, [2.5, 97.5])
rows = []
for (ds, t, m), d in r.groupby(["dataset", "target", "model"]):
    rec = dict(dataset=ds, target=t, model=m, sd=SD[t])
    for p in ("random", "grouped", "grouped_randominner"):
        x = d[d.protocol == p]
        if len(x):
            rec[p] = np.mean([pooled(g) for _, g in x.groupby("rep")])
            if p != "grouped_randominner": lo, hi = boot_ci(x); rec[p + "_lo"], rec[p + "_hi"] = lo, hi
    rows.append(rec)
s = pd.DataFrame(rows)
s["gap"] = s.grouped / s.random
s["tuning_ratio"] = s.grouped_randominner / s.grouped
s["nrmse_random"] = s.random / s.sd; s["nrmse_grouped"] = s.grouped / s.sd
s.round(3).to_csv("results/extra/summary_extra.csv", index=False)
pd.set_option("display.width", 220)
print(s[["dataset", "target", "model", "random", "grouped", "grouped_randominner", "gap", "tuning_ratio", "nrmse_grouped"]].round(2).to_string(index=False))
# figure: gap by dataset/model (include ENB2012 from main results)
e = pd.read_csv("results/summary_all.csv"); 
enb = e[(e.protocol.isin(["random", "loso"]))].pivot_table(index=["target", "model"], columns="protocol", values="RMSE_mean").reset_index()
enb["gap"] = enb.loso / enb.random
M = ["Ridge", "kNN", "SVR (RBF)", "Random forest", "Gradient boosting"]
panels = [("ENB2012 heating", enb[enb.target == "Y1"].set_index("model").gap), ("ENB2012 cooling", enb[enb.target == "Y2"].set_index("model").gap)] + \
         [(f"{ds} ({t.replace('concrete_compressive_strength','strength')})", s[(s.dataset == ds) & (s.target == t)].set_index("model").gap) for ds, t in s[["dataset", "target"]].drop_duplicates().itertuples(index=False)]
fig, ax = plt.subplots(figsize=(9, 3.6)); w = .8 / len(panels)
for i, (lab, g) in enumerate(panels):
    ax.bar(np.arange(len(M)) + i * w, [g.get(m, np.nan) for m in M], w, label=lab)
ax.axhline(1, c="k", lw=.7, ls="--"); ax.set_yscale("log"); ax.set_xticks(np.arange(len(M)) + .4 - w / 2); ax.set_xticklabels(M)
ax.set_ylabel("RMSE ratio: held-out groups / random split"); ax.legend(fontsize=6, ncol=2); plt.tight_layout(); plt.savefig("figures/fig_gap_datasets.png", dpi=220)
