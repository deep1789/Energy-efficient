"""Figure 9: personalisation curve for Parkinson's (calibration recordings of a new patient)."""
import numpy as np, pandas as pd
import matplotlib; matplotlib.use("Agg"); import matplotlib.pyplot as plt
from common import *
plt.rcParams.update({"font.size": 8, "axes.spines.top": False, "axes.spines.right": False, "font.family": "serif"})
BLUE, ORG, GRN, RED, GREY = "#1f4e79", "#d98c3f", "#4f8f4f", "#c0504d", "#8a8a8a"
r = pd.read_csv("results/new/pk_personalisation.csv"); X, ys, g, d = load_pk("all"); y = ys["total_UPDRS"]
subs = sorted(r.subject.unique()); print("patients with results:", len(subs))
def pooled(q): return float(np.sqrt(q.se.sum() / q.cnt.sum()))
rows = []
for (m, v, n), q in r.groupby(["model", "variant", "n"]): rows.append(dict(model=m, variant=v, n=n, rmse=pooled(q)))
R = pd.DataFrame(rows)
# fair reference: the patient's own running mean (no model), using the same first-n recordings, evaluated on recordings 41+
ref = {}
for n in (0, 5, 10, 20, 40):
    se = cnt = 0
    base_all = r[r.model == "mean"].set_index("subject")
    for s in subs:
        i = np.where(g == s)[0]; i = i[np.argsort(d.test_time.values[i])]; ev = i[40:]
        if n == 0:
            se += base_all.loc[s, "se"]; cnt += base_all.loc[s, "cnt"]
        else:
            se += ((y[ev] - y[i[:n]].mean()) ** 2).sum(); cnt += len(ev)
    ref[n] = float(np.sqrt(se / cnt))
R.to_csv("results/new/pk_personalisation_summary.csv", index=False); pd.Series(ref).to_csv("results/new/pk_personalisation_ownmean.csv")
fig, a = plt.subplots(figsize=(4.8, 3.4)); ns = [0, 5, 10, 20, 40]
for (m, v), c, mk, lab in ((("Ridge", "offset"), GREY, "o", "Ridge + offset correction"), (("GBM", "offset"), BLUE, "s", "Gradient boosting + offset correction"), (("GBM", "retrain_w20"), ORG, "^", "Gradient boosting retrained (weight 20)")):
    q = R[(R.model == m) & (R.variant == v)].sort_values("n"); a.plot(q.n, q.rmse, marker=mk, color=c, label=lab, lw=1.5)
a.plot(ns, [ref[n] for n in ns], "k--", marker="x", label="Patient's own running mean (no model)", lw=1.2)
a.set_xlabel("Calibration recordings of the new patient ($n$)"); a.set_ylabel("RMSE on later recordings (UPDRS points)"); a.set_xticks(ns); a.legend(frameon=False, fontsize=6.5, loc="upper right")
plt.tight_layout(); plt.savefig("figures/paper/fig09_personalisation.png", dpi=300); plt.close()
print(R.round(2).to_string()); print({k: round(v, 2) for k, v in ref.items()})
