"""Training-mean predictor under the same random / grouped splits: the no-skill reference for each dataset."""
import numpy as np, pandas as pd
from sklearn.model_selection import KFold, GroupKFold, LeaveOneGroupOut
SEED = 42
def mean_rmse(y, splits):
    return (sum(((y[te] - y[tr].mean()) ** 2).sum() for tr, te in splits) / len(y)) ** .5
def reps(n, g, K, REP, grouped):
    for rep in range(REP):
        if not grouped: yield list(KFold(K, shuffle=True, random_state=SEED + rep).split(np.zeros(n)))
        else:
            ug = np.unique(g); perm = dict(zip(ug, np.random.RandomState(SEED + rep).permutation(len(ug))))
            yield list(GroupKFold(K).split(np.zeros(n), groups=np.array([perm[x] for x in g])))
rows = []
p = pd.read_csv("data/parkinsons_updrs.data")
for t in ("total_UPDRS", "motor_UPDRS"):
    y = p[t].values; g = p["subject#"].values
    rows.append(dict(dataset="parkinsons", target=t, random=np.mean([mean_rmse(y, s) for s in reps(len(y), g, 10, 1, False)]),
                     grouped=np.mean([mean_rmse(y, s) for s in reps(len(y), g, 10, 1, True)])))
c = pd.read_csv("data/concrete_data.csv"); c.columns = [x.strip() for x in c.columns]
for col in c.columns: c[col] = pd.to_numeric(c[col].astype(str).str.strip())
g = pd.factorize(c.iloc[:, :7].astype(str).agg("|".join, axis=1))[0]; y = c.iloc[:, -1].values
rows.append(dict(dataset="concrete", target="strength", random=np.mean([mean_rmse(y, s) for s in reps(len(y), g, 10, 2, False)]),
                 grouped=np.mean([mean_rmse(y, s) for s in reps(len(y), g, 10, 2, True)])))
e = pd.read_csv("data/ENB2012_data.csv"); ge = pd.factorize(e[["X1", "X2", "X3", "X4", "X5"]].astype(str).agg("|".join, axis=1))[0]
for t in ("Y1", "Y2"):
    y = e[t].values
    rows.append(dict(dataset="ENB2012", target=t, random=np.mean([mean_rmse(y, s) for s in reps(len(y), ge, 12, 2, False)]),
                     grouped=mean_rmse(y, list(LeaveOneGroupOut().split(e, groups=ge)))))
pd.DataFrame(rows).round(3).to_csv("results/extra/mean_predictor_baseline.csv", index=False); print(pd.DataFrame(rows).round(2))
