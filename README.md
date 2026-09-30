# Energy-efficient

Leakage-aware re-evaluation of machine-learning models on the UCI Energy Efficiency dataset (ENB2012).

- `paper/paper.md` - journal-style manuscript (draft; authors, affiliations and reference [5] details to be completed)
- `data/ENB2012_data.csv` - dataset (768 rows, 8 inputs, 2 targets)
- `src/run_experiments.py` - main experiments (random 12-fold, leave-one-geometry-out, 3-geometry hold-out; nested tuning)
- `src/ablation_inner_cv.py` - random vs grouped inner-fold tuning under LOSO
- `src/analyze.py` - tables, statistics, figures
- `results/` - per-fold results and summaries (`results/v1_preliminary/` is a superseded first run that used random inner folds inside grouped CV)
- `figures/` - figures used in the paper

Reproduce: `pip install pandas scikit-learn scipy matplotlib`, then run the three scripts in the order above from the repo root (about 30 minutes on 4 cores). Seed 42.
