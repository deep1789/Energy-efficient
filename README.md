# Energy-efficient

Group leakage in tabular regression benchmarks: random-split versus group-held-out evaluation, with matched nested tuning, on three public datasets (UCI Energy Efficiency ENB2012, Concrete Compressive Strength, Parkinson's Telemonitoring).

- `paper/paper.md` - journal-style manuscript (draft; authors, affiliations and references [5], [7], [8] to be completed/verified)
- `data/` - the three datasets
- `src/run_experiments.py` - ENB2012 experiments (random 12-fold, leave-one-geometry-out, 3-geometry hold-out; nested tuning; physics-informed variants)
- `src/ablation_inner_cv.py` - ENB2012: random vs group-matched inner-fold tuning
- `src/run_extra.py` - Concrete and Parkinson's (`python src/run_extra.py concrete`; for Parkinson's optionally `python src/run_extra.py parkinsons <protocol> <target>` to run parts in parallel; results resume after interruption)
- `src/analyze.py`, `src/analyze_extra.py`, `src/baselines.py` - tables, statistics, figures, training-mean baselines
- `results/` - per-fold results and summaries (`results/v1_preliminary/` is a superseded first ENB2012 run that used random inner folds inside grouped CV)
- `figures/` - figures used in the paper

Reproduce: `pip install pandas scikit-learn scipy matplotlib`, then run the scripts above from the repo root. Seed 42.
