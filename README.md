# Energy-efficient

Group leakage in tabular regression benchmarks: random-split versus group-held-out evaluation, with matched nested tuning, on three public datasets (UCI Energy Efficiency ENB2012, Concrete Compressive Strength, Parkinson's Telemonitoring).

## Manuscript
- `paper/Group_leakage_tabular_regression_manuscript.docx` - journal-style manuscript for Word (native equations, line numbers, 15 figures, 18 tables). Authors, affiliations, CRediT, funding and competing-interest statements are placeholders; the generative-AI disclosure must be reviewed.
- `paper/Group_leakage_tabular_regression_preview.pdf` - PDF preview rendered with LibreOffice (fonts differ from Word).
- `paper/build/` - markdown sources of each section (`sec*.md`), reference list (`refs.py`), and the build scripts (`build_docx.py`, `postprocess.py`, `reference.docx`). Rebuild: `cd paper/build && python build_docx.py && python postprocess.py manuscript_raw.docx ../Group_leakage_tabular_regression_manuscript.docx` (needs `pypandoc_binary`, `python-docx`).
- `paper/paper.md` - earlier short draft (superseded).

## Code (`src/`)
- `common.py` - loaders and fixed-hyper-parameter models
- `run_experiments.py` (ENB2012 main protocol comparison), `ablation_inner_cv.py`, `exp_ablate_enb_knn.py` (tuning ablation), `run_extra.py` (Concrete and Parkinson's; resumable, can run per protocol/target in parallel)
- `exp_theory.py` (simulation, ICC, leakage fraction), `exp_icc.py`, `exp_bound_obs.py` (bound versus observed inflation), `exp_curves.py` (learning curves, concrete strata), `exp_parkinsons.py` (feature sets, per-patient skill, calibration), `exp_sensitivity.py`, `exp_tuning_paths.py`
- `analyze.py`, `analyze_extra.py`, `baselines.py`, `fig_structure.py`, `make_paper_figs1..6.py` - tables, statistics, figures

## Data and results
- `data/` - the three datasets; `results/` - per-fold results and summaries (`results/v1_preliminary/` is a superseded first ENB2012 run); `figures/paper/` - figures used in the manuscript.

Seed 42 throughout. Python 3.11, scikit-learn 1.9.1.
