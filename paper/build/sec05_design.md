# Experimental design

## Research questions and hypotheses

The experiments are organised around five research questions (RQ), each with a hypothesis that follows from the theory in Section 3 and that could be falsified by the data.

- **RQ1 (magnitude and model dependence).** How much does random splitting inflate accuracy relative to group-held-out splitting, and does the inflation depend on the model family? *H1:* the inflation is larger for models with higher memorisation capacity $\kappa$ (tree ensembles) than for smooth models (ridge, kernel and neighbour methods), and larger for datasets with higher ICC and larger groups.
- **RQ2 (predictability).** Can the inflation be predicted from group structure? *H2:* the inflation of a flexible learner is bounded by Eq. (6) evaluated with the group-offset share of its held-out residual, and Eq. (11) gives a lower bound on the group-specific share when the bound is exceeded.
- **RQ3 (tuning leakage).** Does tuning with random inner folds degrade group-held-out performance? *H3:* yes for models whose capacity parameter controls memorisation (kernel width, neighbour count), less so for tree ensembles; the inner-error curves with random folds are minimised at higher capacity than the true held-out error.
- **RQ4 (skill and remedies).** Do models retain skill, relative to the training-mean predictor, on unseen groups? Can a few rows from a new group restore accuracy? *H4:* skill is retained where the shared signal $f$ is strong relative to group offsets (ENB2012, Concrete) and may vanish where it is weak (Parkinson's).
- **RQ5 (uncertainty and sensitivity).** Are the conclusions robust to the number of folds, to the assignment of groups to folds and to the number of training groups? *H5:* the point estimates of the grouped risk depend on the number of training groups (learning curve) but only weakly on $K$ and the fold seed; intervals are wide when $G$ is small.

## Protocols

Two families of protocols are compared, both with $K$-fold structure and both with nested hyper-parameter tuning.

1. **Random K-fold.** Rows are assigned to folds uniformly at random without regard to group. ENB2012 uses $K=12$ with 2 repeats (24 folds), Concrete $K=10$ with 2 repeats (20 folds), Parkinson's $K=10$ with 1 repeat.
2. **Group-held-out K-fold.** Whole groups are assigned to folds. ENB2012 uses leave-one-geometry-out (12 folds, one geometry per fold); Concrete uses $K=10$ group folds with 2 repeats (group-to-fold assignment shuffled between repeats); Parkinson's uses $K=10$ patient folds. For ENB2012 we also use a *three-geometry hold-out* protocol (three geometries held out at random, training on the other nine, 20 repeats) as a harder, smaller-training-set robustness check.

With $K=10$–12 the training sets of the two families contain about 90–92% of the rows, so their errors are directly comparable.

**Nested tuning.** Within each outer training set, hyper-parameters are chosen by grid search with 3-fold inner cross-validation, the model is refitted on the full outer training set with the selected values, and the outer test fold is predicted. Inner folds match the outer protocol: random for random outer folds and group-pure (`GroupKFold`) for grouped outer folds. A third arm, used only for the tuning ablation, combines a grouped outer protocol with *random* inner folds.

Algorithm 1 gives the procedure.

```
Algorithm 1: Nested group-held-out evaluation with matched inner folds
Input: data (x, y, g), model family M with grid Λ, outer folds F_1..F_K (group-pure)
for k = 1..K:
    S_k <- rows not in F_k ;  T_k <- rows in F_k
    split S_k into 3 inner folds with all rows of a group in one inner fold   (GroupKFold)
    for λ in Λ:   e(λ) <- mean inner-validation RMSE of M(λ)
    λ* <- argmin e(λ) ;  fit M(λ*) on S_k ;  predict T_k
pool predictions over k; report pooled RMSE, MAE, skill score relative to the
training-mean predictor under the same folds
```

## Models and hyper-parameters

We compare five model families (Table 5): ridge regression [@hoerl1970], $k$-nearest neighbours, support-vector regression with a radial-basis-function kernel [@cortes1995], random forests [@breiman2001] and gradient-boosted regression trees [@friedman2001]. They span a range of capacity to memorise groups: ridge regression is a low-capacity linear model, the radial-basis SVR and the neighbour method are local smoothers whose capacity is governed by a bandwidth or neighbour count, and the tree ensembles can isolate individual groups through splits on group-identifying attributes. All features are standardised for ridge, kNN and SVR; trees use raw features. The implementation is scikit-learn 1.9.1 [@pedregosa2011], with a fixed random seed (42). For ENB2012 we additionally evaluate three physics-informed variants (Section 5.5).

Table: Table 5. Models and tuning grids (nested inner 3-fold cross-validation, RMSE criterion).

| Model | Grid |
|:---|:---|
| Ridge | $\alpha \in \{0.1, 1, 10, 100\}$ |
| kNN | $k \in \{3, 9, 15\}$ |
| SVR (RBF) | $C \in \{1, 30\}$, $\gamma \in \{\text{scale}, 0.02\}$ |
| Random forest (100 trees) | max features $\in \{0.5, 1.0\}$, min. samples per leaf $\in \{1, 5\}$ |
| Gradient boosting (200 trees) | learning rate $\in \{0.05, 0.1\}$, max depth $\in \{2, 3\}$ |

The supplementary analyses that need many model fits (learning curves, feature ablations, sensitivity, calibration, ICC estimation) use fixed, reasonable hyper-parameters instead of nested tuning (ridge $\alpha=10$; SVR $C=30$, $\gamma=\text{scale}$; kNN $k=9$; random forest 100 trees with half of the features per split; gradient boosting 200 trees, learning rate 0.1, depth 3). This isolates the effect of the validation protocol from the effect of tuning, and it is stated explicitly wherever it is used.

## Metrics and uncertainty

The primary metric is the root-mean-square error (RMSE) pooled over all test rows of a protocol repetition, so that folds are weighted by size. We also report the mean absolute error (MAE), pooled out-of-fold $R^2$ and the skill score of Eq. (16) relative to the training-mean predictor computed under the same folds. The inflation ratio $\rho$ is the ratio of grouped to random RMSE. Intervals are 95% percentile bootstrap intervals [@efron1993] over folds (5,000 resamples for ENB2012, 2,000 for the other datasets); because grouped folds are group-pure, resampling folds respects the dependence between rows of one group. Folds of repeated random cross-validation are not independent, so the random-protocol intervals are optimistic, and we say so wherever they are shown. For paired comparisons of models over the twelve geometries of ENB2012 we use the Wilcoxon signed-rank test [@wilcoxon1945]. To compare model rankings between protocols across the five dataset–target combinations we use Kendall's rank correlation and the Friedman test with the post-hoc procedure recommended by Demšar [@demsar2006].

## Physics-informed variants (ENB2012 only)

To test whether domain knowledge restores generalisation across geometries, we add three variants for ENB2012. *Ridge + phys* and *GBM + phys* use an extended feature set with a volume proxy (roof area times height), absolute glazing area (glazing fraction times roof area), the surface-to-volume and wall-to-volume ratios, glazing-to-wall and glazing-to-surface ratios and a one-hot encoding of orientation. The *hybrid* model fits ridge regression to capture the global trend and gradient boosting to its residuals, so that the extrapolable part is linear and the local part is nonlinear.

## Supplementary experiments

Table 6 lists the additional experiments that address the research questions beyond the main protocol comparison. All use the group structure and fixed hyper-parameters stated above unless noted.

Table: Table 6. Supplementary experiments.

| Experiment | Purpose (RQ) | Design |
|:---|:---|:---|
| Synthetic simulation | Validate Eqs. (6)–(8) (RQ2) | Random-effects data with ICC in {0.05, 0.2, 0.4, 0.6, 0.8}, group size in {2, 5, 20, 64}, 50 groups, 3 replications; gradient boosting and ridge under random and grouped 10-fold |
| ICC and bound | Real-data check of the bound (RQ2) | One-way ANOVA ICC of group-held-out residuals; Eq. (6) bound versus observed inflation |
| Leakage fraction | Validate Eq. (12) (RQ1) | Predicted and measured fraction of test rows with a training group-mate |
| Learning curves | Effect of the number of training groups (RQ5) | Train on $g$ randomly chosen groups, test on unseen groups; random comparator with the same number of training rows |
| Concrete stratification | Effect of group size on leakage (RQ1) | Random-split error stratified by the number of other rows of the same mix |
| Tuning paths | Mechanism of tuning leakage (RQ3) | ENB2012 leave-one-geometry-out: inner random, inner grouped and true held-out error versus $\gamma$ and $k$ |
| Parkinson's feature sets | Which features carry patient identity (RQ4) | Five feature subsets, ridge, forest, boosting, random versus patient-held-out |
| Per-patient skill | Heterogeneity of skill (RQ4) | Patient-level RMSE versus the training-mean predictor |
| Calibration | Value of a few rows from a new group (RQ4) | Offset correction and weighted retraining with the first $n\in\{0,5,10,20,40\}$ recordings of a new patient, compared with the patient's own running mean |
| Sensitivity | Robustness to $K$ and fold seed (RQ5) | $K\in\{3,\dots,20\}$, 3 fold-assignment seeds |

## Computing and reproducibility

All experiments were run in a cloud container with four CPU cores (Python 3.11, scikit-learn 1.9.1, numpy 2.4.6, pandas 3.0.6, scipy 1.17.1, matplotlib 3.11.2). The main ENB2012 experiment takes about 30 minutes, the Concrete and Parkinson's experiments together about one hour, and the supplementary experiments about one hour more. Per-fold results, summary tables and figures are stored in the repository; all scripts take the random seed 42 and can be re-run from the repository root.
