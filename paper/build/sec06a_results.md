# Results

## Random versus group-held-out evaluation

### Leakage fraction

Under random splitting, 100% of the test rows of ENB2012 and of Parkinson's have at least one group-mate in the training set, against 75.4% for Concrete (measured with the folds used in the experiments; Eq. (12) predicts 75.3% for $K=10$, Fig. 3). Under group-held-out splitting the fraction is zero by construction. The closed-form prediction agrees with the measured values to within 0.2 percentage points, and the curves in Fig. 3 show that the fraction depends very little on the number of folds once $K\ge 5$ (for Concrete it rises from 74.3% at $K=5$ to 75.7% at $K=20$): leakage cannot be avoided by choosing a different $K$.

![Fig. 3. Expected fraction of test rows with at least one training group-mate under random K-fold cross-validation (Eq. 12, lines) and measured with the folds used in this study (crosses).](../figures/paper/fig03_leakage_fraction.png){width=3.9in}

### ENB2012

Tables 7 and 8 report the RMSE for heating and cooling load under the three protocols. Under random 12-fold cross-validation the ordering of the models is the familiar one: gradient boosting (heating RMSE 0.41, cooling 1.25 kWh/m²) and random forest (0.47, 1.59) are best, followed by SVR, kNN and ridge regression (2.93, 3.19). Under leave-one-geometry-out evaluation the picture changes. The two tree ensembles lose a factor of 8.8–9.5 on heating load and 3.0–3.7 on cooling load, whereas ridge regression, kNN and SVR lose 1.1–1.7 (heating) and 1.1–1.6 (cooling). Support-vector regression (heating 2.95, cooling 3.31) and ridge regression (3.23, 3.50) become the most accurate models, and every model remains clearly better than the training-mean predictor (leave-one-geometry-out RMSE 10.92 for heating and 10.31 for cooling).

Table: Table 7. ENB2012, heating load: RMSE (kWh/m²), mean and 95% bootstrap interval over folds. Gap = leave-one-geometry-out RMSE / random RMSE. $R^2$ is the pooled out-of-fold coefficient of determination under leave-one-geometry-out evaluation.

| Model | Random 12-fold | Leave-one-geometry-out | 3-geometry hold-out | Gap | $R^2$ (LOSO) |
|:---|:---|:---|:---|---:|---:|
| Ridge | 2.93 [2.81, 3.06] | 3.23 [2.09, 4.74] | 4.17 [3.38, 5.05] | 1.1 | 0.84 |
| kNN | 2.21 [2.09, 2.33] | 3.66 [1.97, 5.78] | 4.28 [3.50, 5.14] | 1.7 | 0.75 |
| SVR (RBF) | 1.88 [1.72, 2.05] | **2.95** [1.82, 4.36] | 3.87 [3.36, 4.43] | 1.6 | 0.86 |
| Random forest | 0.47 [0.44, 0.50] | 4.14 [2.09, 6.51] | 4.00 [3.02, 5.10] | 8.8 | 0.68 |
| Gradient boosting | **0.41** [0.38, 0.43] | 3.86 [1.99, 5.91] | 4.40 [3.55, 5.26] | 9.5 | 0.73 |
| Ridge + phys | 2.20 [2.09, 2.31] | 3.40 [2.14, 5.01] | 3.78 [3.13, 4.51] | 1.5 | 0.82 |
| GBM + phys | 0.39 [0.37, 0.41] | 4.41 [2.40, 6.66] | 4.50 [3.63, 5.37] | 11.3 | 0.66 |
| Hybrid (ridge→GBM) + phys | 0.53 [0.49, 0.56] | 3.40 [2.05, 5.07] | **3.68** [2.97, 4.50] | 6.5 | 0.81 |

Table: Table 8. ENB2012, cooling load: RMSE (kWh/m²), same layout as Table 7.

| Model | Random 12-fold | Leave-one-geometry-out | 3-geometry hold-out | Gap | $R^2$ (LOSO) |
|:---|:---|:---|:---|---:|---:|
| Ridge | 3.19 [3.01, 3.37] | 3.50 [2.19, 5.11] | 4.62 [3.79, 5.50] | 1.1 | 0.79 |
| kNN | 2.57 [2.42, 2.72] | 4.02 [2.43, 5.98] | 4.70 [4.05, 5.44] | 1.6 | 0.70 |
| SVR (RBF) | 2.29 [2.17, 2.41] | **3.31** [2.04, 4.91] | 4.01 [3.48, 4.62] | 1.4 | 0.81 |
| Random forest | 1.59 [1.51, 1.67] | 4.82 [2.97, 6.90] | 4.91 [4.21, 5.75] | 3.0 | 0.60 |
| Gradient boosting | 1.25 [1.19, 1.30] | 4.66 [2.83, 6.81] | 4.91 [4.17, 5.66] | 3.7 | 0.62 |
| Ridge + phys | 2.69 [2.58, 2.81] | 3.57 [2.14, 5.25] | 4.16 [3.49, 4.92] | 1.3 | 0.78 |
| GBM + phys | **1.17** [1.11, 1.23] | 4.81 [2.97, 6.89] | 4.64 [3.97, 5.36] | 4.1 | 0.61 |
| Hybrid (ridge→GBM) + phys | 1.63 [1.56, 1.69] | 3.80 [2.32, 5.47] | 4.10 [3.38, 4.89] | 2.3 | 0.76 |

The intervals for the leave-one-geometry-out column are wide (for gradient boosting on heating load 1.99–5.91) because only twelve geometries contribute, and they overlap across models. Paired Wilcoxon signed-rank tests over the twelve geometries do not reach significance for any of the comparisons we examined (all $p\ge 0.34$): hybrid versus ridge, hybrid versus gradient boosting, hybrid versus ridge with physics-informed features, ridge with versus without those features, boosting with versus without them, and gradient boosting versus ridge. We therefore describe the reversal in terms of the consistent shift of point estimates and of the flexible-versus-smooth contrast, and not as a proven ordering among the best few models. The 3-geometry hold-out protocol, which trains on only nine geometries, gives larger errors for every model (3.7–4.5 kWh/m² for heating load) and compresses the differences between models further.

**Physics-informed variants.** Adding engineered features and a linear-plus-boosting hybrid, motivated by the idea that domain knowledge should help extrapolation, gave no reliable benefit. The extended ridge improved on ridge under random splitting (heating 2.20 versus 2.93) but not on unseen geometries (3.40 versus 3.23); boosting with the extended features was not better than plain boosting under leave-one-geometry-out evaluation (4.41 versus 3.86). The hybrid narrowed the gap relative to plain boosting (6.5 versus 9.5) and had the lowest three-geometry hold-out error for heating load (3.68), but its leave-one-geometry-out error was no better than that of ridge or SVR. With twelve geometries, we cannot claim that these features or the hybrid are improvements.

### Concrete

Table 9 shows the contrasting case. With 427 mixes of which 246 are singletons, random and group-held-out RMSE are nearly identical for ridge, kNN and SVR (ratios 1.0), and tree ensembles lose only about a quarter of their accuracy (1.23 and 1.24). The ranking of the models is unchanged: gradient boosting, random forest, SVR, kNN, ridge, in that order under both protocols. All models are far better than the training-mean predictor (RMSE 16.7 under both protocols), so the data support a useful predictor of strength for new mixes. The leakage that does exist is concentrated in the rows that have group-mates (Section 6.4).

Table: Table 9. Concrete compressive strength: pooled RMSE (MPa), mean and 95% bootstrap interval over folds, 10 folds × 2 repeats. Random inner tuning refers to the ablation arm with group-held-out outer folds and random inner folds.

| Model | Random 10-fold | Group-held-out (by mix) | Gap | Random inner tuning | Ratio |
|:---|:---|:---|---:|---:|---:|
| Ridge | 10.47 [10.12, 10.85] | 10.55 [10.26, 10.82] | 1.0 | 10.54 | 1.00 |
| kNN | 8.63 [8.20, 9.07] | 8.57 [8.14, 9.07] | 1.0 | see Table 16 | |
| SVR (RBF) | 6.20 [5.93, 6.47] | 6.46 [6.16, 6.72] | 1.0 | 6.46 | 1.00 |
| Random forest | 4.80 [4.54, 5.04] | 5.91 [5.56, 6.27] | 1.2 | see Table 16 | |
| Gradient boosting | 4.64 [4.32, 4.99] | 5.76 [5.43, 6.14] | 1.2 | 5.76 | 1.00 |

### Parkinson's telemonitoring

Tables 10 and 11 give the corresponding results for Parkinson's data. Under random 10-fold cross-validation a random forest reaches RMSE 1.68 on total UPDRS, against a target standard deviation of 10.7, and 1.31 on motor UPDRS (SD 8.1), which would suggest nearly exact prediction. On held-out patients the same models reach 12.73 and 9.24, increases by factors of 7.6 and 7.1. Gradient boosting loses a factor of about 3, kNN and SVR about 2 and ridge regression 1.2. Fig. 4 summarises the inflation ratios across all five dataset–target combinations: the ratio rises with model flexibility in every dataset, and it is largest where groups are large and tight.

Table: Table 10. Parkinson's telemonitoring, total UPDRS: pooled RMSE (UPDRS points), mean and 95% bootstrap interval over folds, 10 folds. The training-mean predictor has RMSE 10.70 under random and 10.89 under patient-held-out folds.

| Model | Random 10-fold | Held-out patients | Gap | Random inner tuning | Ratio |
|:---|:---|:---|---:|---:|---:|
| Ridge | 9.74 [9.55, 9.96] | 11.55 [9.46, 13.69] | 1.2 | 11.67 | 1.01 |
| kNN | 6.30 [6.12, 6.51] | 12.94 [10.53, 15.18] | 2.1 | see Table 16 | |
| SVR (RBF) | 7.26 [7.15, 7.39] | 13.34 [10.76, 15.53] | 1.8 | 13.74 | 1.03 |
| Random forest | **1.68** [1.52, 1.83] | 12.73 [10.82, 14.29] | 7.6 | see Table 16 | |
| Gradient boosting | 3.92 [3.83, 3.99] | 12.32 [10.45, 13.97] | 3.1 | 13.81 | 1.12 |

Table: Table 11. Parkinson's telemonitoring, motor UPDRS, same layout as Table 10. The training-mean predictor has RMSE 8.13 under random and 8.25 under patient-held-out folds.

| Model | Random 10-fold | Held-out patients | Gap | Random inner tuning | Ratio |
|:---|:---|:---|---:|---:|---:|
| Ridge | 7.50 [7.36, 7.66] | 8.90 [7.65, 10.18] | 1.2 | 8.96 | 1.01 |
| kNN | 4.68 [4.56, 4.80] | 9.63 [8.16, 10.96] | 2.1 | see Table 16 | |
| SVR (RBF) | 5.23 [5.17, 5.28] | 10.39 [9.00, 11.76] | 2.0 | 10.35 | 1.00 |
| Random forest | **1.31** [1.19, 1.43] | 9.24 [7.82, 10.53] | 7.1 | see Table 16 | |
| Gradient boosting | 3.10 [3.03, 3.17] | 9.41 [7.83, 10.88] | 3.0 | 10.07 | 1.07 |

![Fig. 4. Inflation ratio $\rho$ (RMSE on held-out groups divided by RMSE under random splitting) for five model families on the five dataset–target combinations. The dashed line marks no inflation. Values are labelled above the bars; the vertical axis is logarithmic.](../figures/paper/fig04_inflation.png){width=6.4in}

The most important feature of Tables 10 and 11 is the comparison with the no-skill reference. On held-out patients, *every* model has a higher pooled RMSE than the training-mean predictor: for total UPDRS 11.55–13.34 versus 10.89 (skill scores $SS$ between $-0.12$ and $-0.50$) and for motor UPDRS 8.90–10.39 versus 8.25 ($SS$ between $-0.16$ and $-0.59$). The evidence is graded. A paired comparison over the ten folds (Wilcoxon signed-rank test on fold RMSE, `results/new/pk_vs_noskill_paired.csv`) shows that SVR and kNN are significantly worse than the training mean for both targets ($p=0.004$–$0.020$; at most 2 of 10 folds better), whereas ridge regression, random forest and gradient boosting are worse on average but not significantly so ($p=0.105$–$0.625$; 2–5 of 10 folds better). No model is better than the training mean in a statistically meaningful sense. This means that, in the standard feature setup, the apparent skill under random splitting reflects recognition of patients, not information about the disease state of a new patient. We examine this further in Section 6.3.

### ENB2012 per-geometry errors

Fig. 5 shows the leave-one-geometry-out RMSE for each held-out geometry. The errors are highly heterogeneous. Geometries 8–10 (3.5 m buildings in the middle of their height group) are predicted with RMSE of about 0.4–1.1 kWh/m² by SVR and the tree ensembles, whereas geometries 4 and 5 (7 m buildings) are predicted with RMSE of 4–13 kWh/m² by all models, and geometry 7 (the first 3.5 m building) is predicted well by SVR (0.9) and ridge (2.3) but poorly by the tree ensembles (about 10). Tree ensembles are thus particularly poor at one specific type of extrapolation: they cannot predict the transition to a new height group because no leaf covers it, whereas SVR and ridge extrapolate smoothly. The per-geometry means are therefore a more informative summary than a single pooled number.

![Fig. 5. ENB2012: leave-one-geometry-out RMSE (kWh/m²) for each held-out geometry (columns) and model (rows), for heating load (left) and cooling load (right).](../figures/paper/fig05_per_geometry.png){width=6.4in}

## Learning curves and extrapolation

The learning curves in Fig. 6 plot the RMSE against the number of training groups, for group-held-out evaluation (solid) and for a random split with the same number of training rows (dashed). They give a direct view of what additional data would help.

![Fig. 6. Learning curves over the number of training groups. Solid lines: test groups unseen; dashed lines: random split with the same number of training rows; dotted line: standard deviation of the target (no-skill level). Means over 25 (ENB2012), 8 (Concrete) and 6 (Parkinson's) random draws of training and test groups.](../figures/paper/fig06_learning_curves.png){width=6.4in}

**ENB2012.** The random-split curves are flat and low (gradient boosting 0.78 to 0.42 kWh/m² for heating load as the training rows grow). The grouped curves start at the no-skill level with two training geometries (gradient boosting 9.35, ridge 12.54) and fall steeply to 5.4–6.8 with three geometries, after which they flatten or even rise: for gradient boosting the heating-load RMSE is 5.44, 5.34, 5.08, 5.30 and 6.05 for 3, 4, 6, 8 and 10 training geometries, and for SVR it rises from 6.35 to 9.36. Only ridge regression keeps improving (4.17 with ten geometries). The non-monotone behaviour reflects which geometries are held out (extrapolation to a height group not represented in training is harder) and shows that collecting more rows from the same geometries is useless for new geometries; what matters is the coverage of the geometry grid.

**Concrete.** Grouped and random curves decrease together and converge: with 380 training mixes the RMSE of gradient boosting is 4.93 (grouped) and 4.76 (random); with 50 mixes it is 8.65 and 7.70. Concrete has many small groups, so the number of training groups and the number of training rows are almost the same quantity.

**Parkinson's.** The random-split curves of the flexible models decrease gradually (gradient boosting 5.79 to 3.81 as the number of training patients grows from 4 to 36), whereas the patient-held-out curves are flat and *above* the no-skill level for almost all models and sample sizes: gradient boosting 15.4, 11.9, 14.0, 14.6, 13.2 and 12.9 for 4–36 training patients, against a target standard deviation of 10.7. Only ridge regression with 36 patients (10.1) falls just below the no-skill line. Adding patients does not rescue the voice-feature models within the range of the study.

### Distance to the training geometries

We examined whether the held-out error in ENB2012 is related to how far the held-out geometry is from the training geometries in the space of the standardised attributes $X_1$–$X_5$ (Fig. 7a). The relationship is positive but moderate: Spearman correlations between the nearest-training-geometry distance and the per-geometry RMSE are 0.36 (ridge, $p=0.25$), 0.41 (SVR, $p=0.19$), 0.49 (random forest, $p=0.10$) and 0.57 (gradient boosting, $p=0.05$). The correlation with the extremeness of the geometry's mean load is weaker still ($|r_s|\le 0.21$, all $p>0.5$). Geometry 5 illustrates why distance alone is not an adequate explanation: it is not far from its nearest neighbour (distance 0.69) but it is the geometry with the highest load, so models trained on the other geometries must extrapolate in both feature and target space. Dividing the geometries into end-of-grid (1, 6, 7, 12) and interior (2–5, 8–11) groups (Fig. 7b) shows larger mean errors at the ends for ridge (3.5 versus 3.1 kWh/m²), random forest (4.7 versus 3.9) and gradient boosting (5.0 versus 3.3) but not for SVR (2.9 versus 3.0). The differences are small relative to the spread between individual geometries, which is dominated by the hard interior geometries 4 and 5.

![Fig. 7. (a) ENB2012 leave-one-geometry-out RMSE (heating load) of each held-out geometry versus the distance to the nearest training geometry in standardised $X_1$–$X_5$ space; points of gradient boosting are labelled by geometry number. (b) Mean per-geometry RMSE for interior (pale) and end-of-grid (dark) geometries.](../figures/paper/fig07_extrapolation.png){width=6.4in}

## Concrete: leakage is concentrated in rows with group-mates

If the theory is right, leakage should affect a row only if its group has other members. Concrete lets us test this directly because the number of other rows of the same mix varies from 0 to 19. We computed out-of-fold predictions of gradient boosting, random forest and ridge regression under random and group-held-out 10-fold cross-validation (3 repeats, fixed hyper-parameters) and stratified the error by the number of other rows of the same mix (Fig. 10, Table 13).

Table: Table 13. Concrete: RMSE (MPa) of out-of-fold predictions stratified by the number of other rows of the same mix ("group-mates"), under random and group-held-out 10-fold cross-validation.

| Group-mates | $n$ rows | GBM random | GBM grouped | RF random | RF grouped | Ridge random | Ridge grouped |
|:---|---:|---:|---:|---:|---:|---:|---:|
| 0 (singleton mixes) | 246 | 5.90 | 5.74 | 5.50 | 5.33 | 9.06 | 9.08 |
| 1–2 | 106 | 3.85 | 4.14 | 3.71 | 3.99 | 8.67 | 8.67 |
| 3–5 | 594 | 3.83 | 5.84 | 4.19 | 5.85 | 10.83 | 10.93 |
| 6–10 | 37 | 3.76 | 5.44 | 4.13 | 7.55 | 11.45 | 11.56 |
| >10 | 47 | 5.83 | 6.46 | 6.39 | 8.38 | 14.48 | 14.73 |

Rows of singleton mixes show no difference between random and group-held-out evaluation (5.90 versus 5.74 for gradient boosting), as they must, because there is nothing to leak. For rows with three to ten group-mates, the random-split RMSE of gradient boosting (3.8) is about two-thirds of the group-held-out RMSE (5.4–5.8), and for random forest the difference reaches a factor of 1.8 for 6–10 group-mates (4.13 versus 7.55). Ridge regression shows no difference in any stratum. The inflation is therefore concentrated in rows that have group-mates, as the theory predicts. It is not monotone in their number: it is small for one or two group-mates (3.85 versus 4.14), large for 3–10 and small again for more than ten (5.83 versus 6.46), where the errors are high in both protocols, probably because these rows are dominated by other sources of error. Because 246 of 1,030 rows have no group-mates and 594 have only three to five, and because the within-mix variation (age) is itself strongly informative, the aggregate gap for the tree ensembles stays at about 1.2.

![Fig. 10. Concrete: RMSE of gradient boosting under random (blue) and group-held-out (orange) 10-fold cross-validation, stratified by the number of other rows of the same mix.](../figures/paper/fig10_concrete_stratified.png){width=4.6in}

## Ranking reversal across protocols

Fig. 14 compares model ranks under random and group-held-out evaluation for the five dataset–target combinations. For ENB2012 the ranking is reversed: gradient boosting, random forest, SVR, kNN and ridge under random splitting become SVR, ridge, kNN, gradient boosting, random forest for heating load and similarly for cooling load; the Kendall rank correlation between the two protocols is $-0.4$ for both targets. For Concrete the ranking is identical ($\tau=1$). For Parkinson's the rankings are essentially uncorrelated ($\tau=0.0$ for total and $0.2$ for motor UPDRS), with ridge regression moving from last place under random splitting to first place under patient-held-out evaluation for both targets.

![Fig. 14. Model ranks (1 = lowest RMSE) under random splitting and group-held-out evaluation for the five dataset–target combinations.](../figures/paper/fig14_rank_reversal.png){width=6.4in}

Across the five blocks, a Friedman test finds clear differences between the models under random evaluation ($\chi^2=18.1$, $p=0.001$; mean ranks: gradient boosting 1.4, random forest 1.6, SVR 3.4, kNN 3.6, ridge 5.0) but none under group-held-out evaluation ($\chi^2=2.4$, $p=0.66$; mean ranks: ridge 2.2, gradient boosting 2.8, SVR 3.0, random forest 3.4, kNN 3.6). Random splitting therefore produces a seemingly statistically solid ranking of model families, which disappears once the unit of generalisation is respected. With only five blocks the power of the grouped test is low, and the absence of significance does not demonstrate that the models are equal; it shows that the evidence from these benchmarks does not resolve their order.
