# Discussion

## What random-split scores measure

The central empirical finding is that on datasets with large, tight groups, random-split accuracy mostly measures recognition of the group. The Parkinson's feature-set experiment (Section 6.3) is the sharpest illustration: three features that contain no voice information at all, namely age, sex and the number of days since recruitment, allow a random forest to predict total UPDRS with an RMSE of 0.17 under random splitting, against a target standard deviation of 10.7, and the same features give 11–14 UPDRS points of error on new patients. Age and sex identify the patient and the recruitment day places the recording on that patient's trajectory, so a model that has seen other recordings of the patient can interpolate in time. The score is real in the sense that it is reproducible under random splitting; it is meaningless as a statement about the usefulness of the features for a new patient. In ENB2012 the mechanism is the same but benign in intent: the geometry attributes identify one of twelve designs, and the model learns the load of each design as a function of orientation and glazing.

The theory in Section 3 explains the order of magnitude. A learner that can exploit group-specific structure approaches an inflation ratio of $1/\sqrt{1-r_G}$, where $r_G$ is the share of the residual variance that is group-specific; for an ICC of 0.9 that is 3.2, and for the deterministic ENB2012 data, where the within-group residual of a well-fitting model is tiny, the share approaches one and the ratio is limited only by the learner's approximation error. The synthetic simulation confirms the formulae when the group effect is a random intercept; the real data show that group-by-feature interactions can push the inflation beyond what an intercept-only account predicts (Section 6.5). In both cases the quantity that governs the inflation is not the number of rows but the fraction of the residual variance that is attributable to the group and the capacity of the learner to exploit it.

## Why rankings change

The ranking of model families changes between protocols because the families differ in how much group-specific structure they can exploit, and not in how well they capture the shared signal. Tree ensembles partition the feature space into cells and assign each cell a constant; when group-identifying attributes are available, a cell can coincide with a group, and the model recovers the group's offset and its within-group response. Smooth models, in contrast, share parameters across the whole feature space and cannot isolate a group unless the kernel is very narrow. When a new group arrives, the tree ensemble predicts it from the nearest cells, which belong to other groups; a smooth model extrapolates a trend. In ENB2012 this difference is visible in the per-geometry errors: tree ensembles predict geometry 7, the first 3.5 m building, with an error of about 10 kWh/m², whereas SVR predicts it with 0.9. Where the shared signal is weak and noisy relative to patient-specific offsets, as in Parkinson's data, none of the models extracts enough, and the ranking is essentially noise (Kendall $\tau\approx 0$), although ridge regression benefits from its small variance and ends up first.

The practical consequence is that a benchmark leaderboard computed with random splits answers the question "which model best memorises these groups?" and not "which model best predicts a new group?". Because the Friedman test finds a statistically clear ranking under random splitting (p = 0.001) and none under grouped evaluation (p = 0.66), random splitting gives an impression of resolved differences that the data cannot support.

## The no-skill reference and calibration

A group-held-out RMSE of 12 UPDRS points on Parkinson's data does not look alarming until it is compared with the 10.9 points that a constant would achieve. We therefore recommend reporting the training-mean predictor (or another simple baseline that uses no features) under the same split, and the corresponding skill score, as a standard element of any benchmark with grouped data. A model with negative skill should not be deployed for new units, however good its random-split score.

The per-patient analysis shows where the error comes from: between 89% and 94% of the squared error of the models on held-out patients is a patient-level offset, that is, the mean error over the recordings of a patient (Section 6.3). Voice measures appear to carry information about changes within a patient, but the models cannot place a new patient at the right level. A few calibration recordings with known scores do remove most of this error (Section 6.3), but the patient's own running mean over the same recordings is better than every model at every calibration size, so the gain comes from knowing the patient's level and not from the voice features. The same reasoning applies to any domain in which units differ by offsets that cannot be inferred from features: a calibration experiment is only informative if it includes a model-free reference that uses the same calibration rows.

## Tuning is part of the problem

Tuning leakage is the part of the problem that is easiest to overlook, because the final evaluation can be correct while the model that is evaluated has been selected badly. With random inner folds, support-vector regression on ENB2012 selects a kernel width that is optimal for within-geometry interpolation and not for new geometries; its held-out RMSE is then up to three times larger than with group-pure inner folds (Section 6.6). The inner-error curves in Fig. 12 show the mechanism: with random folds the inner error is minimised at the narrowest kernel and the lowest-complexity neighbour count that stays within a group, while the true held-out error is minimised at much smoother settings. Models that are insensitive to this choice, such as tree ensembles with the small grids used here, are less affected, but they pay with the higher memorisation capacity discussed above. Whenever groups exist, inner folds should be group-pure: this is a modelling decision and not just an evaluation detail.

## A practical checklist

Fig. 15 and Table 18 translate the findings into a checklist for researchers who use public tabular benchmarks or who publish new ones.

Table: Table 18. Checklist for evaluating regression models on data with possible groups.

| Step | Question | Action |
|:---|:---|:---|
| 1 | Can rows be grouped (design point, subject, site, batch, mix)? | Identify the grouping variable, or construct it from attributes that are constant within a unit; document it. |
| 2 | How strong is the dependence? | Compute the ICC of the target and of the residual of a simple model, the group-size distribution and the leaked fraction of Eq. (12). |
| 3 | What is the intended use? | If the model will predict new units, use group-held-out splits; if only new rows of known units, use random splits and state it. |
| 4 | How should the splits and the tuning be done? | Use group-pure outer and inner folds; use at least 10 groups per fold-set where possible; report $G$. |
| 5 | What is the reference? | Report the training-mean predictor under the same splits and skill scores; consider a smooth baseline (ridge). |
| 6 | How certain are the conclusions? | Bootstrap over groups; use paired tests across groups; do not rank models if intervals overlap widely. |
| 7 | Does more data help? | Plot learning curves over the number of training groups. |
| 8 | Can a new unit be calibrated? | If a few labelled rows of the new unit are available, test offset correction or weighted retraining. |

![Fig. 15. Decision flow for choosing a validation scheme for tabular regression data.](../figures/paper/fig15_flowchart.png){width=6.0in}

## Implications for building-energy modelling and clinical machine learning

For building energy, the relevant use of surrogate models for early design is the prediction of load for a design that has not been simulated. The group-held-out error of 3–4.5 kWh/m² for the best models on ENB2012 is therefore the number that matters, and it represents about 12–20% of the mean load. Importantly, even the simple models retain a clear skill relative to the training mean, so the dataset supports useful surrogate modelling, though at a lower accuracy than the random-split literature suggests. More informative benchmarks would include far more distinct geometries, which our learning curves show is what limits generalisation, not more variants of the same geometry. For clinical machine learning with repeated measures, our results reinforce earlier warnings on subject-wise validation [@saeb2017; @chaibub2019] and extend them in two ways: leakage can be so severe that the random-split score reflects only identity features, and the baseline that a constant achieves should always be reported.

For curators of benchmarks, the findings suggest publishing a group identifier and recommended group-held-out splits with each dataset, and reporting the ICC and group-size distribution in the dataset documentation. This is a small addition to the metadata and would prevent the most common form of over-optimistic comparison.

## Threats to validity

*Internal validity.* The group definitions are derived from attributes (X1–X5 in ENB2012; identical ingredient quantities in Concrete; the subject identifier in Parkinson's). If two groups that we treat as distinct are in fact near-duplicates (for instance, mixes that differ only slightly in water content), our group-held-out evaluation still leaks across them and underestimates the inflation. Hyper-parameter grids are modest and the fixed-parameter supplementary experiments do not tune; conclusions about relative model performance should not be read as statements about the best achievable performance of each family.

*Construct validity.* RMSE pooled over folds weights rows, not groups, and therefore weights large groups more; group-level averages would give different numbers in unbalanced designs such as Concrete. The skill score uses the training mean as the reference, which is the simplest option and could be replaced with a stronger baseline.

*External validity.* Three datasets, two of which are simulated or small, are not a representative sample of benchmarks. The relationship between group structure and inflation is based on five dataset–target combinations and should be treated as a hypothesis for further testing. Real measured building-energy data with explicit building and site groups (for example the Building Data Genome Project 2 [@miller2020]) would be a natural next test.

*Statistical conclusion validity.* With 12 and 42 groups, group-held-out estimates are noisy, and many model differences are not significant; we report intervals and avoid ranking claims where the evidence is weak. The random-split intervals are optimistic because folds are dependent.
