# Introduction

Public benchmark datasets are the common currency of applied machine learning. A new model is typically judged by how much it improves a held-out error on a small number of well-known datasets, and these comparisons then shape which methods practitioners adopt, which papers are published and which claims about "state-of-the-art" accuracy circulate. The credibility of that currency rests on one assumption that is rarely stated and even more rarely checked: that the rows of the dataset are exchangeable, so that a random split into training and test parts produces a test set that is statistically independent of the training set. Many widely used tabular benchmarks violate this assumption by construction. They were assembled from designed experiments or repeated measurements, in which a modest number of underlying units (building designs, concrete mixes, patients, sites, households) each contribute many rows. Rows from the same unit share most of their information, so a random split places near-copies of the test rows in the training set. A flexible model can then reach very low error by recognising the unit rather than by learning the relationship that the benchmark is meant to probe.

This problem is a form of data leakage, that is, the use of information during training that would not be available when the model is deployed [@kaufman2012]. Leakage is now recognised as a major cause of over-optimistic and irreproducible results in machine-learning-based science: a recent survey identified leakage in 17 research fields, affecting 294 papers, in some cases with wildly inflated conclusions [@kapoor2023]. Studies in neuroimaging, genomics, ecology and digital health have documented the same mechanism under different names: pseudo-replication, subject leakage, spatial autocorrelation or hierarchical dependence [@ambroise2002; @roberts2017; @saeb2017; @rosenblatt2024; @ploton2020]. The remedy is also well known in principle, namely to split by group so that all rows of a unit fall on the same side of the split. Yet the way this remedy interacts with model choice, with hyper-parameter tuning and with the statistical uncertainty of the resulting estimates has not been quantified for the tabular benchmarks that dominate everyday practice.

## Motivating example

![Fig. 1. Random splitting versus group-held-out splitting. Colours denote groups (building geometries, mixes, patients); squares are test rows. Under a random split every test row has training rows from its own group; under a group-held-out split the test groups are absent from training.](../figures/paper/fig01_concept.png){width=6.2in}


Consider the UCI Energy Efficiency dataset (ENB2012), which maps eight building descriptors to heating and cooling loads for 768 simulated residential buildings [@tsanas2012]. It is among the most frequently used regression benchmarks in the building-energy literature, and reported accuracy has become extremely high: under random cross-validation, a tuned gradient-boosting model in our experiments reaches a root-mean-square error (RMSE) of 0.41 kWh/m² for heating load, against a standard deviation of 10.1 kWh/m². Closer inspection shows that the 768 rows are only 12 distinct building geometries, each repeated 64 times across four orientations and sixteen glazing configurations. When the same model is asked to predict the load of a geometry it has not seen, its RMSE rises 9.5-fold, to 3.86 kWh/m². A designer who wants a surrogate model for a new geometry is interested in the second number, not the first. Similar structure exists in other benchmarks: the Concrete Compressive Strength dataset contains 1,030 rows that belong to 427 distinct mixes tested at several ages [@yeh1998], and the Parkinson's Telemonitoring dataset contains 5,875 voice recordings from only 42 patients [@tsanas2010]. In the last of these, we will show that a random forest that appears nearly perfect under random splits is less accurate on a new patient than a constant prediction equal to the training mean.

## Research gap

Existing work leaves four questions open for this class of benchmarks.

1. **Magnitude and model dependence.** How large is the inflation caused by random splitting on commonly used tabular benchmarks, and is it the same for all model families? Anecdotal reports exist, but we are not aware of a controlled comparison across several model families and several datasets in which only the validation protocol differs.
2. **Predictability.** Can the inflation be predicted from the dependence structure of the data (how much of the residual variance is attributable to the unit, and how many rows each unit contributes), so that practitioners can anticipate it before running an expensive protocol?
3. **Tuning leakage.** Hyper-parameter search is itself a cross-validation procedure. If it uses random inner folds inside a grouped outer evaluation, does it select hyper-parameters that favour memorisation and thereby degrade the grouped estimate, in the same way that selection bias contaminates ordinary nested cross-validation [@cawley2010; @varma2006]?
4. **Diagnostics and remedies.** Which simple diagnostics reveal when a benchmark score carries no information about new units, and what can be done when a new unit is available for calibration?

## Contributions

This paper makes the following contributions.

- We formalise random-split and group-held-out risk under a random-effects model, derive a closed-form prediction of the inflation ratio as a function of the intraclass correlation (ICC) and the group size, derive the expected fraction of leaked test rows under K-fold cross-validation, and relate the width of group-level confidence intervals to the design effect (Section 3). We validate the formulae in a synthetic simulation.
- We re-evaluate five model families on three benchmark datasets (five dataset–target combinations) with random and group-held-out protocols and matched nested tuning, and quantify the inflation with bootstrap intervals (Section 6).
- We estimate the ICC of each dataset and test how well the theory predicts the observed inflation (Section 6.5).
- We isolate tuning leakage with an ablation in which only the inner folds change (Section 6.6).
- We introduce the training-mean predictor as a no-skill reference under the same split, and a skill score based on it, and show that on Parkinson's data all tested models have negative skill on new patients (Section 6.3).
- We provide diagnostics (learning curves over the number of training groups, leakage fraction, per-group error maps, sensitivity to the number of folds) and a calibration analysis that shows that the accuracy regained from labelled recordings of a new patient comes from knowing the patient's level and not from the features (Sections 6.2–6.4, 6.7).
- We offer a practical checklist (Section 7) and release all code, per-fold results and figures.

## Positioning

Group-aware validation is standard advice in ecology, remote sensing and clinical machine learning [@roberts2017; @meyer2018; @saeb2017; @chaibub2019], and geometry-blocked validation on ENB2012 has recently been reported as an additional diagnostic in a workflow paper on early-stage load screening [@alazazi2026]. We do not claim priority for the general idea. Our contribution is a controlled, quantitative treatment of the problem on tabular benchmarks: a theory that links the inflation to measurable properties of the data, a protocol-level comparison with matched nested tuning, a measurement of tuning leakage, and a no-skill reference that reveals when a score is uninformative. We did not read the full text of [@alazazi2026] and therefore cannot compare protocols in detail.

## Organisation of the paper

Section 2 reviews related work. Section 3 develops the theory. Section 4 describes the datasets and their group structure. Section 5 describes the experimental design. Section 6 reports results. Section 7 discusses mechanisms, recommendations and threats to validity, Section 8 lists limitations and future work, and Section 9 concludes.
