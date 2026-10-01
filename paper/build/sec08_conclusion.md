# Limitations and future work

**Scope of the evidence.** The study uses three datasets and five dataset–target combinations. The relationship between group structure and inflation is established as a mechanism (theory, simulation and dose–response in Concrete) and as a pattern across the three datasets, but it is not estimated as a statistical relationship; with more datasets one could fit the inflation ratio as a function of the ICC, the group size and the model capacity. Two of the datasets are simulations or small, and none contains real measured building-energy data. Natural extensions are the Building Data Genome Project 2, in which buildings and sites are explicit groups and temporal dependence adds a second leakage channel [@miller2020], and other widely used benchmarks with repeated measures.

**Few groups.** ENB2012 has twelve geometries and Parkinson's 42 patients. Group-held-out estimates have wide intervals, and paired tests have little power; we avoid ranking claims and report the uncertainty. Simulations of the validation procedure itself, with known group-level truth, could quantify the power of group-held-out comparisons as a function of the number of groups.

**Models and tuning.** We tuned small grids for five model families. Neural networks, Gaussian processes, mixed-effects models and models with built-in extrapolation (for example linear trees or monotonic constraints) were not evaluated. Mixed-effects models and Gaussian processes with group-level kernels are particularly interesting because they model the group offsets explicitly and can use calibration rows of a new group in a principled way; our simple offset correction and weighted retraining are a first step in that direction.

**Feature sets.** For Parkinson's data we used the standard feature set and examined subsets, but did not explore alternative voice features or longitudinal models. Whether any voice feature set generalises across patients without calibration remains open; our results show only that the standard measures with the models tested do not.

**Theory.** The random-intercept model is a stylised description. The extension to group-specific functions explains why observed inflation can exceed the intercept-only bound, but a full treatment would model random slopes, non-Gaussian errors and the finite-sample behaviour of specific learners. The formulae are bounds or approximations, and we have validated them in simulation only for a gradient-boosting learner and a linear learner.

**Future work.** (i) A larger meta-analysis of benchmark datasets with a detectable group structure, with the ICC, group-size distribution and inflation recorded for each. (ii) Methods for detecting hidden groups from the data (near-duplicate detection, clustering of identical attribute combinations) so that group-held-out validation can be applied when no identifier is provided. (iii) Evaluation protocols for deployment settings in which calibration rows of a new group become available over time. (iv) Group-aware hyper-parameter search implemented as the default in common machine-learning libraries.

# Conclusions

Random train–test splits on tabular benchmarks with repeated groups measure how well a model recognises the group, and this can dominate what is reported. On three widely used datasets we quantified the effect with matched nested tuning. The inflation of random-split accuracy ranges from negligible (Concrete, where groups are small and often singletons, and ridge regression everywhere) to nearly an order of magnitude (tree ensembles on ENB2012 and Parkinson's data, where groups are large and tight). A random-effects model predicts the inflation of a learner that memorises group offsets as $1/\sqrt{1-r}$ in the ICC $r$ and reproduces it in simulation (correlation 0.93); on real data, interactions between groups and features can push the inflation beyond the intercept-only prediction, and observed ratios imply that group-specific structure accounts for more than 90% of the residual variance in ENB2012 heating load.

Four practical findings follow. First, model rankings change with the protocol: on ENB2012 and Parkinson's data the models that win under random splitting are among the worst on new groups, and the clear statistical ranking under random splitting (Friedman $p=0.001$) disappears under group-held-out evaluation ($p=0.66$). Second, tuning with random inner folds can degrade the model that is eventually deployed, with errors up to three times larger for support-vector regression on ENB2012. Third, a no-skill reference is indispensable: on Parkinson's data all models are at best no better than the training mean on new patients, although they look nearly perfect under random splitting, and the features that produce the near-perfect scores are demographic and temporal identifiers, not voice measures. Fourth, learning curves over the number of training groups show that more rows from the same groups do not help new groups; what helps is more groups.

We recommend that benchmark users identify groups before splitting, use group-pure outer and inner folds, report a no-skill baseline and uncertainty over groups, and that benchmark curators publish group identifiers and recommended splits. The code, per-fold results and figures accompanying this paper make it straightforward to apply the same checks to other datasets.

# Declarations

**CRediT authorship contribution statement.** *To be completed by the authors.* (Conceptualisation; Methodology; Software; Validation; Formal analysis; Investigation; Data curation; Writing – original draft; Writing – review and editing; Visualisation.)

**Declaration of generative AI and AI-assisted technologies in the writing process.** During the preparation of this work the author(s) used Claude (Anthropic) to assist with writing and running the analysis code, with generating figures and tables, with drafting and editing the manuscript text, and with searching for and checking bibliographic details. After using this tool, the author(s) reviewed and edited the content and take full responsibility for the content of the published article. *(Authors: please review this statement and adapt it to the target journal's policy.)*

**Declaration of competing interest.** *To be completed by the authors.*

**Funding.** *To be completed by the authors.*

**Data availability.** The datasets are public (UCI Machine Learning Repository; mirrored on Kaggle). The exact files, all code, per-fold results and figures are available at https://github.com/deep1789/Energy-efficient (repository named `Energy-efficient`).

**Acknowledgements.** *To be completed by the authors.*
