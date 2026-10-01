::: {custom-style="Title"}
Group leakage in tabular regression benchmarks: theory, measurement and remedies
:::

::: {custom-style="Author"}
[Author names to be added]
:::

::: {custom-style="Affiliation"}
[Affiliations and corresponding author to be added]
:::

::: {custom-style="Heading Unnumbered"}
Highlights
:::

- Random splits overstate accuracy on UCI benchmarks that contain repeated groups
- Inflation follows model capacity and the group share of the residual variance
- Random inner folds in hyper-parameter tuning can triple held-out-group error
- On new Parkinson's patients, no tested model is better than the training mean
- A group-held-out checklist with a no-skill baseline is proposed

::: {custom-style="Heading Unnumbered"}
Abstract
:::

Public tabular benchmarks often contain groups of near-identical rows (repeated building geometries, concrete mixes tested at several ages, recordings from the same patient), yet models are usually evaluated with random train–test splits that place members of the same group on both sides. We study how much this inflates accuracy on three widely used datasets (UCI Energy Efficiency with 12 geometries, Concrete Compressive Strength with 427 mixes, Parkinson's Telemonitoring with 42 patients) using five model families, matched nested tuning and group-held-out validation. A random-effects model predicts the inflation of a learner that memorises group offsets as $1/\sqrt{1-r}$ for an intraclass correlation $r$; the formula reproduces a synthetic simulation (correlation 0.93), while on real data group-by-feature interactions can push the inflation beyond it. The random-split RMSE of tree ensembles is 3.0–9.5 times lower than on unseen groups for ENB2012 and 3.0–7.6 times lower for Parkinson's data, but at most 1.24 times lower for Concrete, where groups are small; ridge regression is nearly unaffected (at most 1.2). Model rankings reverse (Kendall $\tau=-0.4$ on ENB2012), and a clear ranking under random splitting (Friedman $p=0.001$) disappears under group-held-out evaluation ($p=0.66$). Tuning with random inner folds raises the held-out error of support-vector regression up to 3.3 times. On held-out Parkinson's patients no model is better than a predictor that returns the training mean, and the near-perfect random-split scores come from age, sex and time features, not from voice. We provide diagnostics, a skill score, a practical checklist and open code.

::: {custom-style="Keywords"}
**Keywords:** data leakage; grouped cross-validation; benchmark evaluation; tabular regression; intraclass correlation; nested tuning; building energy; Parkinson's telemonitoring
:::

# Introduction

Public benchmark datasets are the common currency of applied machine learning. A new model is typically judged by how much it improves a held-out error on a small number of well-known datasets, and these comparisons then shape which methods practitioners adopt, which papers are published and which claims about "state-of-the-art" accuracy circulate. The credibility of that currency rests on one assumption that is rarely stated and even more rarely checked: that the rows of the dataset are exchangeable, so that a random split into training and test parts produces a test set that is statistically independent of the training set. Many widely used tabular benchmarks violate this assumption by construction. They were assembled from designed experiments or repeated measurements, in which a modest number of underlying units (building designs, concrete mixes, patients, sites, households) each contribute many rows. Rows from the same unit share most of their information, so a random split places near-copies of the test rows in the training set. A flexible model can then reach very low error by recognising the unit rather than by learning the relationship that the benchmark is meant to probe.

This problem is a form of data leakage, that is, the use of information during training that would not be available when the model is deployed [1]. Leakage is now recognised as a major cause of over-optimistic and irreproducible results in machine-learning-based science: a recent survey identified leakage in 17 research fields, affecting 294 papers, in some cases with wildly inflated conclusions [2]. Studies in neuroimaging, genomics, ecology and digital health have documented the same mechanism under different names: pseudo-replication, subject leakage, spatial autocorrelation or hierarchical dependence [3–7]. The remedy is also well known in principle, namely to split by group so that all rows of a unit fall on the same side of the split. Yet the way this remedy interacts with model choice, with hyper-parameter tuning and with the statistical uncertainty of the resulting estimates has not been quantified for the tabular benchmarks that dominate everyday practice.

## Motivating example

![Fig. 1. Random splitting versus group-held-out splitting. Colours denote groups (building geometries, mixes, patients); squares are test rows. Under a random split every test row has training rows from its own group; under a group-held-out split the test groups are absent from training.](../figures/paper/fig01_concept.png){width=6.2in}


Consider the UCI Energy Efficiency dataset (ENB2012), which maps eight building descriptors to heating and cooling loads for 768 simulated residential buildings [8]. It is among the most frequently used regression benchmarks in the building-energy literature, and reported accuracy has become extremely high: under random cross-validation, a tuned gradient-boosting model in our experiments reaches a root-mean-square error (RMSE) of 0.41 kWh/m² for heating load, against a standard deviation of 10.1 kWh/m². Closer inspection shows that the 768 rows are only 12 distinct building geometries, each repeated 64 times across four orientations and sixteen glazing configurations. When the same model is asked to predict the load of a geometry it has not seen, its RMSE rises 9.5-fold, to 3.86 kWh/m². A designer who wants a surrogate model for a new geometry is interested in the second number, not the first. Similar structure exists in other benchmarks: the Concrete Compressive Strength dataset contains 1,030 rows that belong to 427 distinct mixes tested at several ages [9], and the Parkinson's Telemonitoring dataset contains 5,875 voice recordings from only 42 patients [10]. In the last of these, we will show that a random forest that appears nearly perfect under random splits is less accurate on a new patient than a constant prediction equal to the training mean.

## Research gap

Existing work leaves four questions open for this class of benchmarks.

1. **Magnitude and model dependence.** How large is the inflation caused by random splitting on commonly used tabular benchmarks, and is it the same for all model families? Anecdotal reports exist, but we are not aware of a controlled comparison across several model families and several datasets in which only the validation protocol differs.
2. **Predictability.** Can the inflation be predicted from the dependence structure of the data (how much of the residual variance is attributable to the unit, and how many rows each unit contributes), so that practitioners can anticipate it before running an expensive protocol?
3. **Tuning leakage.** Hyper-parameter search is itself a cross-validation procedure. If it uses random inner folds inside a grouped outer evaluation, does it select hyper-parameters that favour memorisation and thereby degrade the grouped estimate, in the same way that selection bias contaminates ordinary nested cross-validation [11, 12]?
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

Group-aware validation is standard advice in ecology, remote sensing and clinical machine learning [4, 5, 13, 14], and geometry-blocked validation on ENB2012 has recently been reported as an additional diagnostic in a workflow paper on early-stage load screening [15]. We do not claim priority for the general idea. Our contribution is a controlled, quantitative treatment of the problem on tabular benchmarks: a theory that links the inflation to measurable properties of the data, a protocol-level comparison with matched nested tuning, a measurement of tuning leakage, and a no-skill reference that reveals when a score is uninformative. We did not read the full text of [15] and therefore cannot compare protocols in detail.

## Organisation of the paper

Section 2 reviews related work. Section 3 develops the theory. Section 4 describes the datasets and their group structure. Section 5 describes the experimental design. Section 6 reports results. Section 7 discusses mechanisms, recommendations and threats to validity, Section 8 lists limitations and future work, and Section 9 concludes.

# Background and related work

## Leakage and the reproducibility of machine-learning results

Kaufman and colleagues defined leakage as the introduction of information about the target that should not legitimately be available to the model, and observed that it is usually unintentional, subtle and facilitated by the way data are collected, aggregated and prepared [1]. They proposed a formal definition and recommended a strict separation of learning and prediction. Kapoor and Narayanan surveyed the literature of machine-learning-based science and found leakage in 17 fields, collectively affecting 294 papers, and proposed a taxonomy that includes the absence of a clean train–test separation, the use of features that are proxies for the outcome non-independence between training and test examples (for instance the same patient in both sets) and test sets that are not drawn from the distribution of scientific interest [2]. Rosenblatt and colleagues examined several forms of leakage (feature selection, covariate correction, subject-level leakage through repeated measures) in connectome-based prediction and showed that they affect prediction performance to different degrees, some of them strongly [6]. A much older example is the selection bias that arises when genes are selected using the full dataset before cross-validation [3].

These works establish that leakage is common and consequential, but they mostly study classification or domain-specific pipelines and rarely relate the size of the effect to a measurable property of the data. The present paper focuses on one specific leakage mechanism, the dependence among rows of the same group, in regression benchmarks, and treats the size of the effect as a quantity that can be predicted and estimated.

## Validation under dependence

When rows are dependent, a random split no longer estimates the error that matters for new units. Roberts and colleagues reviewed cross-validation strategies for data with temporal, spatial, hierarchical or phylogenetic structure and concluded that ignoring such structure results in serious underestimation of predictive error; they recommended block cross-validation wherever dependence exists, even if no correlation structure is visible in the residuals of a fitted model [4]. Ploton and colleagues showed that spatial validation reveals poor predictive performance of large-scale ecological mapping models that appeared accurate under random validation [7]. Meyer and colleagues argued that models for spatio-temporal data should be validated with target-oriented splits and showed that this changes which predictors are selected and how well models perform [13].

In clinical machine learning, Saeb and colleagues showed that record-wise cross-validation, which allows recordings from the same subject in both training and test sets, can yield substantially optimistic estimates compared with subject-wise cross-validation, and that the approximation of the intended use-case should drive the choice of validation scheme [5]. Chaibub Neto and colleagues examined the impact of subject characteristics on machine-learning-based diagnostic applications [14]. Temporal dependence has a long history in forecasting evaluation as well, with specific recommendations for time-series cross-validation [16]. The common message of these studies is that the unit of independence must be identified and respected by the split.

Our work differs in scope and in method. We study three tabular benchmarks that are widely used for method comparison rather than domain-specific application studies, compare several model families under identical protocols, quantify the effect with a theory that can be checked against data, and examine the interaction with hyper-parameter tuning.

## Model selection bias and nested validation

Choosing hyper-parameters by cross-validation and then reporting the cross-validated error of the chosen configuration is optimistically biased, because the same data are used for selection and for evaluation. Varma and Simon demonstrated this bias and proposed nested cross-validation as a remedy [12], and Cawley and Talbot analysed over-fitting in model selection and the resulting selection bias in performance evaluation [11]. Arlot and Celisse provide a survey of cross-validation procedures, including their bias and variance properties [17]. Varoquaux showed that with small samples the error bars of cross-validated estimates are large, so that apparent differences between models are often within sampling uncertainty [18], and Bates, Hastie and Tibshirani clarified that cross-validation estimates the average error of models trained on different training sets rather than the error of the specific fitted model, and that its standard errors can be under-estimated [19].

These analyses concern random splits of independent rows. When the rows are grouped, a second issue appears: the inner splits used for tuning can leak in the same way as the outer split. If inner folds are random, the tuning criterion rewards hyper-parameters that memorise groups (for example a very small kernel bandwidth), and the selected configuration then performs badly on new groups. We show this effect empirically and explain it in Section 3.5.

## Machine learning on the three benchmark datasets

**Energy Efficiency (ENB2012).** The dataset was introduced by Tsanas and Xifara, who simulated 768 residential buildings with Ecotect, studied the association of eight inputs with heating and cooling load and compared linear regression with random forests [8]. It has since been used in a large number of studies that compare ensembles, neural networks, support-vector machines, regression splines, partial least squares and metaheuristic hybrids [20–23]. Chou and Bui report evaluation by ten-fold cross-validation [20] (we could verify this only from the available summary of the paper); for the other studies cited we did not verify the evaluation protocol and do not make claims about it. A recent workflow paper reports, among other diagnostics, geometry-blocked validation on this dataset [15].

**Concrete Compressive Strength.** The dataset, introduced by Yeh in a study of high-performance concrete with artificial neural networks, relates the quantities of seven ingredients and the age of the specimen to compressive strength [9]. Subsequent studies compare machine-learning approaches on this and related concrete data, including multi-national data collections [24]. We did not find, in the summaries of these studies that we could access, a discussion of the fact that the same mix is tested at several ages and appears repeatedly in the data.

**Parkinson's Telemonitoring.** Tsanas and colleagues collected voice recordings from 42 people with early-stage Parkinson's disease during a six-month trial and showed that motor and total UPDRS (Unified Parkinson's Disease Rating Scale) scores can be estimated from speech measures [10]. Later studies used the data for regression and for classification of severity [25]. A stacked-ensemble study of classification reported both recording-wise and subject-wise evaluation and found a noticeably lower accuracy for the subject-wise evaluation on unseen subjects (77.8% versus 84.7%) [26]. This is consistent with the mechanism studied here, although it concerns classification and a different pipeline.

## Machine learning for building energy

Reviews of data-driven building-energy prediction document a broad use of regression models, ensembles and neural networks across building types and prediction horizons [27, 28]. The Building Data Genome Project 2 collected hourly meter data from more than a thousand non-residential buildings and was used in the ASHRAE Great Energy Predictor III competition [29]. In these data, the group structure (buildings, sites) is explicit and the temporal dependence is strong, which makes the validation issue even more relevant than in the simulated ENB2012 dataset. Simulated datasets such as ENB2012 are used primarily to illustrate and compare methods for early-stage design, where the intended use is the prediction of load for a design that has not yet been simulated. This intended use corresponds to group-held-out validation.

## Summary of gaps

Table 1 summarises how the present work relates to the strands above.

Table: Table 1. Positioning of the present study relative to the main strands of related work.

| Strand | Typical focus | What is missing for tabular benchmarks | This paper |
|:---|:---|:---|:---|
| Leakage surveys [1, 2, 6] | Taxonomy, prevalence, examples | A quantitative link between data structure and effect size | Closed-form prediction and empirical test |
| Block / grouped CV [4, 7, 13] | Ecology, remote sensing | Several model families, UCI-type benchmarks | Five model families, three benchmarks |
| Subject-wise validation [5, 14, 26] | Clinical models, classification | Regression, a no-skill reference, tuning interaction | Skill score, tuning ablation |
| Nested CV and selection bias [11, 12, 17, 19] | Independent rows | Grouped inner folds | Matched inner and outer folds |
| ENB2012 / concrete / Parkinson's studies [8–10, 20] | Model accuracy | Group-held-out evaluation, inflation analysis | Group-held-out evaluation and diagnostics |

# Theory: random-split and group-held-out risk

This section develops a minimal model of dependence among rows and derives four results: the inflation of random-split accuracy as a function of the intraclass correlation and the group size (Section 3.2), the expected fraction of leaked test rows (Section 3.3), the group-level uncertainty of validation estimates (Section 3.4) and the mechanism of tuning leakage (Section 3.5). Section 3.6 defines the skill score used later. Derivations are given in Appendix A, and the formulae are checked against simulation and data in Section 6. Table 2 lists the notation.

Table: Table 2. Notation.

| Symbol | Meaning |
|:---|:---|
| $G$, $m_i$, $N$ | Number of groups; number of rows in group $i$; total number of rows |
| $(x_{ij}, y_{ij})$ | Features and target of row $j$ in group $i$ |
| $f$, $\hat f$ | Shared (generalisable) signal and its estimate |
| $u_i$, $\varepsilon_{ij}$ | Group-specific offset and row-level noise, with variances $\sigma_u^2$ and $\sigma_\varepsilon^2$ |
| $r$ | Intraclass correlation (ICC) of the residual, $r=\sigma_u^2/(\sigma_u^2+\sigma_\varepsilon^2)$ |
| $K$, $n$ | Number of folds; mean number of training rows per group in a training set |
| $R_{\mathrm{rand}}$, $R_{\mathrm{grp}}$ | Mean-squared error on test rows from seen groups and from unseen groups |
| $\rho$ | Inflation ratio $\sqrt{R_{\mathrm{grp}}/R_{\mathrm{rand}}}$ (ratio of RMSEs) |
| $\kappa$ | Memorisation capacity of a learner, $0\le\kappa\le 1$ |
| $L(K)$ | Expected fraction of test rows with at least one group-mate in training |
| $SS$ | Skill score relative to the training-mean predictor |

## Setting

Let the data consist of $G$ groups, with $m_i$ rows in group $i$ and $N=\sum_i m_i$ rows in total. We assume the random-effects structure

$$y_{ij} = f(x_{ij}) + u_i + \varepsilon_{ij}, \qquad u_i \sim (0,\sigma_u^2), \quad \varepsilon_{ij} \sim (0,\sigma_\varepsilon^2), \qquad\qquad (1)$$

where $f$ is a function that is shared across groups and can in principle be learnt from any group, $u_i$ is an offset that is specific to group $i$ and not explained by $f(x)$, and $\varepsilon_{ij}$ is noise; all random terms are independent. The features $x_{ij}$ may identify the group: in ENB2012 the attributes $X_1,\dots,X_5$ take one combination of values per geometry, in Parkinson's data age and sex are constant within a patient, and in the concrete data the ingredient quantities identify the mix. A flexible learner can therefore associate a value of $x$ with a group and, if it has seen rows of that group in training, learn the offset $u_i$. A model that sees only rows of other groups cannot.

Write $\sigma^2=\sigma_u^2+\sigma_\varepsilon^2$ for the total residual variance and

$$r = \frac{\sigma_u^2}{\sigma_u^2+\sigma_\varepsilon^2} \qquad\qquad (2)$$

for the intraclass correlation of the residual, that is, the fraction of the residual variance that is a group offset [30, 31]. Note that $r$ is defined after removing $f$; the raw ICC of the target can be much larger when $f$ varies strongly between groups.

Let $\hat h$ be a predictor trained on a training set. We distinguish two risks. The *random-split risk* $R_{\mathrm{rand}}$ is the expected squared error on a test row whose group also contributes rows to the training set, which is what a random split measures when groups are large. The *group-held-out risk* $R_{\mathrm{grp}}$ is the expected squared error on a row from a group with no training rows, which is what a grouped split measures and what a user faces when predicting a new unit. The *inflation ratio* is

$$\rho = \sqrt{R_{\mathrm{grp}}/R_{\mathrm{rand}}} \qquad\qquad (3)$$

and the *optimism* is $\Delta=R_{\mathrm{grp}}-R_{\mathrm{rand}}$. With RMSE as the reported metric, $\rho$ is the factor by which the held-out-group RMSE exceeds the random-split RMSE.

## Inflation as a function of ICC and group size

**Idealised memorising learner.** Suppose the learner knows $f$ and estimates the offset of a seen group from its $n$ training rows by the best linear unbiased predictor (BLUP), that is, by shrinking the mean training residual $\bar e_i$ towards zero [32, 33]:

$$\hat u_i = \frac{n\sigma_u^2}{n\sigma_u^2+\sigma_\varepsilon^2}\,\bar e_i . \qquad\qquad (4)$$

The posterior variance of the offset given $n$ rows is

$$V(n) = \operatorname{Var}(u_i - \hat u_i) = \left(\frac{1}{\sigma_u^2}+\frac{n}{\sigma_\varepsilon^2}\right)^{-1} = \frac{\sigma_u^2\sigma_\varepsilon^2}{\sigma_\varepsilon^2+n\sigma_u^2}. \qquad\qquad (5)$$

For a test row of a seen group, the prediction error is $\varepsilon_{ij}+(u_i-\hat u_i)$ and $R_{\mathrm{rand}}=\sigma_\varepsilon^2+V(n)$. For a test row of an unseen group the best prediction is $f(x)$, whose error is $u_i+\varepsilon_{ij}$, so $R_{\mathrm{grp}}=\sigma^2$. Normalising $\sigma^2=1$ so that $\sigma_u^2=r$ and $\sigma_\varepsilon^2=1-r$,

$$\rho^2 = \frac{1}{(1-r)+V(n)}, \qquad V(n)=\frac{r(1-r)}{(1-r)+nr}. \qquad\qquad (6)$$

Two limits are informative. When groups are tiny ($n\to 0$), $V\to r$ and $\rho\to 1$: nothing can be memorised. When groups are large ($n\to\infty$), $V\to 0$ and

$$\rho_{\max}=\frac{1}{\sqrt{1-r}} . \qquad\qquad (7)$$

The inflation ratio therefore cannot exceed $1/\sqrt{1-r}$ for this learner, and it is attained when the group offset is learnt exactly. An ICC of 0.5 gives $\rho_{\max}\approx 1.41$, 0.9 gives 3.2 and 0.99 gives 10. A near-order-of-magnitude inflation requires that almost all residual variance is a group offset, which is the situation in strongly clustered designs such as ENB2012, where the 64 rows of a geometry differ only through orientation and glazing and the geometry determines most of the load.

**Learners of limited capacity.** Real learners do not recover the offset perfectly. Let a learner predict $f(x)+\kappa\hat u_i$ for a seen group, where $\kappa\in[0,1]$ measures how much of the BLUP it captures; $\kappa=0$ is a model that cannot isolate groups (for example a low-order linear model) and $\kappa=1$ is the idealised learner above. Because $u_i-\hat u_i$ is uncorrelated with $\hat u_i$ and $\operatorname{Var}(\hat u_i)=\sigma_u^2-V(n)$,

$$R_{\mathrm{rand}}(\kappa)=\sigma_\varepsilon^2+V(n)+(1-\kappa)^2\bigl(\sigma_u^2-V(n)\bigr), \qquad\qquad (8)$$

and $R_{\mathrm{grp}}=\sigma^2$ is unchanged, since a new group offers nothing to memorise. The corresponding inflation is $\rho^2(\kappa)=\sigma^2/R_{\mathrm{rand}}(\kappa)$, which equals one for $\kappa=0$ and increases monotonically with $\kappa$. This captures the empirical pattern that ridge regression is nearly unaffected while tree ensembles are strongly affected: they differ in $\kappa$, not in the data.

**Approximation error.** If the learner estimates $f$ with error that is similar in both protocols, with mean-squared value $\tau^2$, both risks increase by $\tau^2$ and

$$\rho^2=\frac{\sigma^2+\tau^2}{R_{\mathrm{rand}}(\kappa)+\tau^2}, \qquad\qquad (9)$$

which is closer to one. Approximation error therefore dampens the inflation, and it explains why the ratio is smaller for targets that are hard to model even without leakage.

**Group-specific functions.** Random intercepts are the simplest form of group dependence. In practice groups differ also in how the target responds to within-group features (the effect of glazing area on the load of a building depends on its geometry; the response of a speech measure to disease progression differs between patients). Replace $u_i$ in Eq. (1) by a group-specific function $g_i(x_{ij})$ with mean-squared value $\sigma_G^2$; the random intercept is the special case $g_i\equiv u_i$. A learner with enough capacity and enough rows per group can estimate $g_i$ for a seen group with posterior mean-squared error $V_G(n)\le\sigma_G^2$, so that

$$\rho^2=\frac{\sigma_G^2+\sigma_\varepsilon^2}{V_G(n)+\sigma_\varepsilon^2}\ \le\ 1+\frac{\sigma_G^2}{\sigma_\varepsilon^2}=\frac{1}{1-r_G}, \qquad r_G=\frac{\sigma_G^2}{\sigma_G^2+\sigma_\varepsilon^2}. \qquad\qquad (10)$$

The group-specific share $r_G$ is at least the intercept-only ICC $r$ and is not identified by the ICC alone. The inequality can be turned around: since $R_{\mathrm{rand}}\ge\sigma_\varepsilon^2$ for any learner and, if the risk on unseen groups equals $\sigma_G^2+\sigma_\varepsilon^2$, an observed inflation $\rho$ implies

$$r_G\ \ge\ 1-\frac{1}{\rho^2}. \qquad\qquad (11)$$

(Any additional approximation error that affects only unseen groups enters the bound as well, so $r_G$ is best read as the share of the held-out risk that is group-specific.) For a deterministic simulation such as ENB2012, $\sigma_\varepsilon^2=0$ and the bound in Eq. (10) is infinite, so the inflation is limited only by the learner's approximation error; in noisy data such as voice recordings it is limited by the measurement noise. We use Eq. (11) in Section 6.5 to infer a lower bound on the group-specific variance share from observed inflation ratios, and compare it with the intercept-only ICC.

**Operational form.** In practice $\sigma_u^2$ and $\sigma_\varepsilon^2$ are not known, but they can be estimated from the group-held-out residuals $e_{ij}$ of any model: $\hat\sigma_u^2$ is the mean squared group-mean residual and $\hat\sigma_\varepsilon^2$ is the remaining within-group energy. Substituting them into Eq. (6) yields the inflation that a learner could achieve by exploiting group *offsets*. Section 6.5 compares this value with observed ratios; because real groups also differ in their responses to within-group features, observed ratios can exceed it, in which case Eq. (11) gives the implied group-specific share.

## Leakage fraction under K-fold cross-validation

Under random $K$-fold cross-validation each row is placed in the test fold with probability $1/K$. A test row from a group of size $m$ has no group-mate in the training set only if all other $m-1$ members are also in the test fold, which has probability $K^{-(m-1)}$ when assignments are independent. The expected fraction of test rows with at least one training group-mate is therefore

$$L(K)=\frac{1}{N}\sum_{i=1}^{G} m_i\Bigl(1-K^{-(m_i-1)}\Bigr), \qquad\qquad (12)$$

and the expected number of training group-mates of a test row is $(m_i-1)(K-1)/K$, so that the average number of training rows per group in a training set is about $n\approx m(K-1)/K$. Eq. (12) shows that random splitting leaks almost completely whenever groups have more than a handful of rows: for $K=10$ and $m=5$ the leaked fraction is already $1-10^{-4}$. The fraction is small only when many groups are singletons ($m_i=1$ contributes zero) or very small. Grouped splitting makes $L=0$ by construction.

## Group-level uncertainty

A grouped validation estimate is an average over a limited number $G$ of independent units. If $\bar{\ell}_i$ denotes the mean loss of group $i$ and $s_G^2$ is its variance across groups, the standard error of the group-held-out risk is approximately

$$\operatorname{SE}(\hat R_{\mathrm{grp}})\approx \frac{s_G}{\sqrt{G}}, \qquad\qquad (13)$$

and a $95\%$ interval for the mean has relative half-width $t_{G-1,0.975}\,\mathrm{CV}_G/\sqrt{G}$, where $\mathrm{CV}_G=s_G/\bar\ell$. For $G=12$ the multiplier $t_{11,0.975}/\sqrt{12}=0.64$ is large, for $G=42$ it is $0.31$ and for $G=427$ it is $0.095$. The information content of $N$ clustered rows is correspondingly smaller than $N$: with average group size $\bar m$ and ICC $\varrho_\ell$ of the loss, the effective sample size is

$$n_{\mathrm{eff}}=\frac{N}{1+(\bar m-1)\varrho_\ell}, \qquad\qquad (14)$$

the design effect of Kish [34]. For $\bar m=64$ and even a modest $\varrho_\ell=0.1$ the effective sample size is $N/7.3\approx 105$ rather than 768. Random-split intervals that treat all rows or folds as independent therefore understate uncertainty about performance on new groups, and differences between models that look clear under random splitting may be unresolved under grouped splitting. We bootstrap over groups (or over folds, which are group-pure in the grouped protocol) to respect this dependence [35].

## Tuning leakage

Hyper-parameters are chosen by minimising an inner cross-validation error. Let $\lambda$ index model capacity (for instance the number of neighbours $k$ or the inverse kernel width $\gamma$), $\hat\lambda_{\mathrm{rand}}=\arg\min_\lambda \hat R^{\mathrm{inner}}_{\mathrm{rand}}(\lambda)$ be the selection with random inner folds and $\hat\lambda_{\mathrm{grp}}$ the selection with group-pure inner folds. The *tuning regret* is

$$\Delta_{\mathrm{tune}}=R_{\mathrm{grp}}(\hat\lambda_{\mathrm{rand}})-R_{\mathrm{grp}}(\hat\lambda_{\mathrm{grp}})\ \ge\ 0 . \qquad\qquad (15)$$

The mechanism is easiest to see for $k$-nearest-neighbour regression under Eq. (1) with group-identifying features. With random inner folds, the $k$ nearest neighbours of a validation row are rows of its own group as long as $k$ does not exceed the group's training size, so they share the offset $u_i$, which cancels in the prediction error, and the error is $\sigma_\varepsilon^2(1+1/k)$ plus a bias that grows with the spread of $f$ within the group. The criterion therefore favours the largest $k$ that stays within a group. With group-pure inner folds, the neighbours belong to other groups, the offsets do not cancel, the error is of the form $\sigma_u^2(1+1/k_g)+\sigma_\varepsilon^2(1+1/k)$ plus a bias that grows with the distance between groups, and the criterion favours values of $k$ that average over several groups. In other words, random inner folds measure a capacity trade-off at the *within-group* scale and grouped inner folds at the *between-group* scale. For kernel methods the same logic implies that random inner folds prefer narrow kernels (large $\gamma$) that resolve individual groups, which is the choice that generalises worst to new groups. Section 6.6 documents this empirically, including the full inner-error curves.

Regret (15) is not a bias of the final estimate alone; it is a loss of actual predictive quality, because the deployed model uses $\hat\lambda_{\mathrm{rand}}$. Using group-pure inner folds is therefore a modelling decision, not only an evaluation decision.

## Skill score

A group-held-out RMSE is hard to interpret without a reference. A natural no-skill reference is the training-mean predictor, which uses no features. Let $\mathrm{MSE}_{\mathrm{mean}}$ be its error under the same split and define the skill score

$$SS = 1-\frac{\mathrm{MSE}_{\mathrm{model}}}{\mathrm{MSE}_{\mathrm{mean}}} . \qquad\qquad (16)$$

$SS=1$ is a perfect prediction, $SS=0$ is no better than the training mean and $SS<0$ is worse. The skill score differs from the coefficient of determination $R^2$, which uses the test-set mean and is therefore not a no-skill baseline when the test and training groups differ. Under group-held-out splitting, $\mathrm{MSE}_{\mathrm{mean}}=\operatorname{Var}(y)+\operatorname{Var}(\bar y_{\mathrm{train}})$ and the second term is small, approximately $\operatorname{Var}(y)\bigl(\varrho_y/G_{\mathrm{train}}+(1-\varrho_y)/N_{\mathrm{train}}\bigr)$ with $\varrho_y$ the ICC of the target, so that the reference is essentially the marginal variance of the target. We report $SS$ and the corresponding RMSE ratio $\sqrt{\mathrm{MSE}_{\mathrm{model}}/\mathrm{MSE}_{\mathrm{mean}}}$ throughout, because a model that is worse than the training mean on new groups carries no usable information for them regardless of how good it looks under random splitting.

## Learning curves over the number of training groups

For a smooth shared signal $f$ estimated from $g$ training groups, we expect the group-held-out risk to decrease towards a floor as $g$ grows, $R_{\mathrm{grp}}(g)\approx R_\infty+b\,g^{-c}$, with the rate set by how well the groups cover the feature space; held-out groups at the edge of the design are extrapolations and decrease more slowly. The random-split risk, in contrast, depends mainly on the number of rows per group and is nearly flat in $g$. Plotting both against the number of training groups (Section 6.2) shows whether additional groups, rather than additional rows, would improve generalisation, which is a practical use of the framework: collecting more rows from the same groups does not help new groups.

# Datasets and group structure

We use three public datasets that are frequently used to compare regression methods and that were obtained from designed experiments or repeated measurements. All three are available from the UCI Machine Learning Repository and mirrored on Kaggle; the exact files used are in the `data/` folder of the accompanying repository. Table 3 summarises their size, targets and group structure, and Fig. 2 visualises the grouping.

Table: Table 3. Datasets, targets and group structure. $G$ is the number of groups; $m$ is the number of rows per group; ICC is the intraclass correlation of the target (one-way ANOVA estimator, unbalanced design). SD is the standard deviation of the target.

| Dataset | Rows | Features | Target (unit) | Mean ± SD | $G$ | $m$ (min / median / max) | ICC of target |
|:---|---:|---:|:---|:---|---:|:---|---:|
| ENB2012 | 768 | 8 | Heating load (kWh/m²) | 22.31 ± 10.09 | 12 | 64 / 64 / 64 | 0.914 |
| ENB2012 | 768 | 8 | Cooling load (kWh/m²) | 24.59 ± 9.51 | 12 | 64 / 64 / 64 | 0.927 |
| Concrete | 1,030 | 8 | Compressive strength (MPa) | 35.82 ± 16.71 | 427 | 1 / 1 / 20 | 0.406 |
| Parkinson's | 5,875 | 19 | Total UPDRS (points) | 29.02 ± 10.70 | 42 | 101 / 141 / 168 | 0.935 |
| Parkinson's | 5,875 | 19 | Motor UPDRS (points) | 21.30 ± 8.13 | 42 | 101 / 141 / 168 | 0.919 |

![Fig. 2. Group structure of the three datasets. (a) ENB2012 heating load by building geometry; each point is one of 64 variants and the horizontal line marks the geometry mean. (b) Concrete: histogram of the number of rows per mix; 246 of 427 mixes occur once. (c) Parkinson's: distribution of total UPDRS per patient, sorted by patient mean. (d) Intraclass correlation of each target with respect to its group variable.](../figures/paper/fig02_structure.png){width=6.3in}

## ENB2012: simulated building geometries

The Energy Efficiency dataset contains 768 buildings simulated in Ecotect, all with the same volume (771.75 m³), materials and location, and differing in layout, glazing and orientation [8]. The eight inputs are relative compactness ($X_1$), surface area ($X_2$), wall area ($X_3$), roof area ($X_4$), overall height ($X_5$), orientation ($X_6$, four values), glazing area ($X_7$, four values: 0, 0.10, 0.25, 0.40 of floor area) and glazing-area distribution ($X_8$, six values). The two targets are the heating load $Y_1$ and cooling load $Y_2$. There are no missing values and no duplicate rows. Attributes $X_1$–$X_5$ take exactly twelve combinations, each shared by 64 rows (four orientations times sixteen glazing configurations); we call each combination a *geometry* (Table 4). Six geometries are 7 m high and six are 3.5 m high. The between-geometry share of the target variance is 0.91 (heating) and 0.93 (cooling). Because the data are produced by a deterministic simulation, there is no measurement noise: all variation within a geometry is explained by orientation and glazing.

Table: Table 4. The twelve ENB2012 geometries and their mean loads (kWh/m²) over the 64 variants.

| Geometry | $X_1$ | $X_2$ | $X_3$ | $X_4$ | $X_5$ | Heating | Cooling |
|---:|---:|---:|---:|---:|---:|---:|---:|
| 1 | 0.98 | 514.5 | 294.0 | 110.25 | 7.0 | 27.65 | 29.22 |
| 2 | 0.90 | 563.5 | 318.5 | 122.50 | 7.0 | 31.63 | 33.82 |
| 3 | 0.86 | 588.0 | 294.0 | 147.00 | 7.0 | 28.55 | 30.91 |
| 4 | 0.82 | 612.5 | 318.5 | 147.00 | 7.0 | 25.56 | 28.03 |
| 5 | 0.79 | 637.0 | 343.0 | 147.00 | 7.0 | 38.61 | 40.24 |
| 6 | 0.76 | 661.5 | 416.5 | 122.50 | 7.0 | 35.66 | 36.41 |
| 7 | 0.74 | 686.0 | 245.0 | 220.50 | 3.5 | 11.89 | 14.81 |
| 8 | 0.71 | 710.5 | 269.5 | 220.50 | 3.5 | 12.04 | 15.04 |
| 9 | 0.69 | 735.0 | 294.0 | 220.50 | 3.5 | 12.39 | 15.24 |
| 10 | 0.66 | 759.5 | 318.5 | 220.50 | 3.5 | 12.82 | 15.87 |
| 11 | 0.64 | 784.0 | 343.0 | 220.50 | 3.5 | 16.62 | 20.23 |
| 12 | 0.62 | 808.5 | 367.5 | 220.50 | 3.5 | 14.28 | 15.24 |

The geometry grid has a one-dimensional backbone: relative compactness decreases and surface area increases monotonically from geometry 1 to 12, while height changes once, between geometries 6 and 7. A held-out geometry is therefore an interpolation if it lies between two training geometries of the same height and an extrapolation if it lies at an end of the grid (geometries 1, 6, 7 and 12) or if the height group is represented by few training geometries. We use this structure in Section 6.2.

## Concrete compressive strength

The dataset of Yeh contains 1,030 concrete samples with the quantities (kg/m³) of cement, blast-furnace slag, fly ash, water, superplasticizer, coarse aggregate and fine aggregate, the age of the specimen (1–365 days) and its compressive strength in MPa [9]. There are no missing values; 25 rows are exact duplicates. Rows that share all seven ingredient quantities belong to the same mix and differ in age and strength; this grouping gives 427 mixes with a median size of 1 and a maximum of 20 rows. Of these, 246 mixes occur once, and 784 of the 1,030 rows belong to mixes that occur more than once (Fig. 2b). The between-mix share of strength variance is 0.41, much lower than in the other datasets, because strength depends strongly on age within a mix and on composition across mixes.

## Parkinson's telemonitoring

The Parkinson's Telemonitoring dataset contains 5,875 voice recordings from 42 people with early-stage Parkinson's disease in a six-month home-monitoring trial [10]. Each row contains the subject identifier, age, sex, the time since recruitment in days (`test_time`), the motor and total UPDRS scores [36] interpolated to the recording date, and 16 voice measures (several variants of jitter and shimmer, noise-to-harmonics and harmonics-to-noise ratios, and the nonlinear measures RPDE, DFA and PPE). Patients contribute between 101 and 168 recordings. There are no missing values and no duplicate rows. Age and sex are constant within a patient and therefore identify the patient; `test_time` places a recording on the patient's disease trajectory. Between-patient differences account for 0.93 (total UPDRS) and 0.92 (motor UPDRS) of the variance (Fig. 2c, d). Unless stated otherwise we use all 19 features, which is the standard setup; Section 6.3 examines feature subsets.

## What the group structure implies

The three datasets span a range of dependence. ENB2012 has few, large and deterministic groups with an extremely high ICC; Parkinson's has few, large, noisy groups with an equally high ICC; Concrete has many small groups, nearly half of them singletons, and a moderate ICC. According to the theory in Section 3, we therefore expect severe inflation for flexible learners on the first two datasets and a mild effect on the third. The leaked fraction under random splitting is 100%, 75% and 100% for the three datasets (Section 6.1 and Eq. (12)).

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

We compare five model families (Table 5): ridge regression [37], $k$-nearest neighbours, support-vector regression with a radial-basis-function kernel [38], random forests [39] and gradient-boosted regression trees [40]. They span a range of capacity to memorise groups: ridge regression is a low-capacity linear model, the radial-basis SVR and the neighbour method are local smoothers whose capacity is governed by a bandwidth or neighbour count, and the tree ensembles can isolate individual groups through splits on group-identifying attributes. All features are standardised for ridge, kNN and SVR; trees use raw features. The implementation is scikit-learn 1.9.1 [41], with a fixed random seed (42). For ENB2012 we additionally evaluate three physics-informed variants (Section 5.5).

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

The primary metric is the root-mean-square error (RMSE) pooled over all test rows of a protocol repetition, so that folds are weighted by size. We also report the mean absolute error (MAE), pooled out-of-fold $R^2$ and the skill score of Eq. (16) relative to the training-mean predictor computed under the same folds. The inflation ratio $\rho$ is the ratio of grouped to random RMSE. Intervals are 95% percentile bootstrap intervals [35] over folds (5,000 resamples for ENB2012, 2,000 for the other datasets); because grouped folds are group-pure, resampling folds respects the dependence between rows of one group. Folds of repeated random cross-validation are not independent, so the random-protocol intervals are optimistic, and we say so wherever they are shown. For paired comparisons of models over the twelve geometries of ENB2012 we use the Wilcoxon signed-rank test [42]. To compare model rankings between protocols across the five dataset–target combinations we use Kendall's rank correlation and the Friedman test with the post-hoc procedure recommended by Demšar [43].

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

## Parkinson's telemonitoring: what carries the apparent skill

### Feature sets

Section 6.1 showed that no model beats the training mean on new patients. To find out which features generate the near-perfect random-split scores, we repeated the evaluation with five feature subsets: the three demographic and temporal attributes alone (age, sex and `test_time`), the 16 voice measures alone, and combinations (Table 12, Fig. 8a). The training-mean predictor has RMSE 10.70 under random and 10.89 under patient-held-out splitting.

Table: Table 12. Parkinson's telemonitoring, total UPDRS: RMSE for different feature sets under random 10-fold and patient-held-out 10-fold evaluation, fixed hyper-parameters. The number of features is given in parentheses.

| Features | Ridge random | Ridge held-out | Random forest random | Random forest held-out | Gradient boosting random | Gradient boosting held-out |
|:---|---:|---:|---:|---:|---:|---:|
| Age, sex, time (3) | 10.11 | 11.19 | 0.17 | 12.44 | 4.14 | 13.80 |
| Voice, age, sex (18) | 9.78 | 11.79 | 3.59 | 12.85 | 4.25 | 13.77 |
| All 19 features (19) | 9.74 | 11.79 | 2.96 | 12.82 | 3.92 | 13.90 |
| Voice, time (17) | 10.13 | 11.27 | 8.14 | 11.59 | 8.55 | 11.69 |
| Voice only (16) | 10.17 | 11.28 | 8.37 | 11.69 | 8.99 | 11.84 |

The three attributes that contain no voice information, age, sex and `test_time`, are sufficient for a random forest to reach an RMSE of 0.17 under random splitting, which corresponds to a coefficient of determination of 0.9997, and on new patients the same model has an RMSE of 12.44, worse than the training mean. These attributes identify the patient (age and sex) and locate the recording on that patient's trajectory (time), so that a model with other recordings of the same patient in the training set can interpolate in time. When the 16 voice measures are used alone, the random-split RMSE is 8.4 for the random forest and 9.0 for gradient boosting, a modest skill relative to the training mean (skill scores 0.39 and 0.29 for the forest and boosting) which is plausible for noisy speech features, and the inflation is small (1.4 and 1.3), because voice measures do not identify patients as precisely. On held-out patients, voice-only models are again no better than the training mean (11.7–11.8 versus 10.9 for the forest and boosting, 11.3 for ridge regression). Adding demographic and temporal attributes to the voice measures cuts the random-split RMSE by more than half (to 3.0–4.3) and leaves the held-out RMSE unchanged or higher. In other words, the high accuracies that are obtainable on this dataset under random splitting come almost entirely from patient-identifying features; the voice measures contribute modest within-patient information and, in this setup, none across patients.

### Per-patient skill

Fig. 8b compares the RMSE of ridge regression and gradient boosting with that of the training-mean predictor for each of the 42 held-out patients (all 19 features, fixed hyper-parameters). Models are better than the baseline for only 18 (ridge), 17 (random forest) and 16 (gradient boosting) of the 42 patients, with median ratios of model RMSE to baseline RMSE of 1.08, 1.20 and 1.25. The per-patient RMSE of the models ranges from about 1.5 to 30 UPDRS points, and for a few patients it exceeds 25 points. The decomposition in Fig. 8c shows that 89–94% of the squared error of the models is a patient-level offset, that is, the mean error over the recordings of a patient. This is not an artefact of the models: the training-mean predictor has the same property by construction, since 93% of the variance of total UPDRS lies between patients (Fig. 2d). What the decomposition shows is that the models do not reduce the patient-level error, which is the dominant component, and that whatever voice information they use acts on the within-patient component.

![Fig. 8. Parkinson's telemonitoring, total UPDRS. (a) RMSE of a random forest under random and patient-held-out evaluation for different feature sets; the dotted line is the training-mean predictor under patient-held-out evaluation. (b) RMSE of each of the 42 held-out patients for ridge regression and gradient boosting versus the RMSE of the training-mean predictor; points below the diagonal are patients for whom the model is better than the baseline. (c) Share of the squared held-out error that is a patient-level mean error.](../figures/paper/fig08_parkinsons_skill.png){width=6.4in}

### Calibration with recordings of a new patient

The error of the models on held-out patients is dominated by a patient-level offset, so one may ask whether a few labelled recordings of a new patient can correct it. We tested this by giving each held-out patient's model the first $n\in\{5,10,20,40\}$ recordings (by recruitment day) with their UPDRS scores and evaluating on the patient's later recordings (all recordings after the first 40, a fixed evaluation set of 61 or more recordings per patient). Two ways of using the calibration recordings were compared: an *offset correction*, which adds the mean error of the model on the $n$ recordings to its later predictions, and a *weighted retraining* of gradient boosting on the training patients plus the $n$ recordings with weight 20. As a fair reference we added a model-free predictor, the patient's own running mean over the same $n$ recordings (Fig. 9).

![Fig. 9. Calibration with recordings of a new patient (Parkinson's total UPDRS, 42 held-out patients). RMSE on the patient's recordings after the first 40, as a function of the number $n$ of labelled calibration recordings used. At $n=0$ the models are the patient-held-out models of Section 6.1 and the dashed line is the training mean.](../figures/paper/fig09_personalisation.png){width=4.8in}

The offset correction removes most of the error: with five calibration recordings the RMSE of gradient boosting falls from 14.0 to 6.4 and that of ridge regression from 11.9 to 6.6 (training-mean predictor 11.1 on the same evaluation recordings), and with 40 recordings to 5.2 for both. Weighted retraining is much less effective (11.1 with five recordings, 6.5 with 40), because a few rows have little influence on a model fitted to thousands. However, the model-free reference is better than every model at every calibration size: the patient's own running mean gives an RMSE of 5.96, 5.79, 5.47 and 4.80 for $n=5,10,20,40$, against 6.37, 6.27, 5.87 and 5.20 for gradient boosting with offset correction. In other words, the gain from calibration comes from knowing the patient's level of disease, not from the voice features, and in this setup the voice-based models add no information beyond that level. This does not exclude that better voice features or longitudinal models could do so; it shows that the standard features with the models tested do not, and that a calibration-based evaluation should always include a model-free reference of this kind.

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

## Theory versus data

### Synthetic validation

We simulated data from Eq. (1) with 50 groups, group sizes $m\in\{2,5,20,64\}$ and intraclass correlations $r\in\{0.05,0.2,0.4,0.6,0.8\}$. The shared signal $f$ depended on three group-level features, which identify the group, and on two row-level features; offsets and noise were Gaussian with variances $r$ and $1-r$. We fitted gradient boosting (150 trees, depth 3) and ridge regression under random and group-held-out 10-fold cross-validation (three replications per cell) and compared the observed inflation $\rho$ with Eq. (6) for $\kappa=1$ (Table 14, Fig. 11a).

Table: Table 14. Synthetic validation: observed inflation ratio $\rho$ of gradient boosting, with the prediction of Eq. (6) in parentheses, and the range of the observed ratio for ridge regression. Means of three replications; 50 groups.

| ICC $r$ | $m=2$ | $m=5$ | $m=20$ | $m=64$ | Ridge (all $m$) |
|---:|:---|:---|:---|:---|:---|
| 0.05 | 0.91 (1.00) | 1.06 (1.00) | 1.06 (1.01) | 1.05 (1.02) | 0.99–1.00 |
| 0.20 | 1.08 (1.03) | 1.08 (1.06) | 1.13 (1.09) | 1.28 (1.11) | 1.01–1.02 |
| 0.40 | 1.07 (1.13) | 1.23 (1.20) | 1.36 (1.26) | 1.36 (1.28) | 1.01–1.04 |
| 0.60 | 1.27 (1.33) | 1.31 (1.45) | 1.69 (1.54) | 1.61 (1.57) | 1.01–1.05 |
| 0.80 | 1.45 (1.83) | 1.66 (2.03) | 2.14 (2.18) | 2.41 (2.22) | 1.04–1.10 |

The prediction tracks the observed inflation of gradient boosting well (Pearson correlation 0.93 over the 20 cells, mean absolute difference 0.11, mean signed difference $-0.006$). It captures the growth of the inflation with the ICC and with the group size: at $r=0.8$ the observed ratio rises from 1.45 ($m=2$) to 2.41 ($m=64$), the predicted one from 1.83 to 2.22. Deviations are systematic in two places. For small groups and high ICC the learner does not reach the idealised $\kappa=1$ (observed 1.45 and 1.66 against predicted 1.83 and 2.03 for $m=2$ and 5 at $r=0.8$), because it must learn the offsets from the group-identifying features with only a few rows. For large groups the observed ratio sometimes exceeds the prediction (2.41 versus 2.22 at $r=0.8$, $m=64$; 1.28 versus 1.11 at $r=0.2$), because the learner also memorises the group-level part of the shared signal $f$, which can only be estimated from the 50 groups and is therefore less accurate for unseen groups; this is the unseen-group approximation error that Eq. (9) ignores. Ridge regression, which cannot isolate groups, shows no inflation beyond 1.10 in any cell, consistent with $\kappa\approx 0$.

### Real data

On real data the offsets are not known, but their share can be estimated from the group-held-out residuals of a model: the mean squared group-mean residual estimates the offset energy, and the remainder is within-group energy. Table 15 evaluates Eq. (6) with these estimates (with the average number of training rows per group in the training set) and compares it with the observed inflation of the same fixed-hyper-parameter model; Fig. 11b shows the comparison graphically.

Table: Table 15. Real-data check of the theory. Observed $\rho$ is the ratio of group-held-out to random-split RMSE of the fixed-hyper-parameter model; the Eq. (6) value is the inflation of an idealised learner that recovers group offsets, computed from the model's group-held-out residuals; the offset share is the fraction of the held-out squared error that is a group-level mean error; the last column is the lower bound on the group-specific variance share implied by the observed inflation.

| Dataset (target) | Model | Observed $\rho$ | Eq. (6) value | Offset share of held-out error | Implied $r_G\ge$ (Eq. 11) |
|:---|:---|---:|---:|---:|---:|
| ENB2012 heating | Gradient boosting | 13.44 | 8.53 | 0.99 | 0.99 |
| ENB2012 heating | Ridge | 1.37 | 2.89 | 0.88 | 0.47 |
| ENB2012 cooling | Gradient boosting | 4.62 | 3.95 | 0.94 | 0.95 |
| ENB2012 cooling | Ridge | 1.35 | 2.41 | 0.83 | 0.45 |
| Concrete strength | Gradient boosting | 1.31 | 1.80 | 0.75 | 0.42 |
| Concrete strength | Ridge | 1.00 | 1.20 | 0.41 | 0.01 |
| Parkinson's total UPDRS | Gradient boosting | 3.54 | 4.01 | 0.94 | 0.92 |
| Parkinson's total UPDRS | Ridge | 1.21 | 3.06 | 0.89 | 0.32 |
| Parkinson's motor UPDRS | Gradient boosting | 3.28 | 3.45 | 0.92 | 0.91 |
| Parkinson's motor UPDRS | Ridge | 1.20 | 2.70 | 0.86 | 0.31 |

For ridge regression the observed inflation (1.0–1.4) is far below the value for an idealised memorising learner (1.2–3.1), as expected for a model that cannot isolate groups. For gradient boosting the observed inflation is below the intercept-only value for Concrete (1.31 versus 1.80), Parkinson's total UPDRS (3.54 versus 4.01) and motor UPDRS (3.28 versus 3.45), so Eq. (6) works as an approximate upper bound and, for Parkinson's, a close estimate. It is exceeded for ENB2012: 4.62 versus 3.95 for cooling load and 13.4 versus 8.5 for heating load. ENB2012 is a deterministic simulation in which the load of each geometry is a smooth function of orientation and glazing that differs between geometries, so a learner that has seen a geometry can predict it to within a small fraction of the group-level error, and the group-specific component is not only an offset but a whole function. The implied lower bound on the group-specific share (Eq. 11) is 0.99 for heating and 0.95 for cooling, which agrees with the offset share of the held-out error for cooling (0.94) and slightly exceeds it for heating (0.99). For Concrete the implied share is 0.42 against an offset share of 0.75, which indicates that part of the apparent between-mix error is not something the learner can recover from the other rows of the mix.

Three conclusions follow. First, the ICC of the target is not the right quantity for predicting inflation: ENB2012 and Parkinson's data have almost identical target ICCs (0.91–0.93) but the inflation of gradient boosting ranges from 3.3 to 13.4. What matters is the share of the residual that a learner can recover from other rows of the group, which depends on the learner and on the noise. Second, an upper bound from random intercepts is informative but is exceeded when groups differ in functional form; Eq. (11) can then be used to quantify how much group-specific structure the observed inflation implies. Third, the cheap diagnostic in Table 15, which needs only the group-held-out residuals of one model, tells a practitioner before any random-split evaluation whether random splitting is likely to be misleading.

![Fig. 11. Theory versus simulation and data. (a) Observed inflation ratio in the synthetic experiment against the prediction of Eq. (6); circles gradient boosting (shade = group size), crosses ridge. (b) Real data: observed inflation of fixed-hyper-parameter gradient boosting, the intercept-only value of Eq. (6) computed from its group-held-out residuals, and observed inflation of ridge regression.](../figures/paper/fig11_theory_validation.png){width=6.4in}

## Tuning leakage

### Ablation

Table 16 compares, for each dataset and model, the group-held-out RMSE obtained when hyper-parameters are tuned with group-pure inner folds (the matched protocol) and with random inner folds (the leaky variant). The ratio is the factor by which random inner folds increase the held-out-group error.

Table: Table 16. Group-held-out RMSE with group-matched versus random inner folds. Ratio = random inner / group-matched inner. ENB2012 uses leave-one-geometry-out for both targets (Y1 heating, Y2 cooling); Concrete and Parkinson's use 10 group folds.

| Dataset (target) | Ridge | kNN | SVR (RBF) | Random forest | Gradient boosting |
|:---|:---|:---|:---|:---|:---|
| ENB2012 (heating) | 3.23 → 3.73 (1.15) | 3.66 → 3.75 (1.02) | 2.95 → 9.59 (**3.25**) | 4.14 → 4.43 (1.07) | 3.86 → 4.13 (1.07) |
| ENB2012 (cooling) | 3.50 → 4.17 (1.19) | 4.02 → 4.27 (1.06) | 3.31 → 8.91 (**2.69**) | 4.82 → 4.77 (0.99) | 4.66 → 4.63 (0.99) |
| Concrete (strength) | 10.55 → 10.54 (1.00) | 8.57 → 8.57 (1.00) | 6.46 → 6.46 (1.00) | 5.91 → 5.90 (1.00) | 5.76 → 5.76 (1.00) |
| Parkinson's (total UPDRS) | 11.55 → 11.67 (1.01) | 12.94 → 13.71 (1.06) | 13.34 → 13.74 (1.03) | 12.73 → 13.88 (1.09) | 12.32 → 13.81 (**1.12**) |
| Parkinson's (motor UPDRS) | 8.90 → 8.96 (1.01) | 9.63 → 10.16 (1.06) | 10.39 → 10.35 (1.00) | 9.24 → 9.96 (1.08) | 9.41 → 10.07 (1.07) |

The effect depends on the model and the dataset. For SVR on ENB2012, where the kernel width is a strong determinant of memorisation capacity, random inner folds raise the held-out error by a factor of 2.7–3.3. For ridge regression on ENB2012 the penalty is 15–19%. For the other models and for Parkinson's the effect is 0–12%, and for Concrete there is none, in line with the weak group structure. Tuning leakage is thus not universal, but when it occurs it can be larger than the effect of switching between model families. An early version of our own ENB2012 pipeline used random inner folds inside grouped outer folds and produced markedly worse grouped scores for the same models (preserved in `results/v1_preliminary/`, superseded by the matched protocol).

### Mechanism

Fig. 12 shows why. For each of the twelve leave-one-geometry-out folds on ENB2012 we computed, for a grid of kernel widths ($\gamma$, SVR with $C=30$) and neighbour counts ($k$, kNN), (i) the inner cross-validation error with random folds, (ii) the inner error with group-pure folds and (iii) the true error on the held-out geometry, and averaged the squared errors over the twelve folds.

![Fig. 12. Mechanism of tuning leakage on ENB2012 (heating load, leave-one-geometry-out). For each setting of the capacity parameter, the three curves show the RMSE estimated by inner cross-validation with random folds, by inner cross-validation with group-pure folds, and the true RMSE on the held-out geometry. Circles mark the minimum of each curve.](../figures/paper/fig12_tuning_paths.png){width=6.4in}

For the SVR, the random-fold inner error decreases with the kernel parameter and is minimised at $\gamma=0.3$ with an RMSE of 1.08 kWh/m², which would select a model whose true held-out RMSE is 7.25 kWh/m². The true error is minimised at the smoothest setting, $\gamma=0.003$ (3.60 kWh/m²), which is also where the group-pure inner error is minimised (4.13). The random-fold inner estimate is not only wrong about the optimum but also wrong about the level: at its chosen setting it underestimates the true error by a factor of 6.7. For kNN the random-fold criterion selects $k=3$ (inner RMSE 2.40, true RMSE 5.21), whereas the group-pure criterion selects $k=65$ (inner 4.58), close to the true optimum ($k=65$, 4.28). The curves also show that group-pure inner folds slightly over-estimate the true error (for example at $k=65$, 4.58 versus 4.28), because their training sets are smaller (two-thirds of the outer training set), which is the usual pessimistic bias of cross-validation and much smaller than the optimism of random inner folds.

The curves match the theory in Section 3.5: the random-fold criterion measures a within-group trade-off and favours capacity high enough to resolve individual groups, whereas the group-pure criterion measures a between-group trade-off and favours smooth models. The difference in selected capacity, not merely a difference in the reported estimate, is what degrades the deployed model.

## Sensitivity and uncertainty

### Number of folds and fold assignment

We repeated the comparison of random and group-held-out evaluation for fixed-hyper-parameter gradient boosting and ridge regression with different numbers of folds ($K=3$–20, depending on the number of groups) and three random assignments of groups to folds (Fig. 13a–c, Table 17).

Table: Table 17. Sensitivity of the inflation ratio of gradient boosting (fixed hyper-parameters) to the number of folds $K$. RMSE is the mean over three fold assignments; the standard deviation over assignments is given in parentheses for the group-held-out evaluation.

| Dataset | $K$ | Random RMSE | Group-held-out RMSE (SD) | Inflation $\rho$ |
|:---|---:|---:|---:|---:|
| ENB2012 (heating) | 3 | 0.43 | 6.65 (2.82) | 15.5 |
| ENB2012 (heating) | 4 | 0.42 | 5.19 (1.63) | 12.2 |
| ENB2012 (heating) | 6 | 0.42 | 5.91 (1.78) | 14.2 |
| ENB2012 (heating) | 12 (leave-one-out) | 0.41 | 5.53 (0.00) | 13.5 |
| Concrete | 5 | 4.74 | 6.06 (0.11) | 1.28 |
| Concrete | 10 | 4.63 | 5.82 (0.12) | 1.26 |
| Concrete | 20 | 4.61 | 5.94 (0.05) | 1.29 |
| Parkinson's (total UPDRS) | 5 | 3.91 | 13.04 (0.64) | 3.33 |
| Parkinson's (total UPDRS) | 10 | 3.89 | 13.22 (0.21) | 3.40 |
| Parkinson's (total UPDRS) | 20 | 3.88 | 13.47 (0.05) | 3.47 |

![Fig. 13. Sensitivity to the number of folds and precision of group-level estimates. (a–c) RMSE of gradient boosting and ridge regression under random and group-held-out evaluation for different numbers of folds $K$; error bars are the standard deviation over three fold assignments. (d) Relative half-width of a 95% interval for a mean over $G$ groups, divided by the coefficient of variation of the group-level losses (Eq. 13).](../figures/paper/fig13_sensitivity.png){width=6.2in}

The random-split RMSE is almost independent of $K$ and of the fold assignment (standard deviation at most 0.14), which is why random splitting gives the impression of a stable estimate. The group-held-out RMSE depends more on the design. For Concrete (427 groups) and Parkinson's data (42 patients) it is stable (standard deviations 0.05–0.64) and the inflation ratio varies little with $K$ (1.26–1.29 and 3.33–3.47). For ENB2012, with only twelve geometries, it depends strongly on which geometries are grouped together when $K<12$: with $K=3$, the group-held-out RMSE is 6.65 with a standard deviation of 2.82 over fold assignments, because each test fold then contains four geometries and the training set only eight, and the inflation ranges from 12.2 to 15.5 across the settings. Leave-one-geometry-out evaluation removes the dependence on fold assignment and is the preferable choice when the number of groups is small, at the price of higher computational cost.

### How precise can group-held-out estimates be?

The precision of a group-held-out estimate is set by the number of groups (Eq. 13, Fig. 13d). The relative half-width of a 95% interval for a mean over $G$ groups is $t_{G-1,0.975}/\sqrt{G}$ times the coefficient of variation of the group-level loss: 0.64 for ENB2012, 0.31 for Parkinson's and 0.10 for Concrete. The bootstrap intervals reported in Tables 7–11 follow this ordering: the relative half-widths of the gradient-boosting intervals under group-held-out evaluation are 0.51 (ENB2012 heating, interval 1.99–5.91 around 3.86), 0.14 (Parkinson's total UPDRS, 10.45–13.97 around 12.32) and 0.06 (Concrete, 5.43–6.14 around 5.76). The coefficient of variation of the group-level loss is large in all datasets because the difficulty of groups varies greatly (the per-geometry RMSE of gradient boosting ranges from 0.4 to 11 kWh/m²). For ENB2012 the interval is therefore too wide to separate most models, which is why the paired tests are not significant, whereas Concrete has enough groups for even small differences to be detected. The practical rule is that group-held-out comparisons between models require far more groups than rows, and that a dataset with a dozen groups can show the *existence* of a large inflation but not a precise model ranking.

## Ranking reversal across protocols

Fig. 14 compares model ranks under random and group-held-out evaluation for the five dataset–target combinations. For ENB2012 the ranking is reversed: gradient boosting, random forest, SVR, kNN and ridge under random splitting become SVR, ridge, kNN, gradient boosting, random forest for heating load and similarly for cooling load; the Kendall rank correlation between the two protocols is $-0.4$ for both targets. For Concrete the ranking is identical ($\tau=1$). For Parkinson's the rankings are essentially uncorrelated ($\tau=0.0$ for total and $0.2$ for motor UPDRS), with ridge regression moving from last place under random splitting to first place under patient-held-out evaluation for both targets.

![Fig. 14. Model ranks (1 = lowest RMSE) under random splitting and group-held-out evaluation for the five dataset–target combinations.](../figures/paper/fig14_rank_reversal.png){width=6.4in}

Across the five blocks, a Friedman test finds clear differences between the models under random evaluation ($\chi^2=18.1$, $p=0.001$; mean ranks: gradient boosting 1.4, random forest 1.6, SVR 3.4, kNN 3.6, ridge 5.0) but none under group-held-out evaluation ($\chi^2=2.4$, $p=0.66$; mean ranks: ridge 2.2, gradient boosting 2.8, SVR 3.0, random forest 3.4, kNN 3.6). Random splitting therefore produces a seemingly statistically solid ranking of model families, which disappears once the unit of generalisation is respected. With only five blocks the power of the grouped test is low, and the absence of significance does not demonstrate that the models are equal; it shows that the evidence from these benchmarks does not resolve their order.

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

For building energy, the relevant use of surrogate models for early design is the prediction of load for a design that has not been simulated. The group-held-out error of 3–4.5 kWh/m² for the best models on ENB2012 is therefore the number that matters, and it represents about 12–20% of the mean load. Importantly, even the simple models retain a clear skill relative to the training mean, so the dataset supports useful surrogate modelling, though at a lower accuracy than the random-split literature suggests. More informative benchmarks would include far more distinct geometries, which our learning curves show is what limits generalisation, not more variants of the same geometry. For clinical machine learning with repeated measures, our results reinforce earlier warnings on subject-wise validation [5, 14] and extend them in two ways: leakage can be so severe that the random-split score reflects only identity features, and the baseline that a constant achieves should always be reported.

For curators of benchmarks, the findings suggest publishing a group identifier and recommended group-held-out splits with each dataset, and reporting the ICC and group-size distribution in the dataset documentation. This is a small addition to the metadata and would prevent the most common form of over-optimistic comparison.

## Threats to validity

*Internal validity.* The group definitions are derived from attributes (X1–X5 in ENB2012; identical ingredient quantities in Concrete; the subject identifier in Parkinson's). If two groups that we treat as distinct are in fact near-duplicates (for instance, mixes that differ only slightly in water content), our group-held-out evaluation still leaks across them and underestimates the inflation. Hyper-parameter grids are modest and the fixed-parameter supplementary experiments do not tune; conclusions about relative model performance should not be read as statements about the best achievable performance of each family.

*Construct validity.* RMSE pooled over folds weights rows, not groups, and therefore weights large groups more; group-level averages would give different numbers in unbalanced designs such as Concrete. The skill score uses the training mean as the reference, which is the simplest option and could be replaced with a stronger baseline.

*External validity.* Three datasets, two of which are simulated or small, are not a representative sample of benchmarks. The relationship between group structure and inflation is based on five dataset–target combinations and should be treated as a hypothesis for further testing. Real measured building-energy data with explicit building and site groups (for example the Building Data Genome Project 2 [29]) would be a natural next test.

*Statistical conclusion validity.* With 12 and 42 groups, group-held-out estimates are noisy, and many model differences are not significant; we report intervals and avoid ranking claims where the evidence is weak. The random-split intervals are optimistic because folds are dependent.

# Limitations and future work

**Scope of the evidence.** The study uses three datasets and five dataset–target combinations. The relationship between group structure and inflation is established as a mechanism (theory, simulation and dose–response in Concrete) and as a pattern across the three datasets, but it is not estimated as a statistical relationship; with more datasets one could fit the inflation ratio as a function of the ICC, the group size and the model capacity. Two of the datasets are simulations or small, and none contains real measured building-energy data. Natural extensions are the Building Data Genome Project 2, in which buildings and sites are explicit groups and temporal dependence adds a second leakage channel [29], and other widely used benchmarks with repeated measures.

**Few groups.** ENB2012 has twelve geometries and Parkinson's 42 patients. Group-held-out estimates have wide intervals, and paired tests have little power; we avoid ranking claims and report the uncertainty. Simulations of the validation procedure itself, with known group-level truth, could quantify the power of group-held-out comparisons as a function of the number of groups.

**Models and tuning.** We tuned small grids for five model families. Neural networks, Gaussian processes, mixed-effects models and models with built-in extrapolation (for example linear trees or monotonic constraints) were not evaluated. Mixed-effects models and Gaussian processes with group-level kernels are particularly interesting because they model the group offsets explicitly and can use calibration rows of a new group in a principled way; our simple offset correction and weighted retraining are a first step in that direction.

**Feature sets.** For Parkinson's data we used the standard feature set and examined subsets, but did not explore alternative voice features or longitudinal models. Whether any voice feature set generalises across patients without calibration remains open; our results show only that the standard measures with the models tested do not.

**Theory.** The random-intercept model is a stylised description. The extension to group-specific functions explains why observed inflation can exceed the intercept-only bound, but a full treatment would model random slopes, non-Gaussian errors and the finite-sample behaviour of specific learners. The formulae are bounds or approximations, and we have validated them in simulation only for a gradient-boosting learner and a linear learner.

**Future work.** (i) A larger meta-analysis of benchmark datasets with a detectable group structure, with the ICC, group-size distribution and inflation recorded for each. (ii) Methods for detecting hidden groups from the data (near-duplicate detection, clustering of identical attribute combinations) so that group-held-out validation can be applied when no identifier is provided. (iii) Evaluation protocols for deployment settings in which calibration rows of a new group become available over time. (iv) Group-aware hyper-parameter search implemented as the default in common machine-learning libraries.

# Conclusions

Random train–test splits on tabular benchmarks with repeated groups measure how well a model recognises the group, and this can dominate what is reported. On three widely used datasets we quantified the effect with matched nested tuning. The inflation of random-split accuracy ranges from negligible (Concrete, where groups are small and often singletons, and ridge regression everywhere) to nearly an order of magnitude (tree ensembles on ENB2012 and Parkinson's data, where groups are large and tight). A random-effects model predicts the inflation of a learner that memorises group offsets as $1/\sqrt{1-r}$ in the ICC $r$ and reproduces it in simulation (correlation 0.93); on real data, interactions between groups and features can push the inflation beyond the intercept-only prediction, and observed ratios imply that group-specific structure accounts for more than 90% of the residual variance in ENB2012 heating load.

Four practical findings follow. First, model rankings change with the protocol: on ENB2012 and Parkinson's data the models that win under random splitting are among the worst on new groups, and the clear statistical ranking under random splitting (Friedman $p=0.001$) disappears under group-held-out evaluation ($p=0.66$). Second, tuning with random inner folds can degrade the model that is eventually deployed, with errors up to three times larger for support-vector regression on ENB2012. Third, a no-skill reference is indispensable: on Parkinson's data all models are at best no better than the training mean on new patients, although they look nearly perfect under random splitting, and the features that produce the near-perfect scores are demographic and temporal identifiers, not voice measures. Fourth, learning curves over the number of training groups show that more rows from the same groups do not help new groups; what helps is more groups.

We recommend that benchmark users identify groups before splitting, use group-pure outer and inner folds, report a no-skill baseline and uncertainty over groups, and that benchmark curators publish group identifiers and recommended splits. The code, per-fold results and figures accompanying this paper make it straightforward to apply the same checks to other datasets.

# Declarations {-}

**CRediT authorship contribution statement.** *To be completed by the authors.* (Conceptualisation; Methodology; Software; Validation; Formal analysis; Investigation; Data curation; Writing – original draft; Writing – review and editing; Visualisation.)

**Declaration of generative AI and AI-assisted technologies in the writing process.** During the preparation of this work the author(s) used Claude (Anthropic) to assist with writing and running the analysis code, with generating figures and tables, with drafting and editing the manuscript text, and with searching for and checking bibliographic details. After using this tool, the author(s) reviewed and edited the content and take full responsibility for the content of the published article. *(Authors: please review this statement and adapt it to the target journal's policy.)*

**Declaration of competing interest.** *To be completed by the authors.*

**Funding.** *To be completed by the authors.*

**Data availability.** The datasets are public (UCI Machine Learning Repository; mirrored on Kaggle). The exact files, all code, per-fold results and figures are available at https://github.com/deep1789/Energy-efficient (repository named `Energy-efficient`).

**Acknowledgements.** *To be completed by the authors.*

# Appendix A. Derivations {-}

## A.1 Posterior variance and best linear unbiased predictor (Eqs. 4–5) {-}
Consider one group with offset $u\sim(0,\sigma_u^2)$ and $n$ training rows whose residuals after removing the shared signal are $e_l=u+\varepsilon_l$, with independent noise of variance $\sigma_\varepsilon^2$. The mean residual is $\bar e=u+\bar\varepsilon$ with $\operatorname{Var}(\bar\varepsilon)=\sigma_\varepsilon^2/n$. The best linear predictor of $u$ from $\bar e$ is $\hat u=w\bar e$ with $w=\operatorname{Cov}(u,\bar e)/\operatorname{Var}(\bar e)=\sigma_u^2/(\sigma_u^2+\sigma_\varepsilon^2/n)=n\sigma_u^2/(n\sigma_u^2+\sigma_\varepsilon^2)$, which is Eq. (4). Its mean-squared error is

$$\operatorname{Var}(u-\hat u)=\sigma_u^2-\frac{\operatorname{Cov}(u,\bar e)^2}{\operatorname{Var}(\bar e)}=\sigma_u^2-\frac{\sigma_u^4}{\sigma_u^2+\sigma_\varepsilon^2/n}=\frac{\sigma_u^2\sigma_\varepsilon^2}{\sigma_\varepsilon^2+n\sigma_u^2}, \qquad\qquad (A1)$$

which equals Eq. (5) and, written as $(1/\sigma_u^2+n/\sigma_\varepsilon^2)^{-1}$, is the posterior variance when $u$ and the noise are Gaussian. Without Gaussianity, Eq. (A1) is the error of the best linear predictor, which is the BLUP of the random-effects literature [32, 33]. Two properties are used in the text: $V(n)\le\sigma_u^2$ with $V(0)=\sigma_u^2$, and $V(n)\to 0$ as $n\to\infty$.

## A.2 Inflation of the idealised learner (Eq. 6) {-}
For a test row of a seen group the learner predicts $f(x)+\hat u$, so the error is $\varepsilon+(u-\hat u)$. The noise $\varepsilon$ of the test row is independent of the training rows, and $u-\hat u$ depends only on training rows; hence $R_{\mathrm{rand}}=\sigma_\varepsilon^2+V(n)$. For a test row of an unseen group the learner can only predict $f(x)$ (the offset has mean zero), and the error is $u+\varepsilon$, so $R_{\mathrm{grp}}=\sigma_u^2+\sigma_\varepsilon^2=\sigma^2$. With $\sigma^2=1$, $\sigma_u^2=r$ and $\sigma_\varepsilon^2=1-r$,

$$\rho^2=\frac{R_{\mathrm{grp}}}{R_{\mathrm{rand}}}=\frac{1}{(1-r)+V(n)},\qquad V(n)=\frac{r(1-r)}{(1-r)+nr}. \qquad\qquad (A2)$$

For $n\to\infty$, $V\to 0$ and $\rho^2\to 1/(1-r)$, which is Eq. (7). Monotonicity: $V(n)$ decreases in $n$, so $\rho$ increases with the number of training rows per group, and increases with $r$ for fixed $n$ because both $1/(1-r)$ and the shrinkage term increase.

## A.3 Learners of limited capacity (Eq. 8) {-}
Let the learner predict $f(x)+\kappa\hat u$ for a seen group. The error is $\varepsilon+u-\kappa\hat u=\varepsilon+(u-\hat u)+(1-\kappa)\hat u$. The three terms are uncorrelated ($\varepsilon$ is independent of everything else, and $u-\hat u$ is orthogonal to $\hat u$ because $\hat u$ is the projection of $u$ on $\bar e$). With $\operatorname{Var}(\hat u)=\operatorname{Var}(u)-\operatorname{Var}(u-\hat u)=\sigma_u^2-V(n)$,

$$R_{\mathrm{rand}}(\kappa)=\sigma_\varepsilon^2+V(n)+(1-\kappa)^2\bigl(\sigma_u^2-V(n)\bigr). \qquad\qquad (A3)$$

This is Eq. (8). At $\kappa=1$ it reduces to A.2 and at $\kappa=0$ to $\sigma^2$, i.e. $\rho=1$. $R_{\mathrm{rand}}$ is decreasing in $\kappa$ on $[0,1]$ because $\sigma_u^2-V(n)\ge 0$.

## A.4 Group-specific functions (Eqs. 10–11) {-}
Replace $u_i$ by a group-specific function $g_i(x)$ with $\mathbb E\,g_i(x)^2=\sigma_G^2$. A learner that estimates $g_i$ from the training rows of the group has error $V_G(n)=\mathbb E\,(g_i-\hat g_i)^2\in[0,\sigma_G^2]$. Then $R_{\mathrm{rand}}=\sigma_\varepsilon^2+V_G(n)$ and $R_{\mathrm{grp}}=\sigma_G^2+\sigma_\varepsilon^2$, hence

$$\rho^2=\frac{\sigma_G^2+\sigma_\varepsilon^2}{V_G(n)+\sigma_\varepsilon^2}\le\frac{\sigma_G^2+\sigma_\varepsilon^2}{\sigma_\varepsilon^2}=\frac{1}{1-r_G}.\qquad\qquad (A4)$$

If $\sigma_\varepsilon^2=0$ the bound is infinite. For the converse (Eq. 11), note that $\sigma_\varepsilon^2\le R_{\mathrm{rand}}$ for every learner, so $1-r_G=\sigma_\varepsilon^2/(\sigma_G^2+\sigma_\varepsilon^2)\le R_{\mathrm{rand}}/R_{\mathrm{grp}}=1/\rho^2$ when $R_{\mathrm{grp}}=\sigma_G^2+\sigma_\varepsilon^2$, i.e. $r_G\ge 1-1/\rho^2$. For $\rho=9.5$, as observed for gradient boosting on ENB2012 heating load, $r_G\ge 0.989$.

## A.5 Leakage fraction (Eq. 12) {-}
In random $K$-fold cross-validation with equal-sized folds, the other $m-1$ rows of a group of size $m$ are all in the test fold of a given row with probability $\prod_{j=1}^{m-1}\frac{n_f-j}{N-j}$, where $n_f=N/K$ is the fold size; for $N$ much larger than $m$ this is $K^{-(m-1)}$. A row has at least one training group-mate with the complementary probability. Summing over rows gives Eq. (12). For singleton groups ($m=1$) the probability is zero and no leakage is possible.

## A.6 Tuning with $k$-nearest neighbours (Section 3.5) {-}
Assume features that identify groups, Eq. (1) with a locally constant $f$ within a group, and $k$ not larger than the number of training rows in a group. With random inner folds, the $k$ nearest neighbours of a validation row are training rows of the same group, so the prediction is $\hat y=f+u_i+\bar\varepsilon_k$ and the error is $\varepsilon-\bar\varepsilon_k$ plus bias, with variance $\sigma_\varepsilon^2(1+1/k)$, decreasing in $k$ until the bias from within-group variation of $f$ becomes significant; the minimising $k$ is large relative to one but at most the group's training size. With group-pure inner folds, the neighbours belong to other groups $j_1,\dots,j_{k_g}$, so the error is $(u_i-\bar u_{nb})+\varepsilon-\bar\varepsilon_k$ plus the bias from the difference of $f$ between groups; its variance is $\sigma_u^2(1+1/k_g)+\sigma_\varepsilon^2(1+1/k)$, which decreases until the bias from averaging over distant groups dominates. The scales of the two trade-offs differ (within-group spread of $f$ versus between-group spacing), so the minimisers differ, and the group-pure criterion selects the value that is optimal for new groups. For a radial-basis kernel the same argument associates random inner folds with a small bandwidth.

## A.7 Design effect (Eq. 14) {-}
For $G$ groups of size $\bar m$ with intraclass correlation $\varrho$ of the loss $\ell$, the variance of the overall mean is $\operatorname{Var}(\bar\ell)=\frac{\operatorname{Var}(\ell)}{N}\bigl[1+(\bar m-1)\varrho\bigr]$. The factor in brackets is the design effect, and the effective sample size is $N/[1+(\bar m-1)\varrho]$, so $N_{\mathrm{eff}}\to G$ when $\varrho\to 1$.

# Appendix B. Reproducibility details {-}

All scripts are in the `src/` folder of the repository; `src/common.py` provides the data loaders and the fixed-hyper-parameter models. Main experiments: `run_experiments.py` (ENB2012), `run_extra.py` (Concrete and Parkinson's; Parkinson's can be run per protocol and target to use several cores, and results are saved after every model so that interrupted runs can resume). Supplementary experiments: `exp_theory.py`, `exp_icc.py`, `exp_curves.py`, `exp_parkinsons.py`, `exp_sensitivity.py`, `exp_tuning_paths.py`, `exp_bound_obs.py`, `exp_ablate_enb_knn.py`, `ablation_inner_cv.py`. Figures and tables: `analyze.py`, `analyze_extra.py`, `make_paper_figs*.py`, `fig_structure.py`, `baselines.py`. Per-fold results are stored as CSV files under `results/`.


# References {-}

\[1\] S. Kaufman, S. Rosset, C. Perlich, O. Stitelman, Leakage in data mining: formulation, detection, and avoidance, ACM Transactions on Knowledge Discovery from Data 6 (4) (2012) 15. doi:10.1145/2382577.2382579.

\[2\] S. Kapoor, A. Narayanan, Leakage and the reproducibility crisis in machine-learning-based science, Patterns 4 (9) (2023) 100804. doi:10.1016/j.patter.2023.100804.

\[3\] C. Ambroise, G.J. McLachlan, Selection bias in gene extraction on the basis of microarray gene-expression data, Proceedings of the National Academy of Sciences 99 (10) (2002) 6562–6566. doi:10.1073/pnas.102102699.

\[4\] D.R. Roberts, V. Bahn, S. Ciuti, M.S. Boyce, J. Elith, G. Guillera-Arroita, et al., Cross-validation strategies for data with temporal, spatial, hierarchical, or phylogenetic structure, Ecography 40 (8) (2017) 913–929. doi:10.1111/ecog.02881.

\[5\] S. Saeb, L. Lonini, A. Jayaraman, D.C. Mohr, K.P. Kording, The need to approximate the use-case in clinical machine learning, GigaScience 6 (5) (2017) gix019.

\[6\] M. Rosenblatt, L. Tejavibulya, R. Jiang, et al., Data leakage inflates prediction performance in connectome-based machine learning models, Nature Communications 15 (2024) 1829. doi:10.1038/s41467-024-46150-w.

\[7\] P. Ploton, F. Mortier, et al., Spatial validation reveals poor predictive performance of large-scale ecological mapping models, Nature Communications 11 (2020) 4540.

\[8\] A. Tsanas, A. Xifara, Accurate quantitative estimation of energy performance of residential buildings using statistical machine learning tools, Energy and Buildings 49 (2012) 560–567.

\[9\] I-C. Yeh, Modeling of strength of high-performance concrete using artificial neural networks, Cement and Concrete Research 28 (12) (1998) 1797–1808.

\[10\] A. Tsanas, M.A. Little, P.E. McSharry, L.O. Ramig, Accurate telemonitoring of Parkinson's disease progression by non-invasive speech tests, IEEE Transactions on Biomedical Engineering 57 (4) (2010) 884–893.

\[11\] G.C. Cawley, N.L.C. Talbot, On over-fitting in model selection and subsequent selection bias in performance evaluation, Journal of Machine Learning Research 11 (2010) 2079–2107.

\[12\] S. Varma, R. Simon, Bias in error estimation when using cross-validation for model selection, BMC Bioinformatics 7 (2006) 91.

\[13\] H. Meyer, C. Reudenbach, T. Hengl, M. Katurji, T. Nauss, Improving performance of spatio-temporal machine learning models using forward feature selection and target-oriented validation, Environmental Modelling & Software 101 (2018) 1–9. doi:10.1016/j.envsoft.2017.12.001.

\[14\] E. Chaibub Neto, A. Pratap, T.M. Perumal, et al., Detecting the impact of subject characteristics on machine learning-based diagnostic applications, npj Digital Medicine 2 (2019) 99. doi:10.1038/s41746-019-0178-x.

\[15\] S.A. Al-Azazi, Risk-aware explainable multi-output workflow for early-stage screening of building heating and cooling load indicators, Scientific Reports (2026), article s41598-026-59378-x.

\[16\] C. Bergmeir, J.M. Benítez, On the use of cross-validation for time series predictor evaluation, Information Sciences 191 (2012) 192–213. doi:10.1016/j.ins.2011.12.028.

\[17\] S. Arlot, A. Celisse, A survey of cross-validation procedures for model selection, Statistics Surveys 4 (2010) 40–79. doi:10.1214/09-SS054.

\[18\] G. Varoquaux, Cross-validation failure: small sample sizes lead to large error bars, NeuroImage 180 (2018) 68–77. doi:10.1016/j.neuroimage.2017.06.061.

\[19\] S. Bates, T. Hastie, R. Tibshirani, Cross-validation: what does it estimate and how well does it do it?, Journal of the American Statistical Association 119 (546) (2024) 1434–1445.

\[20\] J-S. Chou, D-K. Bui, Modeling heating and cooling loads by artificial intelligence for energy-efficient building design, Energy and Buildings 82 (2014) 437–446. doi:10.1016/j.enbuild.2014.07.036.

\[21\] M-Y. Cheng, M-T. Cao, Accurately predicting building energy performance using evolutionary multivariate adaptive regression splines, Applied Soft Computing 22 (2014) 178–188.

\[22\] K. Kavaklioglu, Robust modeling of heating and cooling loads using partial least squares towards efficient residential building design, Journal of Building Engineering 18 (2018) 467–475. doi:10.1016/j.jobe.2018.04.018.

\[23\] H. Moayedi, D.T. Bui, A. Dounis, Z. Lyu, L.K. Foong, Predicting heating load in energy-efficient buildings through machine learning techniques, Applied Sciences 9 (20) (2019) 4338. doi:10.3390/app9204338.

\[24\] J-S. Chou, C-F. Tsai, A-D. Pham, Y-H. Lu, Machine learning in concrete strength simulations: multi-nation data analytics, Construction and Building Materials 73 (2014) 771–780. doi:10.1016/j.conbuildmat.2014.09.054.

\[25\] S. Sheikhi, M.T. Kheirabadi, An efficient rotation forest-based ensemble approach for predicting severity of Parkinson's disease, Journal of Healthcare Engineering (2022) 5524852. doi:10.1155/2022/5524852.

\[26\] B.A. Omodunbi, D.B. Olawade, O.F. Awe, A.A. Soladoye, N. Aderinto, S.V. Ovsepian, S. Boussios, Stacked ensemble learning for classification of Parkinson's disease using telemonitoring vocal features, Diagnostics 15 (12) (2025) 1467. doi:10.3390/diagnostics15121467.

\[27\] K. Amasyali, N.M. El-Gohary, A review of data-driven building energy consumption prediction studies, Renewable and Sustainable Energy Reviews 81 (2018) 1192–1205. doi:10.1016/j.rser.2017.04.095.

\[28\] Y. Wei, X. Zhang, Y. Shi, L. Xia, S. Pan, J. Wu, M. Han, X. Zhao, A review of data-driven approaches for prediction and classification of building energy consumption, Renewable and Sustainable Energy Reviews 82 (2018) 1027–1047.

\[29\] C. Miller, A. Kathirgamanathan, et al., The Building Data Genome Project 2, energy meter data from the ASHRAE Great Energy Predictor III competition, Scientific Data 7 (2020) 368. doi:10.1038/s41597-020-00712-x.

\[30\] P.E. Shrout, J.L. Fleiss, Intraclass correlations: uses in assessing rater reliability, Psychological Bulletin 86 (2) (1979) 420–428. doi:10.1037/0033-2909.86.2.420.

\[31\] S.R. Searle, G. Casella, C.E. McCulloch, Variance Components, John Wiley & Sons, New York, 1992.

\[32\] C.R. Henderson, Best linear unbiased estimation and prediction under a selection model, Biometrics 31 (1975) 423–447.

\[33\] G.K. Robinson, That BLUP is a good thing: the estimation of random effects, Statistical Science 6 (1991) 15–32. doi:10.1214/ss/1177011926.

\[34\] L. Kish, Survey Sampling, John Wiley & Sons, New York, 1965.

\[35\] B. Efron, R.J. Tibshirani, An Introduction to the Bootstrap, Chapman & Hall, New York, 1993.

\[36\] S. Fahn, R.L. Elton, UPDRS Program Members, Unified Parkinson's Disease Rating Scale, in: S. Fahn, C.D. Marsden, M. Goldstein, D.B. Calne (Eds.), Recent Developments in Parkinson's Disease, vol. 2, Macmillan Healthcare Information, Florham Park, NJ, 1987, pp. 153–163.

\[37\] A.E. Hoerl, R.W. Kennard, Ridge regression: biased estimation for nonorthogonal problems, Technometrics 12 (1) (1970) 55–67.

\[38\] C. Cortes, V. Vapnik, Support-vector networks, Machine Learning 20 (1995) 273–297. doi:10.1007/BF00994018.

\[39\] L. Breiman, Random forests, Machine Learning 45 (1) (2001) 5–32. doi:10.1023/A:1010933404324.

\[40\] J.H. Friedman, Greedy function approximation: a gradient boosting machine, Annals of Statistics 29 (5) (2001) 1189–1232. doi:10.1214/aos/1013203451.

\[41\] F. Pedregosa, G. Varoquaux, et al., Scikit-learn: machine learning in Python, Journal of Machine Learning Research 12 (2011) 2825–2830.

\[42\] F. Wilcoxon, Individual comparisons by ranking methods, Biometrics Bulletin 1 (6) (1945) 80–83.

\[43\] J. Demšar, Statistical comparisons of classifiers over multiple data sets, Journal of Machine Learning Research 7 (2006) 1–30.
