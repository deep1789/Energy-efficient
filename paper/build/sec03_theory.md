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

for the intraclass correlation of the residual, that is, the fraction of the residual variance that is a group offset [@shrout1979; @searle1992]. Note that $r$ is defined after removing $f$; the raw ICC of the target can be much larger when $f$ varies strongly between groups.

Let $\hat h$ be a predictor trained on a training set. We distinguish two risks. The *random-split risk* $R_{\mathrm{rand}}$ is the expected squared error on a test row whose group also contributes rows to the training set, which is what a random split measures when groups are large. The *group-held-out risk* $R_{\mathrm{grp}}$ is the expected squared error on a row from a group with no training rows, which is what a grouped split measures and what a user faces when predicting a new unit. The *inflation ratio* is

$$\rho = \sqrt{R_{\mathrm{grp}}/R_{\mathrm{rand}}} \qquad\qquad (3)$$

and the *optimism* is $\Delta=R_{\mathrm{grp}}-R_{\mathrm{rand}}$. With RMSE as the reported metric, $\rho$ is the factor by which the held-out-group RMSE exceeds the random-split RMSE.

## Inflation as a function of ICC and group size

**Idealised memorising learner.** Suppose the learner knows $f$ and estimates the offset of a seen group from its $n$ training rows by the best linear unbiased predictor (BLUP), that is, by shrinking the mean training residual $\bar e_i$ towards zero [@henderson1975; @robinson1991]:

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

the design effect of Kish [@kish1965]. For $\bar m=64$ and even a modest $\varrho_\ell=0.1$ the effective sample size is $N/7.3\approx 105$ rather than 768. Random-split intervals that treat all rows or folds as independent therefore understate uncertainty about performance on new groups, and differences between models that look clear under random splitting may be unresolved under grouped splitting. We bootstrap over groups (or over folds, which are group-pure in the grouped protocol) to respect this dependence [@efron1993].

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
