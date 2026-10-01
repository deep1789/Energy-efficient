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
