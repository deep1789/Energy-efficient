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
