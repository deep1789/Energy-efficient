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
