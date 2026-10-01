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
