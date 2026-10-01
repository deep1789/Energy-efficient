# Appendix A. Derivations

## A.1 Posterior variance and best linear unbiased predictor (Eqs. 4–5) {-}
Consider one group with offset $u\sim(0,\sigma_u^2)$ and $n$ training rows whose residuals after removing the shared signal are $e_l=u+\varepsilon_l$, with independent noise of variance $\sigma_\varepsilon^2$. The mean residual is $\bar e=u+\bar\varepsilon$ with $\operatorname{Var}(\bar\varepsilon)=\sigma_\varepsilon^2/n$. The best linear predictor of $u$ from $\bar e$ is $\hat u=w\bar e$ with $w=\operatorname{Cov}(u,\bar e)/\operatorname{Var}(\bar e)=\sigma_u^2/(\sigma_u^2+\sigma_\varepsilon^2/n)=n\sigma_u^2/(n\sigma_u^2+\sigma_\varepsilon^2)$, which is Eq. (4). Its mean-squared error is

$$\operatorname{Var}(u-\hat u)=\sigma_u^2-\frac{\operatorname{Cov}(u,\bar e)^2}{\operatorname{Var}(\bar e)}=\sigma_u^2-\frac{\sigma_u^4}{\sigma_u^2+\sigma_\varepsilon^2/n}=\frac{\sigma_u^2\sigma_\varepsilon^2}{\sigma_\varepsilon^2+n\sigma_u^2}, \qquad\qquad (A1)$$

which equals Eq. (5) and, written as $(1/\sigma_u^2+n/\sigma_\varepsilon^2)^{-1}$, is the posterior variance when $u$ and the noise are Gaussian. Without Gaussianity, Eq. (A1) is the error of the best linear predictor, which is the BLUP of the random-effects literature [@henderson1975; @robinson1991]. Two properties are used in the text: $V(n)\le\sigma_u^2$ with $V(0)=\sigma_u^2$, and $V(n)\to 0$ as $n\to\infty$.

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

# Appendix B. Reproducibility details

All scripts are in the `src/` folder of the repository; `src/common.py` provides the data loaders and the fixed-hyper-parameter models. Main experiments: `run_experiments.py` (ENB2012), `run_extra.py` (Concrete and Parkinson's; Parkinson's can be run per protocol and target to use several cores, and results are saved after every model so that interrupted runs can resume). Supplementary experiments: `exp_theory.py`, `exp_icc.py`, `exp_curves.py`, `exp_parkinsons.py`, `exp_sensitivity.py`, `exp_tuning_paths.py`, `exp_bound_obs.py`, `exp_ablate_enb_knn.py`, `ablation_inner_cv.py`. Figures and tables: `analyze.py`, `analyze_extra.py`, `make_paper_figs*.py`, `fig_structure.py`, `baselines.py`. Per-fold results are stored as CSV files under `results/`.
