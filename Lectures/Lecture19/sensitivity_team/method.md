# Method: global sensitivity with 64 experiments

## Reviewer findings (delegation D001, arXiv corpus only, BM25 retrieval)
- Saltelli / Jansen / Homma-Saltelli pick-freeze (A, B, A_B^(i) design): compares eight total-order estimators on a shared radial design (Puy et al., 2020).
- Sample-based total-index algorithm reusing Sobol' matrix pairs for elementary effects per run (Lo Piano et al., 2017).
- Asymmetric Saltelli design with variable allocation of runs across factors (Lo Piano et al., 2022).
- Polynomial chaos / generalized chaos expansion with Sobol' post-processing and analytic lower bounds (Roustant et al., 2019).
- Gaussian-process surrogate with Sobol' post-processing and active learning under a fixed budget (Lambert et al., 2026).
- Morris elementary effects: mentioned only as a comparator (Lo Piano et al., 2017); no settings found.
- HDMR: not searched.

## Reviewer recommendation
Radial Saltelli design with N = 9 base rows (N(d+2) = 63 runs for d = 5), first-order index by the Saltelli estimator, total-order index by the Jansen estimator, bootstrap 95% intervals. The reviewer stated the corpus gives no accuracy figure for N = 9 at d = 5.

## Conflict with the mandated design
The task fixes the design to 64 points from the f3dasm Sobol sampler (seed 0). A pick-freeze design cannot be built from 64 arbitrary points, so the recommended estimator cannot be applied to the mandated design.

## What was done
- 64 points from f3dasm's Sobol sampler (seed 0) on [-1, 1]^5, stored in training/experiment_data (64 rows, x1..x5).
- Experiment("training") ran all 64 rows (64 of 64 evaluations used).
- Checked the design: pairwise input correlations are about 0.01 or less, so the inputs are not degenerate. (An earlier check mistakenly concluded the columns were identical, because each column has the same set of marginal values; the rows differ.)
- Analysis (sensitivity.py): fit a GP surrogate (ARD Matern 5/2 plus white noise) on the 64 points. 8-fold CV gives R^2 = 0.960, RMSE = 0.327 (std of y = 1.63). Then compute first-order indices with the Saltelli (2010) estimator and total-order indices with the Jansen (1999) estimator on the surrogate, using N = 200000 Monte Carlo samples. This follows the reviewer's corpus-supported route for a fixed budget (GP surrogate with Sobol post-processing, Lambert et al., 2026), since the pick-freeze design is unavailable.
- Caveat: the indices are only as accurate as the surrogate. No bootstrap or replicate design was run, so no confidence intervals are reported.

## Result
results.json holds the surrogate-based first-order and total indices for x1..x5, with n_experiments = 64.
