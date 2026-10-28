Both models fitted per PROBLEM_STATEMENT.md and independently rechecked; overall PASS.

| model | a | b | sd_parameter | coverage | nll | n_train | n_test |
|---|---|---|---|---|---|---|---|
| M1 | 1.5567927 | 0.1003597 | s = 27.983031 | 0.846154 (11/13) | 4.976150 | 37 | 13 |
| M2 | 1.3804030 | 0.1032354 | c = 0.5532311 | 0.846154 (11/13) | 4.503771 | 37 | 13 |

Deliverables in workspace/results/: fit_M1.py, M1.csv, fit_M2.py, M2.csv, check.md. The checker recomputed both models from the raw data (columns identified by name), and all checks passed: CSV schema, split sizes, a, b, sd, coverage, nll (relative diff 0).

Caveats: the std convention is ddof=0 (the statement does not specify it). M2's mean uses a and b from the [1, x] fit of y/x, as written, so a is an intercept of y/x and not a coefficient of x in y. No ground-truth evaluator was registered or used (0 evals); the dataset is the reference.