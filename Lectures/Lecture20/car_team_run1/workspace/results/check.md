# Check of M1 and M2 (independent recomputation by checker, D003)

Overall verdict: **PASS**

Tolerance: relative difference < 1e-6 (|stored - recomputed| / |recomputed|). Integer items compared exactly.
Recomputation script: runs/20261027T232252/debug/delegations/D003/recompute.py (not in workspace/results/; fit_M1.py and fit_M2.py were not imported or run).

Recomputation method: `ExperimentData.from_file(data/car_stopping_distance)`, `to_pandas()`; split with `train_test_split(row index, train_size=0.75, random_state=1987)`. 

## M1 (y ~ a*x + b*x**2, no intercept; sd = s = std of train residuals, ddof=0)

| Item | Stored (M1.csv) | Recomputed | Rel. diff | Result |
|---|---|---|---|---|
| split sizes n_train / n_test | 37 / 13 | 37 / 13 | exact | PASS |
| a | 1.5567927444618306 | 1.5567927444618306 | 0 | PASS |
| b | 0.10035968486568629 | 0.10035968486568629 | 9.7e-16 | PASS |
| sd_parameter (s) | 27.983030806957824 | 27.983030806957824 | 0 | PASS |
| coverage | 0.8461538461538461 | 0.8461538461538461 | 0 | PASS |
| nll | 4.976149641475809 | 4.976149641475809 | 0 | PASS |
| column schema | model,a,b,sd_parameter,coverage,nll,n_train,n_test; 1 row | same | — | PASS |

## M2 (lstsq of y/x on [1, x] gives (p, q); a = p, b = q; sd = c*x, c = std of train residuals of y/x, ddof=0)

| Item | Stored (M2.csv) | Recomputed | Rel. diff | Result |
|---|---|---|---|---|
| split sizes n_train / n_test | 37 / 13 | 37 / 13 | exact | PASS |
| a | 1.3804030477145794 | 1.3804030477145794 | 0 | PASS |
| b | 0.10323542514078463 | 0.10323542514078463 | 2.7e-16 | PASS |
| sd_parameter (c) | 0.5532310661868639 | 0.5532310661868639 | 0 | PASS |
| coverage | 0.8461538461538461 | 0.8461538461538461 | 0 | PASS |
| nll | 4.503770939937339 | 4.503770939937339 | 0 | PASS |
| column schema | model,a,b,sd_parameter,coverage,nll,n_train,n_test; 1 row | same | — | PASS |

Note: the M2 mapping a = p, b = q (from y/x = p + q*x, so y = p*x + q*x**2) is the reading implied by the problem statement's "mean a*x + b*x**2".
