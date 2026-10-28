# Check of M1 and M2 (checker, independent recomputation)

Recomputed by `runs/20261027T232448/debug/delegations/D003/checker_recompute.py` from `data/car_stopping_distance` (ExperimentData.from_file, to_pandas()), without importing or reusing `fit_M1.py` / `fit_M2.py`. Split: `train_test_split(row index, train_size=0.75, random_state=1987)`. Tolerance: relative difference 1e-6.

## Structure checks

| Check | CSV | Recomputed | Result |
|---|---|---|---|
| Header M1 and M2 is `model,a,b,sd_parameter,coverage,nll,n_train,n_test` | as specified | as specified | PASS |
| One data row each (M1, M2) | 1 / 1 | 1 / 1 | PASS |
| n_train / n_test | 37 / 13 (both) | 37 / 13 | PASS |

## M1 (a*x + b*x**2, sd = s)

| Item | CSV | Recomputed | Rel. diff | Result |
|---|---|---|---|---|
| a | 1.5567927444618306 | 1.5567927444618306 | 0 | PASS |
| b | 0.10035968486568629 | 0.10035968486568629 | 0 | PASS |
| sd_parameter (s) | 27.983030806957824 | 27.983030806957824 | 0 | PASS |
| coverage | 0.8461538461538461 | 0.8461538461538461 | 0 | PASS |
| nll | 4.976149641475809 | 4.976149641475809 | 0 | PASS |

## M2 (a*x + b*x**2, sd = c*x)

| Item | CSV | Recomputed | Rel. diff | Result |
|---|---|---|---|---|
| a | 1.3804030477145794 | 1.3804030477145794 | 0 | PASS |
| b | 0.10323542514078463 | 0.10323542514078463 | 0 | PASS |
| sd_parameter (c) | 0.5532310661868639 | 0.5532310661868639 | 0 | PASS |
| coverage | 0.8461538461538461 | 0.8461538461538461 | 0 | PASS |
| nll | 4.503770939937339 | 4.503770939937339 | 0 | PASS |

## Verdict

M1: PASS. M2: PASS. All structure checks: PASS.
