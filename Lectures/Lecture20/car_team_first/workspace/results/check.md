# Independent check of M1 and M2 (checker, D003)

Recomputed from `data/car_stopping_distance` in `D003/recheck.py` (my own script; nothing from `workspace/results/` was imported). Tolerance: relative difference < 1e-6.

**Overall: PASS**

## Column identity
- `to_pandas()` returns 2 tables: table 0 has column `x` (50 rows), table 1 has column `y` (50 rows). Identified by column name, not position.
- Column-order concern: it changes nothing. The builder's `[0]`=x and `[1]`=y mapping is correct. **PASS** (both models)

## Split sizes (train_test_split, train_size=0.75, random_state=1987)
- Recomputed n_train = 37, n_test = 13. CSV: 37 / 13. **PASS** (both models)

## CSV schema
- M1.csv and M2.csv: header `model,a,b,sd_parameter,coverage,nll,n_train,n_test`, one data row each. **PASS** (both models)

## M1 (CSV vs recomputed)
| check | CSV | recomputed | rel. diff | result |
|---|---|---|---|---|
| a | 1.5567927444618306 | 1.5567927444618306 | 0 | PASS |
| b | 0.10035968486568629 | 0.10035968486568629 | 0 | PASS |
| s (sd_parameter) | 27.983030806957824 | 27.983030806957824 | 0 | PASS |
| coverage | 0.8461538461538461 | 0.8461538461538461 | 0 | PASS |
| nll | 4.976149641475809 | 4.976149641475809 | 0 | PASS |

## M2 (CSV vs recomputed)
| check | CSV | recomputed | rel. diff | result |
|---|---|---|---|---|
| a | 1.3804030477145794 | 1.3804030477145794 | 0 | PASS |
| b | 0.10323542514078463 | 0.10323542514078463 | 0 | PASS |
| c (sd_parameter) | 0.5532310661868639 | 0.5532310661868639 | 0 | PASS |
| coverage | 0.8461538461538461 | 0.8461538461538461 | 0 | PASS |
| nll | 4.503770939937339 | 4.503770939937339 | 0 | PASS |

Coverage is 11 of 13 test rows for both models.
