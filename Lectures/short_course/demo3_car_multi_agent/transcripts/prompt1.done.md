Done. Both agents ran, in order, and the notebook still regenerates every figure from `data/` alone.

## What `modeler` changed

Replaced the model class, in `blocks/hetero_blr.py` (`HeteroBLR`): Bayesian linear regression with a degree-`d_mean` polynomial mean **and** `log sd(x)` as a degree-`d_noise` polynomial — the log keeps sd positive and lets the band fan. `tau` and the noise coefficients are fit jointly by type-II marginal likelihood (L-BFGS-B, Woodbury for the likelihood), so nothing is hand-tuned and no held-out data is touched. It ran at hand-picked defaults `d_mean=2, d_noise=1` and printed:

- fitted `tau = 407.371001`, noise coefficients `c = [1.668271, 2.479521]`
- negative log marginal likelihood `273.8905`, train RMSE `26.8667`
- `sd(x)`: **5.80 at x=3 → 63.29 at x=83** — it grows, where the baseline had 27.5632 everywhere

## What `selector` changed

Built its own held-out record `data_test/` (`make_data.py --test --seed 456`, 200 points — a different seed from the training record's 123), enumerated the 12-candidate grid `d_mean ∈ {1,2,3,4} × d_noise ∈ {0,1,2}` by hand as an `ExperimentData`, and scored every row on the held-out record by mean log predictive density. Printed table:

```
    d_mean  d_noise  log_pred_density          mse
0        1        0         -5.297800  2219.847778
1        1        1         -5.237595  8448.002884
2        1        2         -5.047857  3537.268695
3        2        0         -4.576358   505.594752
4        2        1         -4.260413   510.344816
5        2        2         -4.239478   508.749017   <- winner
6        3        0         -4.596493   533.068639
7        3        1         -4.274361   551.291264
8        3        2         -4.255323   553.219471
9        4        0         -4.592207   525.498690
10       4        1         -4.268940   527.261675
11       4        2         -4.248923   530.449656
```

Winner `d_mean=2, d_noise=2` at `log_pred_density = -4.239478`; refit on `data/` gave `tau = 406.228177`, `c = [1.283171, 4.33337, -1.713519]`. The point worth putting on a slide: **every `d_noise=0` row (3, 6, 9) loses to its `d_noise≥1` siblings at the same `d_mean`, while their `mse` is essentially identical** — mse cannot see the difference, held-out log density can. The baseline's error was invisible to the baseline's own score.

## The record

`data/` is still 60 rows, now 11 columns, three writers side by side and each stamped:

```
x, y,
y_pred_baseline, sd_baseline, _source_baseline
y_pred_hblr,     sd_hblr,     _source_modeler
y_pred_selected, sd_selected, _source_selected
```

The noise column is the whole story:

| | sd_baseline | sd_hblr | sd_selected | truth 0.5x |
|---|---|---|---|---|
| x = 3.0 | 27.56 | 6.60 | 4.91 | 1.50 |
| x = 81.8 | 27.56 | 62.60 | 50.60 | 40.88 |

Plus `study_selection/` (the 12 scored candidates as a record) and `data_test/` (the held-out record). Figures: `baseline.png`, `model.png`, `selection.png`.

One honest caveat from the numbers above: the selected model still overestimates the noise at the low end (4.91 against a true 1.50) — a polynomial in `log sd` fits the fan well where the data is informative but is only loosely pinned at x=3. Both agents reported success; neither reported a failure.
