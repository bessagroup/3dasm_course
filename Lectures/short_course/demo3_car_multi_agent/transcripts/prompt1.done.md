## What the two agents did

**`modeler` — changed the model class.** Wrote `blocks/hetero_blr.py`: `HeteroBLR(Block)`, Bayesian linear regression where the mean is a polynomial in `t = x/83` of degree `d_mean` and `log sd(x)` is its own polynomial of degree `d_noise`, so the band can fan. `tau` and the noise coefficients are fit by type-II maximum likelihood on the exact Gaussian log marginal likelihood; the mean posterior is then closed-form. It exposed both degrees as constructor arguments so the selector could refit without editing the block, and it deliberately did not score itself. Its own printed run at the hand-picked `d_mean=2, d_noise=1`:

```
optimizer success = True
fitted tau = 407.458752,  log-noise coefficients c = [1.668179 2.479733]
negative log marginal likelihood = 273.8905
sd(x=3.0) = 5.7997,  sd(x=83.0) = 63.3017   -> grows with x
```

It flagged that this overshoots the true `[1.50, 41.50]` band, partly because `sd_hblr` folds in mean-parameter uncertainty on top of the noise term.

**`selector` — justified the hyperparameters.** Built the held-out record `data_test/` (200 points, seed 456 — training is seed 123), enumerated the 12-candidate `(d_mean, d_noise)` grid by hand as an `ExperimentData`, and scored each on held-out **log predictive density**, not MSE. That choice is the whole demo: look at rows 3–5 below, where MSE is flat at ~505–510 across all three noise models while the log density separates them.

```
    d_mean  d_noise  log_pred_density          mse
0        1        0         -6.440243  2219.081896
1        1        1         -5.237593  8448.582483
2        1        2         -5.047882  3537.359643
3        2        0         -4.576357   505.594556     <- baseline's constant-noise assumption
4        2        1         -4.260417   510.346529     <- modeler's hand-picked guess
5        2        2         -4.239501   508.753890     <- winner
6        3        0         -4.596493   533.069883
...
11       4        2         -4.248931   530.449854
```

Winner `d_mean=2, d_noise=2` beats the modeler's guess by **+0.0209** (small) and the best constant-noise candidate by **+0.3369** (decisive). `d_noise=0` lost at every `d_mean`. The scored grid is stored as its own record in `study_selection/`; the winner was refit on `data/`.

It reported one thing it did not fix: `scipy.optimize.minimize` threw overflow `RuntimeWarning`s on some `d_mean=1` fits (the under-fit corner). All 12 fits completed with finite scores, and it left the modeler's block alone rather than patching it.

## The record

`data/` now carries three writers side by side, each stamped:

| writer | columns | stamp |
|---|---|---|
| `baseline.py` | `y_pred_baseline`, `sd_baseline` | `baseline` |
| `blocks/hetero_blr.py` | `y_pred_hblr`, `sd_hblr` | `modeler` |
| `select_model.py` | `y_pred_selected`, `sd_selected` | `selector` |

And the point of the exercise, in two rows of the record:

```
x =  3.00   truth 0.5x =  1.50   sd_baseline = 27.56   sd_selected =  4.91
x = 81.75   truth 0.5x = 40.88   sd_baseline = 27.56   sd_selected = 50.61
```

The baseline says 27.56 at both ends. The selected model goes 4.91 → 50.61.

`jupyter nbconvert --execute` ran `pipeline.ipynb` end to end without error, regenerating every figure from `data/` alone. New artifacts: `data_test/`, `study_selection/`, `select_model.py`, `blocks/hetero_blr.py`, `figures/model.png`, `figures/selection.png`. `make_data.py`, `baseline.py` and git untouched.
