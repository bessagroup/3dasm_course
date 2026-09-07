Done. `python run_quadratic.py` reproduces everything.

## What I built

**`car_ml.py`** — the module, two objects:

- `QuadraticConstantNoise` — plain numpy. `fit` does `np.linalg.lstsq` on the Vandermonde design `[1, x, x²]` for the mean, then takes one constant `sd` from the residual spread. `mean(x)` / `std(x)`.
- `FitQuadraticConstantNoise(Block)` — an f3dasm `Block` using the `arm`/`call` split the base class is designed for: `arm` fits on the record's own `x`,`y`; `call` reopens the jobs and runs a `@datagenerator` that writes the predictions back.

**`run_quadratic.py`** — fits, stores, reads the record *back* from disk, and draws the figure from the stored columns only.

## What the fit printed

```
mu(x) = -0.656207 +1.44641*x +0.100388*x^2
sd    = 29.9218   (constant; n-p = 47 dof)
sd_mle= 29.0102   (same residuals, /n instead of /(n-p))
RSS   = 42079.7   R^2 = 0.9849   n = 50
```

I report `sd = 29.9218` m (unbiased, 47 dof) and print the MLE alongside it so the choice is visible rather than hidden.

## In the record

`data/experiment_data/output.csv` now carries `y, y_pred_quad, sd_quad, _source_quad`, the last stamped `car_ml.FitQuadraticConstantNoise`. Jobs return to `FINISHED`; re-running overwrites the same three columns, so it's idempotent (I ran it twice).

I deliberately did **not** store a residual column: `y` and `y_pred_quad` are both in the record, so `y - y_pred_quad` is already there. Related: a `@datagenerator` only ever receives *domain inputs*, so it cannot see `y` at all — f3dasm raises `ValueError: ... declares required argument(s) ['y'] that are not present in the Domain`.

## The figure

`figures/quadratic_constant_noise.png` — measurements, `mu(x)`, and a `±2sd` band, with a residual panel below.

One thing worth flagging, since you specified the constant sd: the residual panel shows the assumption is the weak part of this model. Residuals near `x = 3–20` sit within a few metres of the curve, while past `x = 55` they scatter by tens of metres — the noise clearly grows with velocity, but a single `sd` has to average that into one band. That band is visibly too wide at low `x` (it dips below `y = 0`, predicting negative stopping distances) and too narrow at high `x`. Coverage came out at 0.92 inside `±2sd` against 0.95 for a Gaussian — close, because being too wide at one end offsets being too narrow at the other. The model as asked for is fitted, stored and plotted; a velocity-dependent `sd` is the obvious next move if you want it.
