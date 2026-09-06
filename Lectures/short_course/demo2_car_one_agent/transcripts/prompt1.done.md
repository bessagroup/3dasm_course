Done. The model is changed, the record and figure are regenerated, and the notebook runs end to end from `data/` alone.

**What was wrong:** the baseline fit the mean by least squares and then called the residual spread "the noise." Least squares *assumes* constant noise — that pipeline could not have found a fanning band no matter how hard it looked. So the fix had to change the fitting, not just the plotting.

**The replacement** (`blocks/heteroscedastic.py`) fits mean and noise jointly by maximum likelihood:

```
mu(x)     = sum_j a_j x^j                  (d_mean  = 2)
log sd(x) = sum_k b_k (log x)^k            (d_noise = 1)
```

The log-link keeps `sd` positive; expanding in `log x` makes the family nest both the old answer (`d_noise=0` → constant band) and the true one (`d_noise=1` → power law). The truth here, `sd = 0.5x`, is the member with `exp(b0)=0.5, b1=1`. The fit was not told that.

**What it printed** (BFGS converged, ‖grad‖ = 1.08e-07):

```
mean coefs [1, x, x^2]          = [-0.3411  1.4159  0.1008]
log-noise coefs [1, log x]      = [-0.8535  1.0572]
  => sd(x) = 0.4259 * x^1.0572
```

It recovered the fan on its own — exponent 1.0572 against a truth of 1, coefficient 0.4259 against 0.5. At the ends of the range: fitted sd 1.361 vs truth 1.500 at x=3, and 45.5 vs 41.5 at x=83, where the baseline says 27.563 at both.

Mean log predictive density on the training record went from **−4.7104 to −4.3096**, a gain of 0.40 nats/point. Meanwhile RMSE of the mean is unchanged to five digits (26.865266 → 26.865286, max prediction difference 0.08 m). That is the point worth putting on the slide: **the entire improvement is in the noise model.** The MLE mean is in fact a hair *worse* in RMSE, on purpose — it weights each point by 1/sd(x)², so it stops chasing the noisy fast cars.

**Into the record:** `y_pred_model`, `sd_model`, `_source_modeler` in `data/`. `sd_model` is a real column now, one value per row, not a scalar repeated 60 times — which is why the notebook redraws `figures/model.png` and recomputes both scores from stored columns with no re-fitting.

One limitation to flag: every number above is on the **training** record. `data_test/` still doesn't exist, so nothing here is held-out evidence, and the log-density gain is exactly the quantity the fit maximized. Choosing `d_mean`/`d_noise` honestly needs a held-out record (or a proper information criterion) — that's `figures/selection.png`, and I've left it undone rather than pretend the training score settles it.
