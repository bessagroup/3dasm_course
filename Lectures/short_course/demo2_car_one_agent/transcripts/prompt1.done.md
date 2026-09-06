Done. The model changed in exactly one place — the noise — so the comparison stays clean.

**What replaced the constant sd** (`blocks/heteroscedastic.py`)

Same degree-2 mean basis; the noise is now a fitted function with an `exp` link (positive by construction, nothing clipped). Two 2-parameter families were fitted and compared by **log marginal likelihood**, so the choice is evidence rather than my taste:

| noise family | fitted c | log evidence |
|---|---|---|
| `power`: log sd = c₀ + c₁·log x | `[-0.7146, +1.0254]` | **-276.3516** |
| `exp-linear`: log sd = c₀ + c₁·x/83 | `[+1.6678, +2.4803]` | -277.3907 |

The winner is `sd(x) = e^-0.7146 · x^1.0254`. The mean weights aren't point-estimated but integrated over (weak Gaussian prior, α=1e-6), so the predictive sd carries an epistemic term too — 4.330% of the variance on average. Fitted sd runs 1.97 → 47.15 across x ∈ [3, 83] against a truth of 1.50 → 41.50; the baseline's single 27.5632 covered none of it.

**That it's actually better, not just prettier** (`blocks/holdout_check.py`)

`baseline.py` never saw held-out data, so a fanning band proves nothing by itself. I made `data_test/` (200 rows, seed 456) and scored both models there:

| model | mean lpd | RMSE | in 95% band |
|---|---|---|---|
| baseline | -4.5716 | 22.5996 | 96.0% |
| hblr | **-4.2294** | 22.5960 | 97.5% |

Mean lpd difference **+0.3422 ± 0.0430** (1 s.e., 200 points) — about 8 standard errors, higher density on 69.0% of points. RMSE is unchanged to four decimals, which is the point: the mean was never the problem, and RMSE cannot see the difference.

The aggregate coverage is misleading for both, so I split it by speed:

| x range | n | baseline | hblr |
|---|---|---|---|
| [3.7, 26.2] | 67 | 100.0% | 97.0% |
| [26.2, 52.4] | 66 | 100.0% | 100.0% |
| [52.4, 83.0] | 67 | **88.1%** | 95.5% |

The baseline's band is absurdly wide at low speed (it dips below zero stopping distance in `figures/model.png`) and **too narrow at 52–83 m/s** — it under-reports risk exactly where a stopping distance matters. Averaging those two failures gives a healthy-looking 96%.

**Record and gate.** `data/` now carries `y_pred_hblr, sd_hblr, _source_hblr` next to the baseline's columns; `data_test/` carries both models' predictions plus per-point `lpd_baseline`/`lpd_hblr`, so every number above was read back out of a record. The notebook runs end to end and redraws `figures/model.png` from `data/` alone — its re-fit matches the stored columns to max |Δμ| = 5.68e-14, max |Δsd| = 7.11e-15, asserted rather than eyeballed.

One thing I chose and you may want to overrule: I picked the noise family from a two-way evidence comparison and left the mean at degree 2 untouched. A real selection study over (d_mean, d_noise) is a different, larger move.
