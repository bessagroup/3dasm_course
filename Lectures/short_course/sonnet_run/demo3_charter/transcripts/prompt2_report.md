Both subagents are done. Here's what happened and what the numbers actually show — including a disagreement I'm not smoothing over.

## Modeler

Reviewed the three heteroscedastic-noise candidates already in `noise_models.py` (all sharing one OLS quadratic mean, differing only in `sd(x)`): `quad_const_sd` (flat), `quad_linear_sd` (`sd=a+b·x`), `quad_proportional_sd` (`sd=k·|mean(x)|`). Using only in-sample diagnostics (binned residual std vs. each candidate's own fitted sd, log-likelihood), it picked **`quad_linear_sd`** as the fix and wrote it into `data/` (`y_pred_mean`, `y_pred_sd`, `model_name`, `_source_modeler`):

```
quad_const_sd         sigma=29.0102              loglik=-239.33
quad_linear_sd        a=-1.118, b=0.606          loglik=-216.45
quad_proportional_sd  k=0.1328                   loglik=-226.49
```

It flagged this pick as provisional and explicitly did not touch held-out data, as its rules require.

## Selector

Ran leave-one-out CV (refit from scratch on 49 points, score the 1 left out — genuinely unseen) plus a median-split extrapolation test both directions (fit low-x half, score high-x half, and vice versa). Direct answer to what you asked:

- **On LOO (unseen points, in-range): yes, the fix helps.** Both heteroscedastic models beat constant-noise: `quad_linear_sd` −222.7, `quad_proportional_sd` −230.2, vs. `quad_const_sd` −249.4 (log predictive density; higher is better). The modeler's pick wins here.
- **On mean-only RMSE: no difference to find** — identical across all three to the printed digits, since none of them touch the mean.
- **On extrapolation (unseen speed range): the answer reverses.** Going low→high, both fixes beat constant-noise (`quad_proportional_sd` −143.8, `quad_linear_sd` −164.2, vs. `quad_const_sd` −409.9, plus proportional gets 100% coverage vs. const's 36%). But going high→low, **`quad_linear_sd` — the modeler's actual pick — collapses catastrophically** (logdens ≈ −1.1×10¹¹): its fitted `sd(x)=a+b·x` has a zero-crossing at x≈1.85 m/s on the full-data fit (just below the observed minimum of 3.00), and that same structural fragility gets triggered when refit on a narrower subset, flooring the sd to ~0 for most low-speed test points. In that direction even `quad_const_sd` (−193.4) beats it, and `quad_proportional_sd` also underperforms constant-noise there (−1057.6 vs −193.4).

So: the constant-noise assumption is worse on held-out data by the full-distribution metric, but the specific fix the modeler wrote into the record is not safely better — it's the most fragile of the three once you test it on stops outside its fitted range in one direction. The selector also flagged that `quad_linear_sd`'s and `quad_proportional_sd`'s fitted sd-curves correlate at 0.979 in-range, so `quad_linear_sd`'s extra parameter buys a small in-range gain (0.15 nats/point) at the cost of exactly the zero-crossing that wrecks it out of range — and separately noted the high→low extrapolation numbers are confounded because the shared quadratic *mean* itself is unstable when fit on a narrow high-x window (predicts μ≈174 at x=3, actual 5.2), so that direction isn't a clean read on the noise model alone.

**Selector did not declare a clean winner** — it reported the three-way disagreement (LOO favors linear, low→high favors proportional, high→low favors neither fix over baseline) rather than averaging it away, per its own rules.

Files: `noise_models.py`, `refit_and_write.py`, `plot_model.py` (modeler); `study_selection.py`, `study_selection/`, `figures/selection.png` (selector); `data/` updated in place. Reproduce with:
```
python refit_and_write.py && python plot_model.py && python study_selection.py
```

**Net takeaway**: I'd flag to you that the record currently holds `quad_linear_sd` as the "fixed" model, but the selector's own held-out evidence says that specific pick is the fragile one outside its fitted range. If you want, the natural next step is asking the modeler to reconsider its pick in light of this (e.g. a noise form that can't cross zero, like `quad_proportional_sd`, or a linear form with a bounded/positive parameterization) — but that's a new modeler task, not something I should quietly patch in myself.
