Both subagents are done. Here's what happened, in delegation order.

## Modeler

**Built:** `car_noise_models.py` — one shared interface (`fit`, `predict(x)->(mean,sd)`, `.name`, fitted params) implementing three noise candidates, all sharing the same quadratic OLS mean `mu(x) = a·x² + b·x + c`:
- `constant` — the single homoscedastic sd the user asked for
- `powerlaw` — sd(x) = c·xᵖ
- `proportional` — sd(x) = k·|mu(x)| (constant coefficient of variation)

`fit_noise_models.py` fits all three and prints the numbers. `draw_model_figure.py` re-reads the record from disk and draws `figures/model.png`.

**Why:** the modeler argued constant-sigma alone is worth testing but not worth assuming — a tercile diagnostic showed in-sample residual std growing ~6.6× from low to high speed, which a constant model can't represent, and stopping distance's braking term (~v²) has a physical argument for multiplicative (proportional) rather than additive noise.

**Numbers it printed:** quadratic mean `mu(x)=0.100388x² + 1.446409x − 0.656207`; `constant` sigma=29.010249 (loglik −239.33); `powerlaw` c=0.402639, p=1.091171 (loglik −216.75); `proportional` k=0.132771 (loglik −226.49).

**Wrote to record:** it picked `proportional` on physical grounds and wrote `y_pred_proportional`, `sd_proportional`, `_source_proportional` into `data/` — explicitly flagging this as **provisional and unjudged**, since it (correctly, per its rules) refused to use in-sample log-likelihood to rank candidates itself.

## Selector

**Did:** built `selector_loo_cv.py` (didn't touch the modeler's files) and ran true leave-one-out CV — refitting the whole model (mean + noise) on 49 points, scoring the held-out one, for all three candidates. Compared via mean log predictive density, calibration (z-score std, interval coverage), and paired tests (t-test + Wilcoxon) since all three share folds. Wrote `study_loo_folds/`, `study_loo_cv/`, `study_pairwise/` (all stamped `_source_selector`) and `figures/selection.png`.

**Numbers:** LOO mean log-density — constant −4.987, powerlaw **−4.481** (best), proportional −4.604. Calibration (z std, ideal=1.0) — constant 1.198, powerlaw 1.151, proportional **1.038** (best). Powerlaw beats constant with real support (paired t p=0.007). Powerlaw vs. proportional is **not resolved** — t-test p=0.28 vs. Wilcoxon p=0.035 disagree, and the ranking is sensitive to a single high-leverage held-out point (x=76.75).

**What it says the data doesn't show:** with 50 points total, constant can be ruled out relative to the others, but powerlaw vs. proportional (the physically interesting comparison) is not settled by this test — one point moves the result, folds aren't independent, extrapolation outside [3, 81.75] is untested, and Gaussian-residual noise itself is assumed, not verified. Selector's explicit conclusion: don't report either as "shown by the data" to beat the other.

## Bottom line
The record now holds the modeler's provisional pick (`proportional`) plus the selector's independent judgment that it's statistically indistinguishable from `powerlaw`, with `constant` (what you originally asked for) ranked worst by this evidence. Neither subagent overstepped: modeler never scored its own candidates, selector never touched model code.

**To reproduce everything:**
```
python fit_noise_models.py && python draw_model_figure.py && python selector_loo_cv.py
```
