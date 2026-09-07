Both subagents' work checks out on disk — files, record columns, and figures all exist as reported. Here's the full picture.

## What happened

**Modeler** built `noise_models.py` (one shared interface: `fit`, `predict(x)->(mean,sd)`, `.name`, `.params`) with three candidates, all sharing one OLS quadratic mean `y = -0.656 + 1.446x + 0.100x²`:
- `quad_const_sd` — the model you asked for by name: constant noise sd.
- `quad_linear_sd` — sd grows linearly with x.
- `quad_proportional_sd` — sd is a constant fraction of the predicted mean.

It added the heteroscedastic candidates because binned residuals visibly grow with x (std ≈5.8 at low speed → ≈53 at high speed) — physically consistent with reaction-time noise (linear) and friction-coefficient noise (proportional). It flagged, rather than silently handling, that the `car_ml.py`/`run_quadratic.py`/`transcripts/` files its own briefing expected to already exist were not actually present in this folder — it built from scratch instead. It fit all three (in-sample log-likelihoods: const −239.33, linear −216.45, proportional −226.49), wrote the **provisional** pick (`quad_proportional_sd`) into `data/` as `y_pred_mean`, `y_pred_sd`, `_source_modeler`, and drew `figures/model.png` from the record on disk. It was explicit this pick was not yet judged.

**Selector** independently scored all three candidates via LOO cross-validation (refitting from scratch each fold) plus a genuine extrapolation test (fit on low-x half, test on high-x half, and vice versa), on three angles: full predictive density, mean-only RMSE (which it correctly noted is blind here since the mean is identical across candidates), and out-of-range coverage. It also checked nesting: `quad_const_sd` is exactly the `b=0` special case of `quad_linear_sd`, so it's dominated in-range and should be dropped. The angles disagreed:
- In-range LOO: `quad_linear_sd` wins (logdens −222.7 vs −230.2 vs −249.4).
- Extrapolation: `quad_linear_sd` catastrophically fails one direction (logdens −1.12×10¹¹) because its linear sd crosses zero and collapses to a floor outside its fitted range; `quad_proportional_sd` is robust both directions and is the only candidate with sane 95% coverage.

Selector's verdict: `quad_const_sd` is dominated and should be dropped; between the two heteroscedastic candidates it explicitly called this a judgment call — best in-range vs. safest under extrapolation — and defended `quad_proportional_sd` as the more trustworthy choice specifically because it can't hit zero/negative by construction. It wrote its own `study_selection/` record and `figures/selection.png`, stamped `_source_selector`, and was explicit about limits: a single median split with only 25 points/side, no way to test outside the observed 3–82 m/s range, and no test of whether the quadratic mean itself is right (all three angles evaluate noise conditional on that mean).

## Reproduce everything

```
python fit_candidates.py && python plot_model.py && python study_selection.py
```
This regenerates `data/`'s prediction columns, `figures/model.png`, `study_selection/`, and `figures/selection.png` from `data/` alone.
