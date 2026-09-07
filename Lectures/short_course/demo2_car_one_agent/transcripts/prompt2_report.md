Done. `python run_all.py` reproduces everything.

## The fix

The old model averaged one `sd` over every velocity. `car_noise.py` keeps the quadratic mean and lets the noise depend on x, fitting by maximum likelihood — once `sd` varies with x, least squares is gone, because points with small `sd` must be fitted more closely, so the mean is solved by *weighted* least squares inside a search over one or two noise parameters.

Four forms competed, with `const` deliberately among them and fitted by the same code, so only the noise form differs. It reproduced `car_ml`'s numbers exactly (`sd = 29.0102`), which is the check that the new fitter didn't quietly change the old model.

`prop` won: **`sd(x) = 0.5570 · x`**. Two other rows corroborate it rather than merely losing — `affine` was free to keep a constant floor and drove it to `6.2e-14`, and `power` was free to pick any growth rate and picked `q = 1.093`. The data doesn't just want growing noise, it wants proportional noise.

## Better on stops it never saw

Leave-one-out (fit on 49, predict the 50th) and repeated 5-fold (fit on 40, predict 10, ×20), scored by held-out log predictive density:

```
mean held-out log density : -4.4294  vs -4.9874   (+0.5580 nats per stop)
stops where prop scores higher: 35/50
repeated 5-fold           : -4.4392 +/- 0.0452  vs  -5.0237 +/- 0.1005
```

Held-out RMSE barely moves (30.72 vs 31.51) — which is the point: RMSE can't see a noise model, and that's how a constant `sd` passed unnoticed. Calibration is the direct test; standardised held-out residuals should have spread 1.0 in every bin:

```
         x [m/s]      3-17      18-33      36-49      50-66      67-82
    sd(z), const     0.23      0.42      0.78      0.86      2.41
     sd(z), prop     1.18      0.79      1.04      0.78      1.52
```

The old model is 4× too wide at low speed and 2.4× too narrow at high speed; pooling those opposite errors is what made coverage look acceptable at 0.92.

I print a Wilcoxon p of 0.00106 but don't lean on it — the 50 LOO fits share 48 stops each, so those gains aren't independent. The 5-fold column is the cleaner claim.

**Where it loses:** 15 of 50 stops score worse. At `x = 11.75` the new `sd` is 6.3 m against 29.2 m, so a 15 m miss costs 1.115 nats. The largest gain runs far bigger the other way — at `x = 76.75`, 5.13 sd out under the old model vs 2.98 under the new, worth +8.21 nats.

Read as physics, the fit gives a 1.42 s mean reaction time, braking at 4.965 m/s² (0.506 g), an intercept 0.15% of mean `y`, and a scatter of 0.557 s of travel — 39% of the reaction time. That explains *why* proportional: if drivers differ in reaction time and brake alike, the scatter enters only through `t·x`. Consistent with, not proven by, 50 stops.

## Record and figures

`data/` keeps the `*_quad` columns and gains `y_pred_prop, sd_prop, _source_prop` plus, for both models, `y_pred_loo_*, sd_loo_*, logpd_loo_*, _source_loo_*` — the evidence per stop, no entry influenced by the measurement it's scored against. The candidate table is a second record, `study_noise/`. Both figures are drawn from the stored columns after reading the record back.

- `figures/noise_grows_with_velocity.png` — the fanning band, with the old one dashed over it
- `figures/held_out_comparison.png` — held-out residuals per model with per-bin spread against ±1, and the per-stop gain
- `transcripts/prompt2_report.md` — the full write-up

I tested reproducibility the hard way: a clean copy of the folder as committed, plus these scripts, `python run_all.py`. All three figures came out byte-identical and every column I write matched exactly. No git operations.
