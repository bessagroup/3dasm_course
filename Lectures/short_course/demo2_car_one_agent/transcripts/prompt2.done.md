Done. The chain reproduces identical numbers from `make_data.py` onward.

## What "properly" means here

The degrees were previously my taste with a number attached. Now `blocks/selection.py` puts 20 candidates in one table and lets a rule decide: `d_mean` ∈ 1..4, `d_noise` ∈ 0..2, and whether the noise polynomial lives in `log x` or in `x`.

Three choices worth flagging:

- **`d_noise=0` is a constant sd** — so the baseline's noise model is a *candidate in the table*, not an outsider to it. It loses on its merits.
- **The criterion is 6-fold CV log predictive density**, not RMSE. The `cv_rmse` column shows why: it runs 29.5–30.9 across every d_mean≥2 candidate regardless of the noise model. RMSE is blind to exactly the thing being fixed.
- **The training log evidence is stored alongside but not used to select.** It's the right-hand panel of `figures/selection.png` for contrast — and it disagrees: evidence ranks `d_noise=2` best (-276.2374 vs -276.3516) because the noise weights are point-optimised inside it, so it under-pays for a flexible noise model. CV pays the full price.

Selection ran on `data/` only. `data_test/` was never touched by it.

## The verdict

| | d_mean | d_noise | link | cv_lpd | n_params |
|---|---|---|---|---|---|
| best CV score | 2 | 1 | log | **-4.4306 ± 0.1537** | 5 |
| 1-s.e. floor: -4.5843 | | | | 12 of 20 qualify | |
| **selected** (fewest params within 1 s.e.) | **2** | **1** | **log** | -4.4306 | **5** |

The one-s.e. rule picks the same model as the raw best — the simplest qualifier *is* the top scorer. So cross-validation **confirmed** the hand-pick rather than overturning it. That's a weaker result than finding a better model, but it's the honest one, and the hand-pick is now earned rather than asserted.

Against the baseline's noise model, paired on the same folds: **+0.4707 ± 0.1486** (60 points). On the held-out record, which selection never saw: **-4.2294 vs -4.5716**, difference **+0.3422 ± 0.0430**. High-speed coverage 95.5% vs the baseline's 88.1%.

## The record

`study_selection/` — 20 rows, one per candidate, `_source_selection` stamped:

```
   d_mean  d_noise link    cv_lpd  cv_lpd_se   cv_rmse  log_evidence  n_params
6       2        1  log   -4.4306     0.1537   29.5746     -276.3516         5   <- selected
8       2        2  log   -4.4322     0.1529   29.5912     -276.2374         6
7       2        1  lin   -4.4428     0.1391   29.5011     -277.3907         5
...
5       2        0  log   -4.9013     0.2366   30.2284     -297.8919         4   <- baseline's noise
1       1        1  log   -5.6120     0.1239   66.1293     -334.5629         4
```

`data/` — 13 output columns, three writers:

```
_source_baseline  baseline -> [y_pred_baseline, sd_baseline]
_source_hblr      hblr     -> [y_pred_hblr, sd_hblr]
_source_selected  selected -> [y_pred_selected, sd_selected,
                               d_mean_selected, d_noise_selected, link_selected]
```

The last three columns are the point: **the record remembers which model was chosen, not just what it predicted.** `holdout_check.py` reads the hyperparameters out of `data/` rather than having them retyped, so there's no path by which the scored model drifts from the selected one.

## Gates

The notebook re-runs the entire study from `data/` alone (3.3 s) and asserts four things: the re-run matches `study_selection/` to **3.55e-15** across all 20 candidates; `data/` names the same model the notebook just selected; and both re-fits match their stored columns to ~1e-13. `figures/selection.png` and `figures/selected.png` are drawn by functions shared with the block, so the script and the notebook cannot diverge.

One limitation to name: `alpha=1e-6` is fixed, not selected. It's a deliberately weak prior on a standardized basis, so it's doing almost nothing — but it is the one knob still set by hand.
