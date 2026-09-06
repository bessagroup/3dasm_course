Done. Selection is by cross-validation with a principled tiebreak, and everything lives in records.

**The method.** 12 candidates (`d_mean` 1–4 × `d_noise` 0–2), scored by 5-fold × 5-repeat CV on `data/` alone — 300 fits, each seeing 48 points and scored on the 12 it never saw. The criterion is held-out **log predictive density**, a proper scoring rule that grades the whole predictive distribution. RMSE is carried alongside purely to show what it can't do.

**One thing had to be fixed first, and it mattered.** The first run reported `n_fit_failures` of 12/25 for the winner and 25/25 for several candidates. That wasn't cosmetic: the failures tracked the Vandermonde condition number exactly (1e2 at `d_mean=1` → 1e8 at `d_mean=4`), so BFGS stalled *worse the more flexible the model* — biasing the comparison against the high-degree candidates, which is precisely the direction that would have made my hand-picked (2,1) win for a fake reason. The optimiser now runs on a QR-orthonormalised design and maps coefficients back to the raw basis. All 300 fits converge (‖grad‖∞ ~1e-9), and the likelihood is now monotone in parameter count within each nested family — the check that says these are real optima. The ranking survived the fix; it just means something now.

**`study_selection/` — the record:**

```
 d_mean  d_noise  cv_log_pred_density  cv_lpd_se   cv_rmse  n_params  n_fit_failures
      2        1            -4.429031   0.015330 27.828325         5               0
      3        1            -4.476395   0.031548 28.152560         6               0
      2        2            -4.507796   0.043466 27.867381         6               0
      3        2            -4.628038   0.128153 28.217646         7               0
      4        1            -4.669103   0.054554 28.870444         7               0
      2        0            -4.849147   0.019392 28.113636         4               0   <- the baseline's choice
      ...
      1        1            -5.582341   0.036036 64.902833         4               0
```

The top two differ by 0.0474 nats with pooled se 0.0351 — **not separated**. So the argmax is not a fact, and the pick is made by the **one-standard-error rule** (keep everything within 1 se, take the fewest parameters). Here only one candidate survives the threshold, so both rules agree: **`d_mean=2, d_noise=1`**, refit on all 60 points to `sd(x) = 0.4259·x^1.0572` against a truth of `0.5·x`.

**Why the scoring rule had to be the lpd.** RMSE happens to land on the same cell, but by 1.0% along that row versus 0.42 nats — and it never evaluates `sd` at all: inflating the band 10× leaves RMSE at 26.865286, identical to six digits, while the lpd collapses from −4.31 to −6.12. The right panel of `figures/selection.png` shows this directly — RMSE spends its whole colour range separating `d_mean=1`, leaving rows 2–4 a flat block where `d_noise` is invisible.

**Independent confirmation.** I made the held-out record (`python make_data.py --test --seed 456`, 200 points) and scored once. Selected beats the baseline's choice by **+0.3429 nats/point** on data no fit or fold ever saw, and edges the CV runner-up by 0.0002 — consistent with CV saying they weren't separated.

The sharpest number is the calibration. Aggregate ±2sd coverage barely distinguishes them (0.9600 baseline vs 0.9750 selected, nominal 0.9545) — but split by speed it stops cancelling:

```
                                     x in [3.7,26.2]  [26.2,52.4]  [52.4,83.0]
baseline's choice (constant band)             1.0000       1.0000       0.8806
selected (d_mean=2, d_noise=1)                0.9701       1.0000       0.9552
```

The constant band is not merely imprecise, it is wrong in opposite directions at the two ends — vacuous for slow cars (claims ±26.9 m where the truth is ±2.5) and overconfident for fast ones, missing 12% of them at 2 sd. Averaged over `x` the two errors hide each other, which is why the aggregate looked fine.

**Files:** `blocks/selection.py`, `blocks/holdout_check.py`, `study_selection/`, `data/` gained `y_pred_selected, sd_selected, _source_selected` (11 columns, 3 provenance stamps), `data_test/` gained its prediction columns, plus `figures/selection.png` and `figures/selected.png`. The notebook runs clean end to end and reads only `data/` and `study_selection/` — `holdout_check.py` is the only thing that ever opens `data_test/`.

One caveat: `cv_lpd_se` is the spread across the 5 repeats, which is optimistic as a standard error since the repeats share all 60 points. It's adequate for "are these two cells separated," but I wouldn't quote it as a confidence interval.
