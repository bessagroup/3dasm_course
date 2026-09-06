"""The confirmation: score the selected model on a record it has never seen.

The selection in `blocks/selection.py` used `data/` and nothing else -- that is
what makes it reproducible from the record, and it is why `pipeline.ipynb` can
redraw it.  But cross-validation reuses the same 60 points 25 times, so it is
still an estimate made on the training record.  This script is the independent
check: fit on `data/`, predict on `data_test/`, score once, and stop.

`data_test/` did not exist until someone made it, deliberately.  It is made by

    python make_data.py --test --seed 456          # 200 points, unseen

Three models, all fitted on `data/` alone, all scored on the same held-out
points:

    the baseline's choice      d_mean = 2, d_noise = 0   (constant band)
    the selected model         d_mean, d_noise from study_selection/
    the CV runner-up           the second row of that table

Writes the predictions into `data_test/` as columns, so the numbers below can
be recomputed from the record rather than taken on trust:

    y_pred_selected, sd_selected, y_pred_constant, sd_constant, _source_holdout

Note this is the ONLY thing in the demo that touches `data_test/`.
`pipeline.ipynb` never reads it: the reproduction gate is about `data/`.

Run:  python -m blocks.holdout_check
"""

from __future__ import annotations

import numpy as np
from f3dasm import ExperimentData, datagenerator

from blocks.heteroscedastic import (fit_heteroscedastic, mean_log_density,
                                    mean_of, sd_of)


def main() -> None:
    train = ExperimentData.from_file("data")
    ti, to = train.to_pandas()
    x, y = ti["x"].to_numpy(float), to["y"].to_numpy(float)

    test = ExperimentData.from_file("data_test")
    vi, vo = test.to_pandas()
    xt, yt = vi["x"].to_numpy(float), vo["y"].to_numpy(float)
    print(f"fit on data/ ({len(x)} points), scored on data_test/ "
          f"({len(xt)} points, never seen by any fit or any fold)")

    # which model the cross-validation chose, read back from its own record
    study = ExperimentData.from_file("study_selection")
    si, so = study.to_pandas()
    table = si.join(so).sort_values("cv_log_pred_density", ascending=False)
    best = table.iloc[0]
    runner = table.iloc[1]
    d_mean, d_noise = int(best["d_mean"]), int(best["d_noise"])
    print(f"  study_selection/ says: d_mean = {d_mean}, d_noise = {d_noise}\n")

    candidates = [
        ("baseline's choice (constant band)", 2, 0),
        (f"selected (d_mean={d_mean}, d_noise={d_noise})", d_mean, d_noise),
        (f"CV runner-up (d_mean={int(runner['d_mean'])}, "
         f"d_noise={int(runner['d_noise'])})",
         int(runner["d_mean"]), int(runner["d_noise"])),
    ]

    fits, results = {}, []
    for label, dm, dn in candidates:
        a, b, _, res = fit_heteroscedastic(x, y, dm, dn)
        if not res.success:
            print(f"  !! {label}: optimiser did NOT converge "
                  f"(|grad|inf = {res.grad_inf:.2e}) -- not scored")
            continue
        lpd = mean_log_density(xt, yt, a, b, dm, dn)
        rmse = float(np.sqrt(np.mean((yt - mean_of(xt, a, dm)) ** 2)))
        fits[(dm, dn)] = (a, b)
        results.append((label, dm, dn, lpd, rmse))

    print(f"{'model':40s} {'held-out lpd':>13s} {'held-out RMSE':>14s}")
    for label, dm, dn, lpd, rmse in results:
        print(f"{label:40s} {lpd:13.4f} {rmse:14.4f}")

    base = next(r for r in results if (r[1], r[2]) == (2, 0))
    sel = next(r for r in results if (r[1], r[2]) == (d_mean, d_noise))
    print(f"\nselected minus baseline: {sel[3] - base[3]:+.4f} nats/point "
          f"on data the fit never saw")
    if sel[3] >= max(r[3] for r in results):
        print("  the CV pick is also the best of these on the held-out record")
    else:
        print("  NOTE: the CV pick is NOT the best here -- the selection did "
              "not generalise")

    # how honest is the band?  a calibration check the lpd alone does not show
    print("\ncalibration on the held-out record "
          "(fraction of points inside +/-2 sd; nominal 0.9545):")
    for label, dm, dn, _, _ in results:
        a, b = fits[(dm, dn)]
        inside = np.abs(yt - mean_of(xt, a, dm)) <= 2 * sd_of(xt, b, dn)
        print(f"  {label:40s} {inside.mean():.4f}")
    print("  -- read on before believing this: aggregated over the whole "
          "speed range it barely separates them.")

    # The aggregate hides the defect.  A constant band is too WIDE for slow
    # cars and too NARROW for fast ones; averaged over x the two errors cancel
    # and the coverage looks fine.  Split by speed and it stops cancelling.
    edges = np.quantile(xt, [0.0, 1 / 3, 2 / 3, 1.0])
    print(f"\nthe same check split into thirds of the speed range "
          f"(nominal 0.9545 in every bin):")
    header = "".join(
        f"{f'x in [{edges[k]:.1f}, {edges[k + 1]:.1f}]':>24s}"
        for k in range(3))
    print(f"{'model':40s}{header}")
    for label, dm, dn, _, _ in results:
        a, b = fits[(dm, dn)]
        inside = np.abs(yt - mean_of(xt, a, dm)) <= 2 * sd_of(xt, b, dn)
        cells = ""
        for k in range(3):
            m = (xt >= edges[k]) & (xt <= edges[k + 1])
            cells += f"{inside[m].mean():24.4f}"
        print(f"{label:40s}{cells}")

    # and what the two models actually claim, in metres, at either end
    a_s, b_s = fits[(d_mean, d_noise)]
    a_c, b_c = fits[(2, 0)]
    print("\nthe width of the claim, in metres:")
    for xv in (5.0, 80.0):
        print(f"  at x = {xv:4.1f} m/s:  constant band sd = "
              f"{float(sd_of(xv, b_c, 0)[0]):7.3f}   "
              f"selected sd = {float(sd_of(xv, b_s, d_noise)[0]):7.3f}   "
              f"truth 0.5x = {0.5 * xv:7.3f}")

    # --- write the predictions into data_test/ so this is a record too ------
    a_sel, b_sel = fits[(d_mean, d_noise)]
    a_con, b_con = fits[(2, 0)]

    @datagenerator(output_names=["y_pred_selected", "sd_selected",
                                 "y_pred_constant", "sd_constant",
                                 "_source_holdout"])
    def predict_both(x: float):
        return (float(mean_of(np.array([x]), a_sel, d_mean)[0]),
                float(sd_of(np.array([x]), b_sel, d_noise)[0]),
                float(mean_of(np.array([x]), a_con, 2)[0]),
                float(sd_of(np.array([x]), b_con, 0)[0]),
                "holdout_check")

    test = predict_both.call(test.mark_all("open"), mode="sequential")
    test.store("data_test")
    print("\n  wrote y_pred_selected, sd_selected, y_pred_constant, "
          "sd_constant, _source_holdout into data_test/")


if __name__ == "__main__":
    main()
