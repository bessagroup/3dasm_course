"""Score the new model against the baseline on data the fit never saw.

`baseline.py` only ever sees `data/`, so nothing scores it but the person who
wrote it.  A fanning band is a nicer picture than a constant one, but a nicer
picture is not evidence.  This script makes the held-out record that the
baseline never had and scores both models on it.

Both models are re-fitted here on `data/` alone -- the baseline exactly as
`baseline.py` fits it (degree-2 scikit-learn least squares, one constant
residual sd) -- and then evaluated on `data_test/`, which neither fit has seen.

Three models are scored:

    baseline   degree-2 least squares, one constant sd
    hblr       the hand-picked heteroscedastic model (d_mean=2, d_noise=1)
    selected   whatever `blocks/selection.py` chose -- the hyperparameters are
               READ OUT of `data/` (columns d_mean_selected, d_noise_selected,
               link_selected), never retyped here

That last point is what makes this a real held-out check: selection ran on
`data/` with cross-validation and never saw `data_test/`, so this score is the
first time the chosen model meets these points.

Everything lands in the held-out record as output columns:

    y_pred_<tag>, sd_<tag>, lpd_<tag>, _source_<tag>   for each tag above

`lpd_*` is the log predictive density of the *observed* held-out y under each
model, log N(y ; mu(x), sd(x)^2).  It is the metric that can tell the models
apart: they share the same kind of mean, so RMSE cannot.

Run:  python -m blocks.holdout_check [--seed 456]
"""

from __future__ import annotations

import argparse
import os

import numpy as np
from f3dasm import ExperimentData, datagenerator
from sklearn.linear_model import LinearRegression
from sklearn.pipeline import make_pipeline
from sklearn.preprocessing import PolynomialFeatures

from blocks.heteroscedastic import fit as fit_hblr
from make_data import N_TEST, make_record

DEGREE = 2


def fit_baseline(x: np.ndarray, y: np.ndarray):
    """The 2019 baseline, refitted here exactly as `baseline.py` fits it."""
    model = make_pipeline(
        PolynomialFeatures(degree=DEGREE, include_bias=True),
        LinearRegression(fit_intercept=False),
    )
    model.fit(x.reshape(-1, 1), y)
    resid = y - model.predict(x.reshape(-1, 1))
    sd_const = float(np.sqrt(resid @ resid / (len(y) - (DEGREE + 1))))
    return (lambda xq: model.predict(np.atleast_1d(xq).reshape(-1, 1))), sd_const


def log_predictive_density(y, mu, sd):
    """log N(y ; mu, sd^2), elementwise."""
    return -0.5 * np.log(2.0 * np.pi * sd ** 2) - 0.5 * ((y - mu) / sd) ** 2


def main() -> None:
    p = argparse.ArgumentParser(description=__doc__)
    p.add_argument("--seed", type=int, default=456,
                   help="seed for the held-out record (default 456)")
    args = p.parse_args()

    # --- train on data/ only -------------------------------------------------
    train = ExperimentData.from_file("data")
    ti, to = train.to_pandas()
    x_tr = ti["x"].to_numpy(float)
    y_tr = to["y"].to_numpy(float)

    mu_base, sd_base = fit_baseline(x_tr, y_tr)
    hblr = fit_hblr(x_tr, y_tr)                      # module defaults: the hand-pick
    print(f"refitted on data/ ({len(train)} rows): baseline constant sd = "
          f"{sd_base:.4f}; hblr d_mean={hblr['d_mean']}, "
          f"d_noise={hblr['d_noise']}, link='{hblr['link']}'")

    models = [("baseline", mu_base,
               lambda xq: np.full(np.shape(np.atleast_1d(xq)), sd_base)),
              ("hblr", hblr["mu_of"], hblr["sd_of"])]

    # The chosen hyperparameters are read out of the record, not retyped.
    if "d_mean_selected" in to.columns:
        dm = int(to["d_mean_selected"].iloc[0])
        dn = int(to["d_noise_selected"].iloc[0])
        lk = str(to["link_selected"].iloc[0])
        sel = fit_hblr(x_tr, y_tr, dm, dn, lk)
        print(f"  data/ says the selected model is d_mean={dm}, d_noise={dn}, "
              f"link='{lk}' -- refitted from that")
        models.append(("selected", sel["mu_of"], sel["sd_of"]))
    else:
        print("  data/ has no d_mean_selected column yet -- "
              "run `python -m blocks.selection` first to score the chosen model")

    # --- the held-out record the baseline never had --------------------------
    if not os.path.isdir("data_test"):
        print(f"\ndata_test/ does not exist -- making it (seed {args.seed})")
        make_record("data_test", N_TEST, "random_sampler",
                    args.seed, args.seed + 1)
    test = ExperimentData.from_file("data_test")
    xi, xo = test.to_pandas()
    x_te = xi["x"].to_numpy(float)
    y_te = xo["y"].to_numpy(float)
    print(f"\nscoring on data_test/: {len(test)} rows, "
          f"x in [{x_te.min():.2f}, {x_te.max():.2f}] -- neither fit saw these")

    # f3dasm passes only the *input* columns to a datagenerator, so the observed
    # y is taken by position.  mode='sequential' fixes the row order (the same
    # guarantee make_data.py relies on for its noise draws); asserted below.
    for tag, mu_of, sd_of in models:
        row = {"i": 0}

        @datagenerator(output_names=[f"y_pred_{tag}", f"sd_{tag}",
                                     f"lpd_{tag}", f"_source_{tag}"])
        def predict(x: float, _mu=mu_of, _sd=sd_of, _tag=tag, _row=row):
            mu = float(np.atleast_1d(_mu(x))[0])
            sd = float(np.atleast_1d(_sd(x))[0])
            lpd = float(log_predictive_density(y_te[_row["i"]], mu, sd))
            _row["i"] += 1
            return mu, sd, lpd, _tag

        test = predict.call(test.mark_all("open"), mode="sequential")
        assert row["i"] == len(test), (
            f"{tag}: wrote {row['i']} of {len(test)} rows -- look for "
            f"'Error in experiment_sample' above")

    test.store("data_test")
    print("  wrote y_pred_*, sd_*, lpd_*, _source_* into data_test/")

    # --- the verdict, read back out of the record ----------------------------
    _, out = test.to_pandas()
    y = out["y"].to_numpy(float)
    print("\nHELD-OUT SCORES (read back from data_test/, 200 rows)")
    print(f"  {'model':10s} {'mean lpd':>12s} {'RMSE':>10s} "
          f"{'sd(x=3)':>9s} {'sd(x=83)':>9s} {'in 95% band':>12s}")
    tags = [t for t, _, _ in models]
    for label in tags:
        tag = label
        mu = out[f"y_pred_{tag}"].to_numpy(float)
        sd = out[f"sd_{tag}"].to_numpy(float)
        lpd = out[f"lpd_{tag}"].to_numpy(float)
        rmse = float(np.sqrt(np.mean((y - mu) ** 2)))
        cover = float(np.mean(np.abs(y - mu) <= 1.96 * sd)) * 100
        lo = float(np.interp(3.0, np.sort(x_te), sd[np.argsort(x_te)]))
        hi = float(np.interp(83.0, np.sort(x_te), sd[np.argsort(x_te)]))
        print(f"  {label:10s} {lpd.mean():12.4f} {rmse:10.4f} "
              f"{lo:9.3f} {hi:9.3f} {cover:11.1f}%")
    print()
    for tag in tags[1:]:
        d = (out[f"lpd_{tag}"].to_numpy(float)
             - out["lpd_baseline"].to_numpy(float))
        se = float(d.std(ddof=1) / np.sqrt(len(d)))
        print(f"  mean lpd difference ({tag} - baseline) = {d.mean():+.4f} "
              f"+/- {se:.4f} (paired, {len(d)} points); higher density on "
              f"{float(np.mean(d > 0)) * 100:.1f}% of them")
    if "selected" in tags:
        d = out["lpd_selected"].to_numpy(float) - out["lpd_hblr"].to_numpy(float)
        if np.abs(d).max() < 1e-9:
            print("  selected - hblr = 0 exactly: cross-validation landed on "
                  "the same model the\n  hand-pick guessed, so those two "
                  "columns coincide")
        else:
            print(f"  selected - hblr = {d.mean():+.4f} +/- "
                  f"{float(d.std(ddof=1) / np.sqrt(len(d))):.4f} (paired)")

    # Aggregate coverage hides the failure: a band that is far too wide at low
    # x and too narrow at high x still averages out to about 95%.  Split by x.
    edges = np.quantile(x_te, [0.0, 1 / 3, 2 / 3, 1.0])
    print("\nCOVERAGE OF THE 95% BAND, SPLIT BY SPEED (nominal 95.0%)")
    print(f"  {'x range':>16s} {'n':>4s}" + "".join(f"{t:>11s}" for t in tags))
    for lo_e, hi_e in zip(edges[:-1], edges[1:]):
        sel = (x_te >= lo_e) & (x_te <= hi_e)
        cov = []
        for tag in tags:
            mu = out[f"y_pred_{tag}"].to_numpy(float)[sel]
            sd = out[f"sd_{tag}"].to_numpy(float)[sel]
            cov.append(float(np.mean(np.abs(y[sel] - mu) <= 1.96 * sd)) * 100)
        print(f"  [{lo_e:5.1f}, {hi_e:5.1f}] {sel.sum():4d}"
              + "".join(f"{c:10.1f}%" for c in cov))


if __name__ == "__main__":
    main()
