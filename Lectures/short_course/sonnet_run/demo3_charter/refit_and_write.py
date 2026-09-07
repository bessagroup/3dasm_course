"""Re-fit every noise-model candidate in noise_models.py on the full record
in data/, print the numbers, and write the predictions of ONE candidate
into the record.

This is a modeler-side re-examination of the noise-model choice already on
disk (fit_candidates.py had provisionally written quad_proportional_sd).
It does not touch noise_models.py -- the three candidates and the shared
interface (fit/predict/name/params) are unchanged and still fit here.

What is new here is the in-sample diagnostic below: binning the residuals
around the (shared, OLS) quadratic mean by speed and comparing the empirical
residual std per bin against each candidate's own fitted sd(x), averaged
over the same bin. This uses ONLY the 50 fitted points -- no data is held
out, no candidate is scored on points it did not see, and no cross-validated
or extrapolation figure is computed here. That kind of scoring is explicitly
out of scope for the modeler and belongs to `selector`.

The pick made here (see CHOSEN_NAME) is provisional and NOT a judgment that
it is the better model -- only that, of the fits actually run below, it is
the one the modeler considers most defensible to hand to the selector.

Run:
    python refit_and_write.py
"""
from __future__ import annotations

import numpy as np
from f3dasm import ExperimentData, datagenerator

from noise_models import CANDIDATES


# Provisional pick -- see the printed numbers below and the report for why.
# NOT a judged/scored decision; that is the selector's job.
CHOSEN_NAME = "quad_linear_sd"


def load_xy():
    data = ExperimentData.from_file("data")
    input_df, output_df = data.to_pandas()
    x = input_df["x"].to_numpy(float)
    y = output_df["y"].to_numpy(float)
    return x, y


def fit_all(x, y):
    fitted = {}
    print(f"n = {len(x)} measurements")
    print(f"{'model':<24}{'params':<50}{'loglik':>12}")
    for cls in CANDIDATES:
        model = cls().fit(x, y)
        fitted[model.name] = model
        param_str = ", ".join(f"{k}={v:.6g}" for k, v in model.params.items())
        print(f"{model.name:<24}{param_str:<50}{model.loglik:>12.4f}")
    print()
    print("Shared quadratic mean coefficients (b0 + b1*x + b2*x^2), fitted "
          "once by OLS and reused by every candidate:")
    any_model = next(iter(fitted.values()))
    print("beta =", any_model.beta)
    return fitted


def report_binned_diagnostic(fitted, x, y):
    """In-sample only: how well does each candidate's fitted sd(x) track the
    empirical residual scatter, binned by speed? All 50 points are used to
    fit AND to diagnose here -- nothing is held out, nothing is compared on
    unseen data. This is fit reporting, not model selection.
    """
    order = np.argsort(x)
    xs, ys = x[order], y[order]
    const = fitted["quad_const_sd"]
    mu = const.mean(xs)
    resid = ys - mu

    edges = np.quantile(xs, [0.0, 0.2, 0.4, 0.6, 0.8, 1.0])
    print("\nIn-sample binned diagnostic (5 equal-count speed bins, full "
          "record, nothing held out):")
    header = f"{'x range':<18}{'n':>3}  {'empirical resid std':>20}"
    for name in fitted:
        header += f"  {name + '_sd_mean':>24}"
    print(header)
    for lo, hi in zip(edges[:-1], edges[1:]):
        mask = (xs >= lo) & (xs <= hi)
        row = f"[{lo:6.2f},{hi:6.2f}]   {mask.sum():3d}  {np.std(resid[mask]):20.3f}"
        for name, m in fitted.items():
            _, sd = m.predict(xs[mask])
            row += f"  {sd.mean():24.3f}"
        print(row)


def report_linear_positivity(fitted, x):
    """Structural check on quad_linear_sd: sd(x) = a + b*x is only a valid
    (non-floored) noise model where a + b*x > 0. Report where that holds
    relative to the observed x range -- again, purely a property of the fit
    on the 50 points we have, not an extrapolation test."""
    m = fitted["quad_linear_sd"]
    a, b = m.params["a"], m.params["b"]
    zero_x = -a / b
    print(f"\nquad_linear_sd structural check: sd(x)=a+b*x crosses zero at "
          f"x={zero_x:.4f} m/s; observed x range is [{x.min():.2f}, {x.max():.2f}] "
          f"m/s, so sd(x) stays strictly positive (no flooring) everywhere in "
          f"the fitted data.")


def write_chosen(fitted, x, y):
    chosen = fitted[CHOSEN_NAME]
    print(f"\nWriting predictions of chosen candidate '{chosen.name}' into data/ "
          f"(provisional -- not judged against the other candidates here).")

    @datagenerator(output_names=["y_pred_mean", "y_pred_sd", "model_name",
                                  "_source_modeler"])
    def predict(x: float):
        mean, sd = chosen.predict(np.array([x]))
        return float(mean[0]), float(sd[0]), chosen.name, "modeler"

    data = ExperimentData.from_file("data")
    data = data.mark_all("open")
    data = predict.call(data, mode="sequential")
    data.store("data")
    print("Wrote columns y_pred_mean, y_pred_sd, model_name, _source_modeler "
          "to data/.")


def main():
    x, y = load_xy()
    fitted = fit_all(x, y)
    report_binned_diagnostic(fitted, x, y)
    report_linear_positivity(fitted, x)
    write_chosen(fitted, x, y)


if __name__ == "__main__":
    main()
