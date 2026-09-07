"""Fit every noise-model candidate on data/, report the numbers, and write
the predictions of one candidate into the record.

This script does NOT judge the candidates against each other on held-out
data -- it only fits each one on the full 50-point record and prints what
the fit found. The choice of which candidate to write into the record is a
provisional, defensible-on-its-face pick (stated explicitly below and in
the report), not a scored decision. Scoring/selection is out of scope here
and belongs to a separate evaluation step.

Run:
    python fit_candidates.py
"""
from __future__ import annotations

import numpy as np
from f3dasm import ExperimentData, datagenerator

from noise_models import CANDIDATES


CHOSEN_NAME = "quad_proportional_sd"  # see report for why; provisional, not judged


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
          "once by OLS and reused by all three candidates:")
    any_model = next(iter(fitted.values()))
    print("beta =", any_model.beta)
    return fitted


def write_chosen(fitted, x, y):
    chosen = fitted[CHOSEN_NAME]
    print(f"\nWriting predictions of chosen candidate '{chosen.name}' into data/ "
          f"(provisional -- not yet judged against the other candidates).")

    @datagenerator(output_names=["y_pred_mean", "y_pred_sd", "_source_modeler"])
    def predict(x: float):
        mean, sd = chosen.predict(np.array([x]))
        return float(mean[0]), float(sd[0]), "modeler"

    data = ExperimentData.from_file("data")
    data = data.mark_all("open")
    data = predict.call(data, mode="sequential")
    data.store("data")
    print("Wrote columns y_pred_mean, y_pred_sd, _source_modeler to data/.")


def main():
    x, y = load_xy()
    fitted = fit_all(x, y)
    write_chosen(fitted, x, y)


if __name__ == "__main__":
    main()
