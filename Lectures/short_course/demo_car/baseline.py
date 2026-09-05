"""The "2019" baseline: a person decided everything.

Degree-2 polynomial least squares for the mean, and one constant residual
standard deviation for the noise.  The mean is fine.  The noise model is
wrong: the truth is sd[y|x] = 0.5 x, which fans out, and a constant band
cannot fan.

Writes into the f3dasm record `data/` (course-canonical way: mark the jobs
open again and run a second @datagenerator whose output_names are new, so
the columns are appended next to `y`):

    y_pred_baseline    least-squares mean prediction
    sd_baseline        one number, repeated on every row
    _source_baseline   provenance stamp, 'baseline'

Writes figures/baseline.png.

There is no held-out record here on purpose: this script only ever sees
`data/`.  Nothing scores this model but the person who wrote it.

Run:  python make_data.py && python baseline.py
"""

from __future__ import annotations

import matplotlib
matplotlib.use("Agg")

import matplotlib.pyplot as plt
import numpy as np
from f3dasm import Block, ExperimentData, datagenerator
from sklearn.linear_model import LinearRegression
from sklearn.pipeline import make_pipeline
from sklearn.preprocessing import PolynomialFeatures

from make_data import X_HIGH, X_LOW, true_mean, true_sd

DEGREE = 2


class QuadraticLeastSquares(Block):
    """The machine learning block, the 2019 way.

    A person decided: a degree-2 polynomial fitted by scikit-learn least
    squares for the mean, and one constant residual standard deviation for the
    noise.  Reads the record, writes y_pred_baseline, sd_baseline and a
    provenance stamp back into it.
    """

    degree = DEGREE

    def call(self, data: ExperimentData, **kwargs) -> ExperimentData:
        input_df, output_df = data.to_pandas()
        x = input_df["x"].to_numpy(dtype=float).reshape(-1, 1)
        y = output_df["y"].to_numpy(dtype=float)

        # --- the model: polynomial mean, one constant noise sd -------------
        model = make_pipeline(
            PolynomialFeatures(degree=self.degree, include_bias=True),
            LinearRegression(fit_intercept=False),
        )
        model.fit(x, y)
        resid = y - model.predict(x)
        sd_const = float(np.sqrt(resid @ resid / (len(y) - (self.degree + 1))))
        self.coef, self.sd_const = model[-1].coef_, sd_const

        print("BASELINE  (degree-2 least squares, constant noise)")
        print(f"  coefficients [1, x, x^2] = "
              f"{np.array2string(self.coef, precision=4)}")
        print(f"  constant residual sd     = {sd_const:.4f}")
        print(f"  train RMSE               = {np.sqrt(np.mean(resid ** 2)):.4f}")
        print(f"  truth: sd[y|x] = 0.5 x runs over "
              f"[{true_sd(X_LOW):.2f}, {true_sd(X_HIGH):.2f}] -- "
              f"one number cannot cover that")

        # f3dasm bookkeeping: reopen the rows, append the new columns
        @datagenerator(output_names=["y_pred_baseline", "sd_baseline",
                                     "_source_baseline"])
        def predict_baseline(x: float):
            return float(model.predict([[x]])[0]), sd_const, "baseline"

        return predict_baseline.call(data.mark_all("open"), mode="sequential")


def main() -> None:
    data = ExperimentData.from_file("data")
    block = QuadraticLeastSquares()
    data = block.call(data)
    data.store("data")
    print("\n  wrote y_pred_baseline, sd_baseline, _source_baseline into data/")

    input_df, output_df = data.to_pandas()
    x = input_df["x"].to_numpy(dtype=float)
    y = output_df["y"].to_numpy(dtype=float)
    coef, sd_const = block.coef, block.sd_const

    # --- figure ------------------------------------------------------------
    grid = np.linspace(X_LOW, X_HIGH, 400)
    mean_grid = np.vander(grid, DEGREE + 1, increasing=True) @ coef

    fig, ax = plt.subplots(figsize=(7, 4.5))
    ax.scatter(x, y, s=18, color="#333333", zorder=3, label="training data")
    ax.plot(grid, mean_grid, color="#1f77b4", lw=2, label="baseline mean (deg 2)")
    ax.fill_between(grid, mean_grid - 2 * sd_const, mean_grid + 2 * sd_const,
                    color="#1f77b4", alpha=0.20,
                    label=f"baseline $\\pm2$ sd (constant, sd = {sd_const:.1f})")
    ax.plot(grid, true_mean(grid) + 2 * true_sd(grid), "k--", lw=1.4,
            label="true $\\pm2$ sd (sd = 0.5 x)")
    ax.plot(grid, true_mean(grid) - 2 * true_sd(grid), "k--", lw=1.4)
    ax.set_xlabel("speed x [m/s]")
    ax.set_ylabel("stopping distance y [m]")
    ax.set_title("Baseline: right mean, wrong noise")
    ax.legend(loc="upper left", fontsize=8)
    fig.tight_layout()
    fig.savefig("figures/baseline.png", dpi=150)
    print("  wrote figures/baseline.png")


if __name__ == "__main__":
    main()
