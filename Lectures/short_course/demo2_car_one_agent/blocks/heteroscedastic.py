"""A heteroscedastic maximum-likelihood model: the noise is allowed to fan.

The baseline got the mean right and the noise wrong.  It fitted the mean by
least squares and then called `np.std(residuals)` the noise -- one number for
the whole speed range.  Least squares *assumes* constant noise, so that pipeline
could never have discovered anything else.

Here mean and noise are fitted **together**, by maximising the Gaussian
log-likelihood

    log p(y | x) = - log sd(x) - (y - mu(x))^2 / (2 sd(x)^2) + const

with

    mu(x)     = sum_{j=0..d_mean}  a_j x^j
    log sd(x) = sum_{k=0..d_noise} b_k (log x)^k

The log-link keeps sd positive by construction, and expanding it in `log x`
makes the family nest the baseline and the truth:

    d_noise = 0  ->  sd(x) = exp(b0)            constant  (the baseline)
    d_noise = 1  ->  sd(x) = exp(b0) x^{b1}     a power law
    d_noise = 2  ->  curvature in log-log

The truth of this problem, sd[y|x] = 0.5 x, is the d_noise = 1 member with
exp(b0) = 0.5 and b1 = 1.  The fit is not told that; whether it recovers it is
something the printed numbers have to show.

Writes into the record `data/`:

    y_pred_model     mean prediction   mu(x)
    sd_model         noise prediction  sd(x)   <-- varies row to row now
    _source_modeler  provenance stamp, 'modeler'

Writes figures/model.png.

Run:  python -m blocks.heteroscedastic
"""

from __future__ import annotations

import matplotlib

matplotlib.use("Agg")

import matplotlib.pyplot as plt
import numpy as np
from f3dasm import Block, ExperimentData, datagenerator
from scipy.optimize import minimize

from make_data import X_HIGH, X_LOW, true_mean, true_sd

D_MEAN = 2
D_NOISE = 1


def mean_design(x, d_mean: int) -> np.ndarray:
    """[1, x, x^2, ...] up to degree d_mean."""
    return np.vander(np.atleast_1d(np.asarray(x, float)), d_mean + 1,
                     increasing=True)


def noise_design(x, d_noise: int) -> np.ndarray:
    """[1, log x, (log x)^2, ...] up to degree d_noise."""
    return np.vander(np.log(np.atleast_1d(np.asarray(x, float))), d_noise + 1,
                     increasing=True)


class FitResult:
    """What the optimiser did.  `success` is judged on the gradient, not on
    scipy's line-search verdict (see the note in fit_heteroscedastic)."""

    def __init__(self, success, grad_inf, nit, message, nll):
        self.success, self.grad_inf = success, grad_inf
        self.nit, self.message, self.nll = nit, message, nll


GTOL = 1e-8


def fit_heteroscedastic(x, y, d_mean: int = D_MEAN, d_noise: int = D_NOISE):
    """Joint MLE of (a, b).  Returns (a, b, negative log-likelihood, result).

    A raw polynomial design [1, x, ..., x^d] on x in [3, 83] is viciously
    ill-conditioned: cond = 1e2 at d = 1 but 1e8 at d = 4.  BFGS then stalls in
    the line search and returns "precision loss" *before* reaching the optimum,
    and it stalls worse the higher the degree -- which would quietly rig any
    comparison across degrees against the flexible models.

    So the optimiser is handed an orthonormalised design instead (QR of the
    Vandermonde on the training points; condition number ~ 1 by construction),
    and the coefficients are mapped back to the raw [1, x, x^2, ...] basis at
    the end.  The model is identical -- only the coordinates the search runs in
    change -- so everything downstream is untouched.
    """
    x = np.asarray(x, float)
    y = np.asarray(y, float)
    M_raw = mean_design(x, d_mean)
    L_raw = noise_design(x, d_noise)

    Rm_inv = np.linalg.inv(np.linalg.qr(M_raw, mode="r"))
    Rl_inv = np.linalg.inv(np.linalg.qr(L_raw, mode="r"))
    M, L = M_raw @ Rm_inv, L_raw @ Rl_inv          # orthonormal columns

    # Start from the baseline's answer: least-squares mean, constant sd.
    aq0 = np.linalg.lstsq(M, y, rcond=None)[0]
    sd0 = np.std(y - M @ aq0, ddof=min(d_mean + 1, len(y) - 1))
    # constant log-sd expressed in the orthonormalised noise basis
    bq0 = np.log(sd0) * np.linalg.inv(Rl_inv)[:, 0]

    def nll(theta):
        a, b = theta[: d_mean + 1], theta[d_mean + 1:]
        log_sd = L @ b
        r = (y - M @ a) * np.exp(-log_sd)
        return float(np.sum(log_sd) + 0.5 * float(r @ r))

    def grad(theta):
        a, b = theta[: d_mean + 1], theta[d_mean + 1:]
        log_sd = L @ b
        inv_var = np.exp(-2.0 * log_sd)
        r = y - M @ a
        g_a = -M.T @ (r * inv_var)
        g_b = L.T @ (1.0 - r ** 2 * inv_var)
        return np.concatenate([g_a, g_b])

    res = minimize(nll, np.concatenate([aq0, bq0]), jac=grad, method="BFGS",
                   options={"maxiter": 10000, "gtol": GTOL})
    aq, bq = res.x[: d_mean + 1], res.x[d_mean + 1:]

    # scipy reports "precision loss" whenever the line search cannot improve,
    # which happens routinely *at* a flat optimum.  The gradient is the honest
    # test, so it is what `success` reports.
    grad_inf = float(np.max(np.abs(grad(res.x))))
    out = FitResult(bool(grad_inf < 1e-4), grad_inf, int(res.nit),
                    str(res.message), float(res.fun))
    return Rm_inv @ aq, Rl_inv @ bq, float(res.fun), out


def mean_of(x, a, d_mean: int = D_MEAN) -> np.ndarray:
    return mean_design(x, d_mean) @ a


def sd_of(x, b, d_noise: int = D_NOISE) -> np.ndarray:
    return np.exp(noise_design(x, d_noise) @ b)


def mean_log_density(x, y, a, b, d_mean=D_MEAN, d_noise=D_NOISE) -> float:
    """Average Gaussian log predictive density -- the score that judges noise."""
    mu = mean_of(x, a, d_mean)
    sd = sd_of(x, b, d_noise)
    return float(np.mean(-np.log(sd) - 0.5 * ((y - mu) / sd) ** 2
                         - 0.5 * np.log(2 * np.pi)))


class HeteroscedasticMLE(Block):
    """Fit mu(x) and sd(x) jointly by maximum likelihood; write both back."""

    def __init__(self, d_mean: int = D_MEAN, d_noise: int = D_NOISE):
        self.d_mean, self.d_noise = d_mean, d_noise

    def call(self, data: ExperimentData, **kwargs) -> ExperimentData:
        input_df, output_df = data.to_pandas()
        x = input_df["x"].to_numpy(float)
        y = output_df["y"].to_numpy(float)

        a, b, nll, res = fit_heteroscedastic(x, y, self.d_mean, self.d_noise)
        self.a, self.b = a, b

        # The baseline, refitted here only so the two are scored the same way.
        M = mean_design(x, self.d_mean)
        a_ls = np.linalg.lstsq(M, y, rcond=None)[0]
        b_ls = np.array([np.log(np.std(y - M @ a_ls, ddof=self.d_mean + 1))])

        lpd_new = mean_log_density(x, y, a, b, self.d_mean, self.d_noise)
        lpd_base = mean_log_density(x, y, a_ls, b_ls, self.d_mean, 0)

        print(f"HETEROSCEDASTIC MLE  (d_mean = {self.d_mean}, "
              f"d_noise = {self.d_noise})")
        print(f"  optimiser converged      = {res.success} "
              f"in {res.nit} iterations, |grad|inf = {res.grad_inf:.2e}")
        print(f"  mean coefs [1, x, ...]   = {np.array2string(a, precision=4)}")
        print(f"  log-noise coefs [1, log x, ...] = "
              f"{np.array2string(b, precision=4)}")
        print(f"  => sd(x) = {np.exp(b[0]):.4f} * x^{b[1]:.4f}"
              if self.d_noise == 1 else "")
        print(f"  negative log-likelihood  = {nll:.4f}")
        print("  sd at the ends of the range:")
        for xv in (X_LOW, X_HIGH):
            print(f"    x = {xv:5.1f}:  fitted sd = "
                  f"{float(sd_of(xv, b, self.d_noise)[0]):7.3f}   "
                  f"truth 0.5x = {float(true_sd(xv)):7.3f}   "
                  f"baseline = {float(np.exp(b_ls[0])):7.3f}")
        print("  mean log predictive density on the training record "
              "(higher is better):")
        print(f"    baseline (constant sd) = {lpd_base:.4f}")
        print(f"    this model             = {lpd_new:.4f}   "
              f"(gain {lpd_new - lpd_base:+.4f} nats/point)")

        d_mean, d_noise = self.d_mean, self.d_noise

        @datagenerator(output_names=["y_pred_model", "sd_model",
                                     "_source_modeler"])
        def predict_model(x: float):
            return (float(mean_of(np.array([x]), a, d_mean)[0]),
                    float(sd_of(np.array([x]), b, d_noise)[0]),
                    "modeler")

        return predict_model.call(data.mark_all("open"), mode="sequential")


def main() -> None:
    data = ExperimentData.from_file("data")
    block = HeteroscedasticMLE()
    data = block.call(data)
    data.store("data")
    print("\n  wrote y_pred_model, sd_model, _source_modeler into data/")

    input_df, output_df = data.to_pandas()
    x = input_df["x"].to_numpy(float)
    y = output_df["y"].to_numpy(float)
    sd_base = float(output_df["sd_baseline"].iloc[0])

    grid = np.linspace(X_LOW, X_HIGH, 400)
    mu_g = mean_of(grid, block.a)
    sd_g = sd_of(grid, block.b)

    fig, ax = plt.subplots(figsize=(7, 4.5))
    ax.scatter(x, y, s=18, color="#333333", zorder=3, label="training data")
    ax.plot(grid, mu_g, color="#d62728", lw=2, label="MLE mean")
    ax.fill_between(grid, mu_g - 2 * sd_g, mu_g + 2 * sd_g, color="#d62728",
                    alpha=0.20, label="MLE $\\pm2$ sd (fanning)")
    ax.plot(grid, true_mean(grid) + 2 * true_sd(grid), "k--", lw=1.4,
            label="true $\\pm2$ sd (sd = 0.5 x)")
    ax.plot(grid, true_mean(grid) - 2 * true_sd(grid), "k--", lw=1.4)
    ax.plot(grid, mean_of(grid, block.a) + 2 * sd_base, color="#1f77b4",
            lw=1.2, ls=":", label="baseline $\\pm2$ sd (constant)")
    ax.plot(grid, mean_of(grid, block.a) - 2 * sd_base, color="#1f77b4",
            lw=1.2, ls=":")
    ax.set_xlabel("speed x [m/s]")
    ax.set_ylabel("stopping distance y [m]")
    ax.set_title("Heteroscedastic MLE: the band fans with the truth")
    ax.legend(loc="upper left", fontsize=8)
    fig.tight_layout()
    fig.savefig("figures/model.png", dpi=150)
    print("  wrote figures/model.png")


if __name__ == "__main__":
    main()
