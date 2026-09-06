"""Bayesian linear regression with heteroscedastic (input-dependent) noise.

The baseline (`baseline.py`) fits a degree-2 polynomial mean by least squares
and assumes ONE constant residual standard deviation for every x.  The truth
is `sd[y|x] = 0.5 x`: the noise fans out with speed, and a constant band
cannot represent that.

This block keeps the polynomial mean (degree `d_mean`, default 2) but lets
the log of the noise standard deviation be its own polynomial in x, degree
`d_noise` (default 1):

    sd(x) = exp(c_0 + c_1 t + ... + c_{d_noise} t**d_noise),   t = x / 83

`d_noise = 0` reproduces the baseline's constant-noise assumption exactly (a
useful sanity check); `d_noise >= 1` is what lets the band fan out.

Fitting is type-II maximum likelihood: `tau` (the prior std on the mean
weights) and the noise coefficients `c` are chosen to maximise the log
marginal likelihood of the training targets, via
`scipy.optimize.minimize(method='L-BFGS-B')`.  Given `tau` and `c`, the
posterior over the mean weights is closed form (Bayesian linear regression
with known, input-dependent noise variance):

    Phi          = [1, t, t^2, ..., t^d_mean],           t = x / 83
    W            = diag(1 / sd(x)**2)
    A            = Phi.T @ W @ Phi + I / tau**2
    S            = inv(A)
    m            = S @ Phi.T @ W @ y

Predictive distribution at a new x:

    mean(x)      = phi(x) @ m
    var(x)       = phi(x) @ S @ phi(x) + sd(x)**2

`d_mean` and `d_noise` are the exposed hyperparameters; a `selector` agent can
import `HeteroBLR` and refit at other settings without touching this file.

Writes into the f3dasm record `data/`:

    y_pred_hblr        posterior predictive mean
    sd_hblr             posterior predictive standard deviation
                        (parameter uncertainty + heteroscedastic noise)
    _source_hblr        provenance stamp, 'modeler'

Run:  python blocks/hetero_blr.py
"""

from __future__ import annotations

import sys
from pathlib import Path

import matplotlib
matplotlib.use("Agg")

import matplotlib.pyplot as plt
import numpy as np
from f3dasm import Block, ExperimentData, datagenerator
from scipy.optimize import minimize

# make_data.py lives at the repo root, one level up from blocks/
sys.path.insert(0, str(Path(__file__).resolve().parent.parent))
from make_data import X_HIGH, X_LOW, true_mean, true_sd

X_SCALE = 83.0


class HeteroBLR(Block):
    """Bayesian linear regression, polynomial mean, polynomial-log noise.

    Parameters
    ----------
    d_mean : int
        Degree of the polynomial mean function (default 2).
    d_noise : int
        Degree of the polynomial for log sd(x) (default 1).  0 reproduces a
        constant noise model (the baseline's assumption); >=1 lets the noise
        band fan out with x.
    x_scale : float
        Divide x by this before taking powers, to keep the design matrix
        well conditioned (default 83, the top of the training range).
    """

    def __init__(self, d_mean: int = 2, d_noise: int = 1,
                 x_scale: float = X_SCALE):
        self.d_mean = d_mean
        self.d_noise = d_noise
        self.x_scale = x_scale
        # fitted parameters, populated by fit()
        self.tau = None
        self.c = None
        self.m = None
        self.S = None
        self.nll = None
        self.success = None
        self.message = None

    # -- feature maps --------------------------------------------------
    def _phi_mean(self, x: np.ndarray) -> np.ndarray:
        t = np.asarray(x, dtype=float) / self.x_scale
        return np.vander(t, self.d_mean + 1, increasing=True)

    def _phi_noise(self, x: np.ndarray) -> np.ndarray:
        t = np.asarray(x, dtype=float) / self.x_scale
        return np.vander(t, self.d_noise + 1, increasing=True)

    def _sd(self, x: np.ndarray, c: np.ndarray) -> np.ndarray:
        return np.exp(self._phi_noise(x) @ c)

    # -- fitting: type-II maximum likelihood ----------------------------
    def _neg_log_marginal_likelihood(self, theta: np.ndarray,
                                      Phi: np.ndarray, y: np.ndarray,
                                      x: np.ndarray) -> float:
        log_tau = theta[0]
        c = theta[1:]
        tau = np.exp(log_tau)
        sd = self._sd(x, c)
        n, k = Phi.shape

        Sigma_y = tau ** 2 * (Phi @ Phi.T) + np.diag(sd ** 2)
        try:
            L = np.linalg.cholesky(Sigma_y)
        except np.linalg.LinAlgError:
            return 1e10

        alpha = np.linalg.solve(L.T, np.linalg.solve(L, y))
        log_det = 2.0 * np.sum(np.log(np.diag(L)))
        nll = 0.5 * (y @ alpha + log_det + n * np.log(2 * np.pi))
        return float(nll)

    def fit(self, x: np.ndarray, y: np.ndarray) -> None:
        Phi = self._phi_mean(x)
        theta0 = np.zeros(1 + self.d_noise + 1)
        theta0[0] = 0.0    # log tau = 0 -> tau = 1
        theta0[1] = np.log(np.std(y) + 1e-6)  # constant term of log-sd

        res = minimize(self._neg_log_marginal_likelihood, theta0,
                        args=(Phi, y, x), method="L-BFGS-B")

        self.success = bool(res.success)
        self.message = str(res.message)
        theta = res.x
        self.tau = float(np.exp(theta[0]))
        self.c = theta[1:]
        self.nll = float(res.fun)

        sd = self._sd(x, self.c)
        W = np.diag(1.0 / sd ** 2)
        A = Phi.T @ W @ Phi + np.eye(Phi.shape[1]) / self.tau ** 2
        self.S = np.linalg.inv(A)
        self.m = self.S @ Phi.T @ W @ y

    def predict(self, x):
        """Return (mean, sd) arrays of the posterior predictive at x."""
        x = np.atleast_1d(np.asarray(x, dtype=float))
        Phi = self._phi_mean(x)
        mean = Phi @ self.m
        noise_sd = self._sd(x, self.c)
        var_param = np.einsum("ij,jk,ik->i", Phi, self.S, Phi)
        sd = np.sqrt(var_param + noise_sd ** 2)
        return mean, sd

    # -- f3dasm Block interface -----------------------------------------
    def call(self, data: ExperimentData, **kwargs) -> ExperimentData:
        input_df, output_df = data.to_pandas()
        x = input_df["x"].to_numpy(dtype=float)
        y = output_df["y"].to_numpy(dtype=float)

        self.fit(x, y)

        print("HETERO_BLR  (Bayesian linear regression, "
              f"d_mean={self.d_mean}, d_noise={self.d_noise})")
        print(f"  optimizer success = {self.success}, message = {self.message}")
        print(f"  fitted tau (prior std on mean weights) = {self.tau:.6f}")
        print(f"  fitted log-noise coefficients c         = "
              f"{np.array2string(self.c, precision=6)}")
        print(f"  negative log marginal likelihood         = {self.nll:.4f}")
        print(f"  posterior mean weights m                 = "
              f"{np.array2string(self.m, precision=6)}")

        sd_low = float(self._sd(np.array([X_LOW]), self.c)[0])
        sd_high = float(self._sd(np.array([X_HIGH]), self.c)[0])
        grows = sd_high > sd_low
        print(f"  sanity check: sd(x={X_LOW:.1f}) = {sd_low:.4f}, "
              f"sd(x={X_HIGH:.1f}) = {sd_high:.4f}  "
              f"-> {'GROWS with x (correct)' if grows else 'DOES NOT GROW with x (wrong)'}")
        print(f"  (truth: sd[y|x] = 0.5 x runs over "
              f"[{true_sd(X_LOW):.2f}, {true_sd(X_HIGH):.2f}])")

        m, S, c = self.m, self.S, self.c
        d_mean, d_noise, x_scale = self.d_mean, self.d_noise, self.x_scale

        @datagenerator(output_names=["y_pred_hblr", "sd_hblr",
                                     "_source_hblr"])
        def predict_hblr(x: float):
            t = np.array([x / x_scale])
            phi_m = np.vander(t, d_mean + 1, increasing=True)[0]
            phi_n = np.vander(t, d_noise + 1, increasing=True)[0]
            mean = float(phi_m @ m)
            noise_sd = float(np.exp(phi_n @ c))
            var_param = float(phi_m @ S @ phi_m)
            sd = float(np.sqrt(var_param + noise_sd ** 2))
            return mean, sd, "modeler"

        return predict_hblr.call(data.mark_all("open"), mode="sequential")


def main() -> None:
    data = ExperimentData.from_file("data")
    block = HeteroBLR(d_mean=2, d_noise=1)
    data = block.call(data)
    data.store("data")
    print("\n  wrote y_pred_hblr, sd_hblr, _source_hblr into data/")

    input_df, output_df = data.to_pandas()
    x = input_df["x"].to_numpy(dtype=float)
    y = output_df["y"].to_numpy(dtype=float)

    grid = np.linspace(X_LOW, X_HIGH, 400)
    mean_grid, sd_grid = block.predict(grid)

    fig, ax = plt.subplots(figsize=(7, 4.5))
    ax.scatter(x, y, s=18, color="#333333", zorder=3, label="training data")
    ax.plot(grid, mean_grid, color="#d62728", lw=2,
            label=f"HeteroBLR mean (d_mean={block.d_mean})")
    ax.fill_between(grid, mean_grid - 2 * sd_grid, mean_grid + 2 * sd_grid,
                    color="#d62728", alpha=0.20,
                    label=f"HeteroBLR $\\pm2$ sd (d_noise={block.d_noise})")
    ax.plot(grid, true_mean(grid) + 2 * true_sd(grid), "k--", lw=1.4,
            label="true $\\pm2$ sd (sd = 0.5 x)")
    ax.plot(grid, true_mean(grid) - 2 * true_sd(grid), "k--", lw=1.4)
    ax.set_xlabel("speed x [m/s]")
    ax.set_ylabel("stopping distance y [m]")
    ax.set_title("HeteroBLR: polynomial mean, polynomial log-noise")
    ax.legend(loc="upper left", fontsize=8)
    fig.tight_layout()
    fig.savefig("figures/model.png", dpi=150)
    print("  wrote figures/model.png")


if __name__ == "__main__":
    main()
