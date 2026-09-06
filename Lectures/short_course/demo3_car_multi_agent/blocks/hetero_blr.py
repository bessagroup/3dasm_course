"""Bayesian linear regression with heteroscedastic (input-dependent) noise.

The 2019 baseline (`baseline.py`) fits a degree-2 least-squares mean and then
assumes ONE constant residual standard deviation for every point.  The truth
is `sd[y|x] = 0.5 x`: the noise band fans out with speed, and a constant
band cannot follow it.

This block keeps the polynomial mean (now with a Bayesian prior instead of a
point estimate) and lets the log of the noise standard deviation be its own
polynomial in `x`:

    mean:  phi(x) @ w,             phi(x) = [1, t, t^2, ..., t^d_mean]
    noise: sd(x) = exp(c_0 + c_1 t + ... + c_{d_noise} t^d_noise)

    t = x / 83   (the input is scaled so the powers don't blow up)

`tau` (the prior std on the mean weights) and `c` (the noise-log-poly
coefficients) are fit jointly by type-II maximum likelihood: maximise the
log marginal likelihood of the training targets under the linear-Gaussian
model, using `scipy.optimize.minimize(method='L-BFGS-B')`.  Given `tau` and
`c`, the mean posterior is closed form (Bayesian linear regression with
known, `x`-dependent noise variance):

    W = diag(1 / sd(x)**2)
    A = Phi.T @ W @ Phi + I / tau**2
    S = inv(A)
    m = S @ Phi.T @ W @ y

Predictive distribution at a new x:

    mean(x) = phi(x) @ m
    var(x)  = phi(x) @ S @ phi(x) + sd(x)**2

`d_mean` and `d_noise` are explicit, settable constructor arguments so a
later `selector` can refit this exact class at other degrees without editing
this file.

Writes into the f3dasm record `data/` (course-canonical: mark jobs open,
append new output columns):

    y_pred_hblr     posterior predictive mean
    sd_hblr         posterior predictive standard deviation
    _source_modeler provenance stamp, 'modeler'

Writes figures/model.png.

Run:  python blocks/hetero_blr.py
"""

from __future__ import annotations

import sys
from pathlib import Path

import matplotlib
matplotlib.use("Agg")

import matplotlib.pyplot as plt
import numpy as np
from scipy.optimize import minimize

from f3dasm import Block, ExperimentData, datagenerator

# Make `make_data` importable whether this file is run from the repo root or
# as `python blocks/hetero_blr.py`.
sys.path.insert(0, str(Path(__file__).resolve().parent.parent))
from make_data import X_HIGH, X_LOW, true_mean, true_sd  # noqa: E402

X_SCALE = 83.0


class HeteroBLR(Block):
    """Bayesian linear regression, polynomial mean, polynomial-log noise.

    Parameters
    ----------
    d_mean : int
        Degree of the polynomial mean (in the scaled input t = x / X_SCALE).
    d_noise : int
        Degree of the polynomial for log sd(x).  `d_noise = 0` reproduces a
        constant noise level (the baseline's assumption); `d_noise >= 1`
        lets the band fan out with x.
    x_scale : float
        Scale applied to x before taking powers (t = x / x_scale).
    """

    def __init__(self, d_mean: int = 2, d_noise: int = 1,
                 x_scale: float = X_SCALE):
        self.d_mean = d_mean
        self.d_noise = d_noise
        self.x_scale = x_scale
        # Fitted parameters, populated by call()/fit().
        self.tau = None
        self.c = None
        self.m = None
        self.S = None
        self.nll = None

    # -- feature maps --------------------------------------------------
    def _phi(self, x: np.ndarray) -> np.ndarray:
        """Polynomial design matrix for the mean, degree d_mean."""
        t = np.asarray(x, dtype=float) / self.x_scale
        return np.vander(t, self.d_mean + 1, increasing=True)

    def _psi(self, x: np.ndarray) -> np.ndarray:
        """Polynomial design matrix for log sd, degree d_noise."""
        t = np.asarray(x, dtype=float) / self.x_scale
        return np.vander(t, self.d_noise + 1, increasing=True)

    def _sd(self, x: np.ndarray, c: np.ndarray) -> np.ndarray:
        return np.exp(self._psi(x) @ c)

    # -- type-II marginal-likelihood fit --------------------------------
    def _neg_log_marginal_likelihood(self, theta: np.ndarray,
                                      Phi: np.ndarray, y: np.ndarray,
                                      x: np.ndarray) -> float:
        log_tau = theta[0]
        c = theta[1:]
        tau2 = np.exp(2.0 * log_tau)
        sd = self._sd(x, c)
        n, k = Phi.shape

        # Marginal covariance of y: Sigma = tau^2 * Phi Phi^T + diag(sd^2)
        # Use the Woodbury identity so we only invert a (k x k) matrix.
        Dinv = 1.0 / sd ** 2
        PtDinvP = (Phi * Dinv[:, None]).T @ Phi          # k x k
        A = np.eye(k) / tau2 + PtDinvP                    # k x k
        try:
            Achol = np.linalg.cholesky(A)
        except np.linalg.LinAlgError:
            return 1e12

        DinvY = Dinv * y
        PtDinvY = Phi.T @ DinvY                            # k
        z = np.linalg.solve(Achol, PtDinvY)
        quad = np.sum(y * DinvY) - z @ z

        # log|Sigma| = log|D| + log|A| + k log(tau^2)
        logdetD = np.sum(np.log(sd ** 2))
        logdetA = 2.0 * np.sum(np.log(np.diag(Achol)))
        logdetSigma = logdetD + logdetA + k * np.log(tau2)

        nll = 0.5 * (quad + logdetSigma + n * np.log(2 * np.pi))
        return float(nll)

    def fit(self, x: np.ndarray, y: np.ndarray) -> "HeteroBLR":
        x = np.asarray(x, dtype=float)
        y = np.asarray(y, dtype=float)
        Phi = self._phi(x)

        theta0 = np.zeros(1 + self.d_noise + 1)
        theta0[0] = np.log(np.std(y) if np.std(y) > 0 else 1.0)   # log tau
        theta0[1] = np.log(np.std(y) if np.std(y) > 0 else 1.0)   # c_0

        res = minimize(self._neg_log_marginal_likelihood, theta0,
                        args=(Phi, y, x), method="L-BFGS-B")
        self.opt_result = res
        if not res.success:
            print(f"  WARNING: optimisation did not converge: {res.message}")

        log_tau = res.x[0]
        c = res.x[1:]
        self.tau = float(np.exp(log_tau))
        self.c = c
        self.nll = float(res.fun)

        sd = self._sd(x, c)
        W = 1.0 / sd ** 2
        A = Phi.T @ (Phi * W[:, None]) + np.eye(Phi.shape[1]) / self.tau ** 2
        self.S = np.linalg.inv(A)
        self.m = self.S @ (Phi.T @ (W * y))
        return self

    def predict(self, x):
        x = np.atleast_1d(np.asarray(x, dtype=float))
        Phi = self._phi(x)
        mean = Phi @ self.m
        var_epistemic = np.einsum("ij,jk,ik->i", Phi, self.S, Phi)
        sd_noise = self._sd(x, self.c)
        sd_total = np.sqrt(var_epistemic + sd_noise ** 2)
        return mean, sd_total

    # -- f3dasm Block interface -----------------------------------------
    def call(self, data: ExperimentData, **kwargs) -> ExperimentData:
        input_df, output_df = data.to_pandas()
        x = input_df["x"].to_numpy(dtype=float)
        y = output_df["y"].to_numpy(dtype=float)

        self.fit(x, y)

        print("HETERO-BLR  (polynomial mean, polynomial-log noise, "
              "type-II ML)")
        print(f"  d_mean = {self.d_mean}, d_noise = {self.d_noise}")
        print(f"  optimisation success = {self.opt_result.success}, "
              f"message = {self.opt_result.message}")
        print(f"  fitted tau (mean prior std)         = {self.tau:.6f}")
        print(f"  fitted noise-log-poly coefficients c = "
              f"{np.array2string(self.c, precision=6)}")
        print(f"  posterior mean weights m            = "
              f"{np.array2string(self.m, precision=6)}")
        print(f"  negative log marginal likelihood     = {self.nll:.4f}")

        sd_low = float(self._sd(np.array([X_LOW]), self.c)[0])
        sd_high = float(self._sd(np.array([X_HIGH]), self.c)[0])
        grows = sd_high > sd_low
        print(f"  sanity check: sd(x) at x={X_LOW} -> {sd_low:.4f}, "
              f"at x={X_HIGH} -> {sd_high:.4f}  "
              f"({'GROWS with x, as expected' if grows else 'DOES NOT GROW -- check the fit'})")

        mean_pred, sd_pred = self.predict(x)
        rmse = float(np.sqrt(np.mean((y - mean_pred) ** 2)))
        print(f"  train RMSE (posterior mean)          = {rmse:.4f}")

        @datagenerator(output_names=["y_pred_hblr", "sd_hblr",
                                     "_source_modeler"])
        def predict_hblr(x: float):
            mu, sd = self.predict(np.array([x]))
            return float(mu[0]), float(sd[0]), "modeler"

        return predict_hblr.call(data.mark_all("open"), mode="sequential")


def main() -> None:
    data = ExperimentData.from_file("data")
    block = HeteroBLR(d_mean=2, d_noise=1)
    data = block.call(data)
    data.store("data")
    print("\n  wrote y_pred_hblr, sd_hblr, _source_modeler into data/ "
          "(baseline columns kept)")

    input_df, output_df = data.to_pandas()
    x = input_df["x"].to_numpy(dtype=float)
    y = output_df["y"].to_numpy(dtype=float)

    grid = np.linspace(X_LOW, X_HIGH, 400)
    mean_grid, sd_grid = block.predict(grid)

    fig, ax = plt.subplots(figsize=(7, 4.5))
    ax.scatter(x, y, s=18, color="#333333", zorder=3, label="training data")
    ax.plot(grid, mean_grid, color="#d62728", lw=2,
            label=f"HeteroBLR mean (d_mean={block.d_mean}, d_noise={block.d_noise})")
    ax.fill_between(grid, mean_grid - 2 * sd_grid, mean_grid + 2 * sd_grid,
                    color="#d62728", alpha=0.20, label="HeteroBLR $\\pm2$ sd")
    ax.plot(grid, true_mean(grid) + 2 * true_sd(grid), "k--", lw=1.4,
            label="true $\\pm2$ sd (sd = 0.5 x)")
    ax.plot(grid, true_mean(grid) - 2 * true_sd(grid), "k--", lw=1.4)
    ax.set_xlabel("speed x [m/s]")
    ax.set_ylabel("stopping distance y [m]")
    ax.set_title("HeteroBLR: polynomial mean, noise that fans out")
    ax.legend(loc="upper left", fontsize=8)
    fig.tight_layout()
    fig.savefig("figures/model.png", dpi=150)
    print("  wrote figures/model.png")


if __name__ == "__main__":
    main()
