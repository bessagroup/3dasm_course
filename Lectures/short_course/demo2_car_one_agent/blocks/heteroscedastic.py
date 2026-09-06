"""Heteroscedastic Bayesian linear regression -- the model the baseline should
have been.

The baseline keeps a degree-2 polynomial mean (which is right) and one constant
residual standard deviation (which is wrong: the data fans out).  So this model
changes the noise from one number to a fitted function:

    mean   E[y|x]  = phi(x) . w,        phi = [1, t, ..., t^d_mean]
    noise  sd[y|x] = exp(n(x) . c),     n   = [1, u, ..., u^d_noise]

both bases rescaled to [-1, 1] for conditioning:

    t = x rescaled                    (always)
    u = log x rescaled, if link='log' -> sd is a polynomial in log x
    u = x   rescaled, if link='lin'   -> sd is exp of a polynomial in x

The exp link makes sd strictly positive by construction, so nothing has to be
clipped.  Two hyperparameters (d_mean, d_noise) and one categorical (link) are
left free on purpose -- `blocks/selection.py` chooses them; nothing here
decides them by taste.  Note the nesting:

    d_noise = 0  ->  sd is one constant       ==  the baseline's noise model
    d_noise = 1, link='log'  ->  sd = e^c0 x^c1   (a power law)

so the baseline is a *member* of this family, not a rival to it.

Given the noise, the mean weights are not point-estimated but integrated over:
with a weak Gaussian prior w ~ N(0, alpha^-1 I) the posterior is Gaussian in
closed form, and the predictive variance splits into two pieces

    var[y|x] = phi(x)^T A^-1 phi(x)  +  sd(x)^2
               \_ epistemic _/          \_ aleatoric _/

The noise weights c are chosen by maximising the log marginal likelihood (the
evidence), which integrates the mean weights out rather than fitting them.

Running this module writes into the f3dasm record `data/`, next to the
baseline's columns, using the DEFAULT hyperparameters below:

    y_pred_hblr    posterior predictive mean
    sd_hblr        posterior predictive sd (epistemic + aleatoric)
    _source_hblr   provenance stamp, 'hblr'

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

# Defaults: a degree-2 mean (the baseline's, on purpose) and a power-law noise.
# These are only defaults -- blocks/selection.py picks them from the data.
D_MEAN, D_NOISE, LINK = 2, 1, "log"
ALPHA = 1e-6        # prior precision on the mean weights: deliberately weak
LOG_2PI = float(np.log(2.0 * np.pi))
LINKS = ("log", "lin")


# ----------------------------------------------------------------------------
# Bases
# ----------------------------------------------------------------------------
def _rescale(v, lo, hi):
    """Map [lo, hi] onto [-1, 1]."""
    return (2.0 * np.asarray(v, float) - (lo + hi)) / (hi - lo)


def mean_basis(x, d_mean: int = D_MEAN) -> np.ndarray:
    """[1, t, ..., t^d_mean] with t = x rescaled to [-1, 1]."""
    t = _rescale(np.atleast_1d(x), X_LOW, X_HIGH)
    return np.vander(t, d_mean + 1, increasing=True)


def noise_basis(x, d_noise: int = D_NOISE, link: str = LINK) -> np.ndarray:
    """[1, u, ..., u^d_noise]; u is rescaled log x ('log') or rescaled x ('lin').

    At d_noise = 0 both links give the same single column of ones, i.e. one
    constant sd -- the baseline's noise model.
    """
    if link == "log":
        u = _rescale(np.log(np.atleast_1d(x)), np.log(X_LOW), np.log(X_HIGH))
    elif link == "lin":
        u = _rescale(np.atleast_1d(x), X_LOW, X_HIGH)
    else:
        raise ValueError(f"unknown link {link!r}, expected one of {LINKS}")
    return np.vander(u, d_noise + 1, increasing=True)


# ----------------------------------------------------------------------------
# The fit
# ----------------------------------------------------------------------------
def _posterior(phi: np.ndarray, y: np.ndarray, sd: np.ndarray, alpha: float):
    """Gaussian posterior over the mean weights for a *given* noise function."""
    sd2 = sd ** 2
    a_mat = alpha * np.eye(phi.shape[1]) + phi.T @ (phi / sd2[:, None])
    b_vec = phi.T @ (y / sd2)
    return a_mat, b_vec, np.linalg.solve(a_mat, b_vec)


def _neg_log_evidence(c, phi, nb, y, alpha) -> float:
    """-log p(y | c): the mean weights are integrated out, not fitted."""
    log_sd = nb @ c
    if not np.all(np.isfinite(log_sd)) or log_sd.max() > 50.0:
        return 1e12
    sd = np.exp(log_sd)
    a_mat, b_vec, m_vec = _posterior(phi, y, sd, alpha)
    sign, logdet = np.linalg.slogdet(a_mat)
    if sign <= 0:
        return 1e12
    quad = float(y @ (y / sd ** 2) - m_vec @ b_vec)
    n, d = len(y), phi.shape[1]
    return 0.5 * (n * LOG_2PI + 2.0 * log_sd.sum() + logdet
                  - d * np.log(alpha) + quad)


def fit(x, y, d_mean: int = D_MEAN, d_noise: int = D_NOISE, link: str = LINK,
        alpha: float = ALPHA, verbose: bool = False) -> dict:
    """Fit the model at fixed hyperparameters.  Returns mu(x), sd(x) and scores."""
    x, y = np.asarray(x, float), np.asarray(y, float)
    phi, nb = mean_basis(x, d_mean), noise_basis(x, d_noise, link)

    # Start from the homoscedastic answer: constant term = log(sd_ols), rest 0.
    resid_ols = y - phi @ np.linalg.lstsq(phi, y, rcond=None)[0]
    dof = max(len(y) - phi.shape[1], 1)
    c_init = np.zeros(d_noise + 1)
    c_init[0] = np.log(max(resid_ols.std(ddof=0) * np.sqrt(len(y) / dof), 1e-6))

    opt = minimize(_neg_log_evidence, c_init, args=(phi, nb, y, alpha),
                   method="Nelder-Mead",
                   options={"xatol": 1e-10, "fatol": 1e-10, "maxiter": 20000})
    c = opt.x

    sd_train = np.exp(nb @ c)
    a_mat, _, w = _posterior(phi, y, sd_train, alpha)
    a_inv = np.linalg.inv(a_mat)

    def mu_of(xq):
        return mean_basis(xq, d_mean) @ w

    def sd_aleatoric_of(xq):
        return np.exp(noise_basis(xq, d_noise, link) @ c)

    def sd_of(xq):
        ph = mean_basis(xq, d_mean)
        epistemic = np.einsum("ij,jk,ik->i", ph, a_inv, ph)
        return np.sqrt(epistemic + sd_aleatoric_of(xq) ** 2)

    model = {"d_mean": d_mean, "d_noise": d_noise, "link": link, "alpha": alpha,
             "c": c, "w": w, "a_inv": a_inv, "converged": bool(opt.success),
             "log_evidence": -float(opt.fun),
             "n_params": (d_mean + 1) + (d_noise + 1),
             "mu_of": mu_of, "sd_of": sd_of, "sd_aleatoric_of": sd_aleatoric_of}

    if verbose:
        resid = y - mu_of(x)
        ep = sd_of(x) ** 2 - sd_aleatoric_of(x) ** 2
        print(f"HETEROSCEDASTIC BLR  d_mean={d_mean}, d_noise={d_noise}, "
              f"link='{link}', alpha={alpha:g}")
        print(f"  noise weights c          = "
              f"{np.array2string(c, precision=4)}   (converged = {opt.success})")
        print(f"  posterior mean weights w = {np.array2string(w, precision=4)}")
        print(f"  log evidence             = {model['log_evidence']:.4f}   "
              f"({model['n_params']} parameters)")
        print(f"  train RMSE               = {np.sqrt(np.mean(resid ** 2)):.4f}")
        for xq in (3.0, 43.0, 83.0):
            print(f"  fitted sd(x = {xq:4.0f})       = {float(sd_of(xq)[0]):8.4f}"
                  f"   (truth {float(true_sd(xq)):.4f})")
        print(f"  epistemic share of var   = "
              f"{float(np.mean(ep / sd_of(x) ** 2)) * 100:.3f}%  (mean over "
              f"the training x)")
    return model


# ----------------------------------------------------------------------------
# The f3dasm Block
# ----------------------------------------------------------------------------
class HeteroscedasticBLR(Block):
    """Fit the model on a record and append its columns to that same record."""

    def __init__(self, d_mean: int = D_MEAN, d_noise: int = D_NOISE,
                 link: str = LINK, alpha: float = ALPHA, tag: str = "hblr"):
        self.d_mean, self.d_noise, self.link = d_mean, d_noise, link
        self.alpha, self.tag = alpha, tag

    def call(self, data: ExperimentData, **kwargs) -> ExperimentData:
        input_df, output_df = data.to_pandas()
        x = input_df["x"].to_numpy(float)
        y = output_df["y"].to_numpy(float)

        self.model = fit(x, y, self.d_mean, self.d_noise, self.link, self.alpha,
                         verbose=kwargs.get("verbose", True))
        mu_of, sd_of, tag = self.model["mu_of"], self.model["sd_of"], self.tag

        @datagenerator(output_names=[f"y_pred_{tag}", f"sd_{tag}", f"_source_{tag}"])
        def predict(x: float):
            return float(mu_of(x)[0]), float(sd_of(x)[0]), tag

        return predict.call(data.mark_all("open"), mode="sequential")


# ----------------------------------------------------------------------------
def plot_model(ax, x, y, model, sd_b=None, label="model"):
    """The shared picture: data, fitted band, the baseline's band, the truth."""
    grid = np.linspace(X_LOW, X_HIGH, 400)
    mu_g, sd_g = model["mu_of"](grid), model["sd_of"](grid)
    ax.scatter(x, y, s=18, color="#333333", zorder=3, label="training data")
    ax.fill_between(grid, mu_g - 2 * sd_g, mu_g + 2 * sd_g, color="#d62728",
                    alpha=0.20, label=f"{label} $\\pm2$ sd (fitted, fans out)")
    ax.plot(grid, mu_g, color="#d62728", lw=2, label=f"{label} mean")
    if sd_b is not None:
        ax.plot(grid, mu_g + 2 * sd_b, color="#1f77b4", lw=1.2, ls=":",
                label=f"baseline $\\pm2$ sd (constant, sd = {sd_b:.1f})")
        ax.plot(grid, mu_g - 2 * sd_b, color="#1f77b4", lw=1.2, ls=":")
    ax.plot(grid, true_mean(grid) + 2 * true_sd(grid), "k--", lw=1.4,
            label="true $\\pm2$ sd (sd = 0.5 x)")
    ax.plot(grid, true_mean(grid) - 2 * true_sd(grid), "k--", lw=1.4)
    ax.set_xlabel("speed x [m/s]")
    ax.set_ylabel("stopping distance y [m]")
    ax.legend(loc="upper left", fontsize=8)
    return ax


def main() -> None:
    data = ExperimentData.from_file("data")
    block = HeteroscedasticBLR()
    data = block.call(data)
    data.store("data")
    print("\n  wrote y_pred_hblr, sd_hblr, _source_hblr into data/")

    input_df, output_df = data.to_pandas()
    x = input_df["x"].to_numpy(float)
    y = output_df["y"].to_numpy(float)
    sd_b = float(output_df["sd_baseline"].iloc[0])

    fig, ax = plt.subplots(figsize=(7, 4.5))
    plot_model(ax, x, y, block.model, sd_b)
    ax.set_title(f"Heteroscedastic BLR: the band fans with the data "
                 f"(d_mean={D_MEAN}, d_noise={D_NOISE}, link='{LINK}')")
    fig.tight_layout()
    fig.savefig("figures/model.png", dpi=150)
    print("  wrote figures/model.png")


if __name__ == "__main__":
    main()
