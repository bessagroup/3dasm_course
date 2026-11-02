import warnings

import numpy as np
from scipy.stats import norm
from scipy.optimize import minimize
from sklearn.gaussian_process import GaussianProcessRegressor
from sklearn.gaussian_process.kernels import ConstantKernel, Matern, WhiteKernel


def _fit_gp(X, y, seed):
    d = X.shape[1]
    kernel = ConstantKernel(1.0, (1e-3, 1e3)) * Matern(
        length_scale=np.full(d, 0.3), length_scale_bounds=(1e-2, 1e2), nu=2.5
    ) + WhiteKernel(noise_level=1e-4, noise_level_bounds=(1e-8, 1e-1))
    gp = GaussianProcessRegressor(
        kernel=kernel,
        normalize_y=True,
        n_restarts_optimizer=5,
        random_state=seed,
    )
    with warnings.catch_warnings():
        warnings.simplefilter("ignore")
        gp.fit(X, y)
    return gp


def _expected_improvement(gp, Xc, y_best, xi=0.01):
    mu, sigma = gp.predict(Xc, return_std=True)
    sigma = np.maximum(sigma, 1e-12)
    imp = mu - y_best - xi
    z = imp / sigma
    ei = imp * norm.cdf(z) + sigma * norm.pdf(z)
    ei[sigma <= 1e-12] = 0.0
    return ei


def _propose(gp, y_best, rng, d=3, n_cand=2000, n_starts=10):
    cand = rng.uniform(0.0, 1.0, size=(n_cand, d))
    ei = _expected_improvement(gp, cand, y_best)
    order = np.argsort(-ei)[:n_starts]
    best_x, best_v = cand[order[0]], -ei[order[0]]
    for i in order:
        res = minimize(
            lambda x: -_expected_improvement(gp, x.reshape(1, -1), y_best)[0],
            cand[i],
            bounds=[(0.0, 1.0)] * d,
            method="L-BFGS-B",
        )
        if res.fun < best_v:
            best_v, best_x = res.fun, res.x
    return np.clip(best_x, 0.0, 1.0)


def strategy(probe, X0, y0, budget, seed):
    """Sequential GP-EI Bayesian optimisation (Matern-5/2 ARD + white noise, EI xi=0.01)."""
    rng = np.random.default_rng(seed)
    X = np.array(X0, dtype=float).copy()
    y = np.array(y0, dtype=float).copy()
    for _ in range(int(budget)):
        gp = _fit_gp(X, y, seed=int(rng.integers(1e9)))
        x_new = _propose(gp, float(np.max(y)), rng)
        if np.min(np.linalg.norm(X - x_new, axis=1)) < 1e-6:
            x_new = rng.uniform(0.0, 1.0, size=X.shape[1])
        y_new = float(probe(x_new))
        X = np.vstack([X, x_new])
        y = np.append(y, y_new)
