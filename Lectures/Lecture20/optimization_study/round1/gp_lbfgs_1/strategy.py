import warnings

import numpy as np
from scipy.optimize import minimize
from sklearn.gaussian_process import GaussianProcessRegressor
from sklearn.gaussian_process.kernels import ConstantKernel, Matern, WhiteKernel


def _fit_gp(X, y, seed):
    kernel = ConstantKernel(1.0, (1e-3, 1e3)) * Matern(
        length_scale=np.ones(X.shape[1]), length_scale_bounds=(1e-2, 1e2), nu=2.5
    ) + WhiteKernel(noise_level=1e-4, noise_level_bounds=(1e-8, 1e-1))
    gp = GaussianProcessRegressor(
        kernel=kernel,
        normalize_y=True,
        n_restarts_optimizer=10,
        random_state=seed,
    )
    with warnings.catch_warnings():
        warnings.simplefilter("ignore")
        gp.fit(X, y)
    return gp


def _maximize_mean(gp, X, y, seed, n_starts=20):
    rng = np.random.default_rng(seed)
    starts = np.vstack([X[np.argsort(y)[-5:]], rng.uniform(0, 1, size=(n_starts, X.shape[1]))])
    best_x, best_val = None, np.inf
    for x0 in starts:
        res = minimize(
            lambda x: -gp.predict(x.reshape(1, -1))[0],
            x0,
            method="L-BFGS-B",
            bounds=[(0.0, 1.0)] * X.shape[1],
        )
        if res.fun < best_val:
            best_val, best_x = res.fun, res.x
    return np.clip(best_x, 0.0, 1.0)


def strategy(probe, X0, y0, budget, seed):
    X = np.array(X0, dtype=float).copy()
    y = np.array(y0, dtype=float).copy()
    for i in range(budget):
        gp = _fit_gp(X, y, seed + i)
        x_new = _maximize_mean(gp, X, y, seed + i)
        y_new = probe(x_new)
        X = np.vstack([X, x_new])
        y = np.append(y, y_new)
