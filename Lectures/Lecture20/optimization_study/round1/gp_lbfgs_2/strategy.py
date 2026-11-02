import time
import warnings

import numpy as np
from scipy.optimize import minimize
from sklearn.gaussian_process import GaussianProcessRegressor
from sklearn.gaussian_process.kernels import ConstantKernel, Matern, WhiteKernel

# Settings
N_RESTARTS_HYP = 5      # restarts for GP hyperparameter fitting
N_STARTS_ACQ = 30       # L-BFGS-B starts when maximizing the GP mean
TIME_LIMIT = 100.0      # seconds, safety margin under the 2-minute limit


def _fit_gp(X, y, seed):
    d = X.shape[1]
    kernel = ConstantKernel(1.0, (1e-3, 1e3)) * Matern(
        length_scale=np.ones(d), length_scale_bounds=(1e-2, 1e2), nu=2.5
    ) + WhiteKernel(noise_level=1e-4, noise_level_bounds=(1e-8, 1e-1))
    gp = GaussianProcessRegressor(
        kernel=kernel,
        normalize_y=True,
        n_restarts_optimizer=N_RESTARTS_HYP,
        random_state=seed,
    )
    with warnings.catch_warnings():
        warnings.simplefilter("ignore")
        gp.fit(X, y)
    return gp


def _maximize_mean(gp, X, rng):
    d = X.shape[1]
    starts = rng.uniform(0.0, 1.0, size=(N_STARTS_ACQ, d))
    starts = np.vstack([starts, X[np.argsort(-gp.predict(X))[:5]]])

    def neg_mean(x):
        return -gp.predict(x.reshape(1, -1))[0]

    best_x, best_v = None, np.inf
    for x0 in starts:
        res = minimize(neg_mean, x0, method="L-BFGS-B", bounds=[(0.0, 1.0)] * d)
        if res.fun < best_v:
            best_v, best_x = res.fun, res.x
    return np.clip(best_x, 0.0, 1.0)


def strategy(probe, X0, y0, budget, seed):
    rng = np.random.default_rng(seed)
    X = np.asarray(X0, dtype=float).copy()
    y = np.asarray(y0, dtype=float).copy()
    t0 = time.time()

    for _ in range(budget):
        if time.time() - t0 > TIME_LIMIT:
            break
        try:
            gp = _fit_gp(X, y, seed)
            x_new = _maximize_mean(gp, X, rng)
            # avoid re-probing an existing point (singular GP); jitter slightly if so
            if np.min(np.linalg.norm(X - x_new, axis=1)) < 1e-6:
                x_new = np.clip(x_new + rng.normal(0, 1e-3, size=x_new.shape), 0, 1)
        except Exception:
            x_new = rng.uniform(0.0, 1.0, size=X.shape[1])
        y_new = float(probe(x_new))
        X = np.vstack([X, x_new])
        y = np.append(y, y_new)
