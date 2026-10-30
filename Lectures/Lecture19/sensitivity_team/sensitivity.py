"""Global sensitivity of y to x1..x5 from the 64 training experiments only.

Design: 64 points of a 5-D Sobol-type sequence (f3dasm sampler, seed 0), not a
pick-freeze design, so the reviewer's Saltelli/Jansen estimator cannot be applied
directly. Route: fit a Gaussian-process surrogate (ARD Matern 5/2) on the 64 points,
check it by k-fold cross-validation, then estimate first-order (Saltelli 2010) and
total-order (Jansen 1999) Sobol indices on the surrogate with Monte Carlo sampling.
"""
import json
import warnings
from pathlib import Path

import numpy as np
from f3dasm import ExperimentData
from sklearn.gaussian_process import GaussianProcessRegressor
from sklearn.gaussian_process.kernels import ConstantKernel, Matern, WhiteKernel
from sklearn.model_selection import KFold

warnings.filterwarnings("ignore")
HERE = Path(__file__).resolve().parent
rng = np.random.default_rng(0)

data = ExperimentData.from_file(HERE / "training")
X, Y = data.to_pandas()
X = X.to_numpy(dtype=float)
y = Y.to_numpy(dtype=float).ravel()
n, d = X.shape
assert n == 64 and d == 5 and np.all(np.isfinite(y))


def make_gp():
    kernel = ConstantKernel(1.0, (1e-3, 1e3)) * Matern(
        length_scale=np.ones(d), length_scale_bounds=(1e-2, 1e2), nu=2.5
    ) + WhiteKernel(1e-4, (1e-8, 1e-1))
    return GaussianProcessRegressor(
        kernel=kernel, normalize_y=True, n_restarts_optimizer=8, random_state=0
    )


# Cross-validated check of the surrogate
pred = np.empty(n)
for tr, te in KFold(n_splits=8, shuffle=True, random_state=0).split(X):
    pred[te] = make_gp().fit(X[tr], y[tr]).predict(X[te])
cv_r2 = 1 - np.sum((y - pred) ** 2) / np.sum((y - y.mean()) ** 2)
cv_rmse = float(np.sqrt(np.mean((y - pred) ** 2)))

gp = make_gp().fit(X, y)

# Sobol indices of the surrogate (Saltelli sampling on [-1, 1]^5)
N = 200_000
A = rng.uniform(-1, 1, size=(N, d))
B = rng.uniform(-1, 1, size=(N, d))
fA = gp.predict(A)
fB = gp.predict(B)
var = np.var(np.concatenate([fA, fB]))
first, total = [], []
for i in range(d):
    ABi = A.copy()
    ABi[:, i] = B[:, i]
    fABi = gp.predict(ABi)
    first.append(float(np.mean(fB * (fABi - fA)) / var))          # Saltelli 2010
    total.append(float(0.5 * np.mean((fA - fABi) ** 2) / var))    # Jansen 1999

method = (
    "Gaussian-process surrogate (ARD Matern 5/2 + white noise, sklearn) fitted on the 64 "
    "Sobol-type experiments; first-order indices by Saltelli (2010) and total-order indices "
    "by Jansen (1999) pick-freeze estimators evaluated on the surrogate with N=200000 "
    f"Monte Carlo samples. Surrogate check: 8-fold CV R^2={cv_r2:.3f}, RMSE={cv_rmse:.3f} "
    f"(std of y = {y.std():.3f}). The reviewer's pick-freeze design needs 63 runs and could "
    "not be applied to the mandated 64-point Sobol design. Indices are only as accurate as "
    "the surrogate."
)
results = {
    "method": method,
    "first_order": [round(v, 4) for v in first],
    "total": [round(v, 4) for v in total],
    "n_experiments": n,
}
(HERE / "results.json").write_text(json.dumps(results, indent=2))
print(json.dumps(results, indent=2))
