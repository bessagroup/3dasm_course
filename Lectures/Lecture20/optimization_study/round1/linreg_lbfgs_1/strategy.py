import itertools

import numpy as np
from scipy.optimize import minimize

# Settings of the linear-regression surrogate + L-BFGS method
DEGREE = 2        # polynomial degree of the basis
ALPHA = 1e-3      # ridge regularization strength (on standardized features)
N_STARTS = 20     # number of L-BFGS-B starts for maximizing the surrogate
DUP_TOL = 1e-6    # points closer than this to an already-probed point count as duplicates


def _basis(X, degree=DEGREE):
    """Polynomial basis (all monomials of total degree <= degree) without the constant term."""
    X = np.atleast_2d(X)
    n, d = X.shape
    cols = []
    for p in range(1, degree + 1):
        for combo in itertools.combinations_with_replacement(range(d), p):
            cols.append(np.prod(X[:, list(combo)], axis=1))
    return np.column_stack(cols)


class _Surrogate:
    def fit(self, X, y):
        self.ym = y.mean()
        self.ys = y.std() if y.std() > 0 else 1.0
        F = _basis(X)
        self.fm = F.mean(axis=0)
        self.fs = F.std(axis=0)
        self.fs[self.fs == 0] = 1.0
        Z = (F - self.fm) / self.fs
        A = Z.T @ Z + ALPHA * len(y) * np.eye(Z.shape[1])
        self.b = np.linalg.solve(A, Z.T @ ((y - self.ym) / self.ys))
        return self

    def predict(self, X):
        Z = (_basis(X) - self.fm) / self.fs
        return self.ym + self.ys * (Z @ self.b)


def _maximize(model, X, rng):
    best_x, best_v = None, -np.inf

    def neg(x):
        return -float(model.predict(x[None, :])[0])

    starts = [X[np.argmax(model.predict(X))]]
    starts += list(rng.uniform(0, 1, size=(N_STARTS, X.shape[1])))
    for x0 in starts:
        r = minimize(neg, x0, method="L-BFGS-B", bounds=[(0, 1)] * X.shape[1])
        v = -r.fun
        if v > best_v:
            best_v, best_x = v, np.clip(r.x, 0, 1)
    return best_x


def strategy(probe, X0, y0, budget, seed):
    rng = np.random.default_rng(seed)
    X = np.array(X0, dtype=float)
    y = np.array(y0, dtype=float)
    probed = [x.copy() for x in X]

    for _ in range(budget):
        model = _Surrogate().fit(X, y)
        x_new = _maximize(model, X, rng)
        # avoid re-probing an existing point: fall back to a random point if duplicate
        if any(np.linalg.norm(x_new - p) < DUP_TOL for p in probed):
            x_new = rng.uniform(0, 1, size=X.shape[1])
        y_new = float(probe(x_new))
        X = np.vstack([X, x_new])
        y = np.append(y, y_new)
        probed.append(x_new.copy())
