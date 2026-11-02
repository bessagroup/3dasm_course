import numpy as np
from scipy.optimize import minimize

# Linear-regression surrogate (ridge-regularised linear model on quadratic
# polynomial basis functions), maximised with L-BFGS-B on [0, 1]^3.
# Loop: fit -> maximise surrogate (multi-start) -> probe -> repeat.

DEGREE = 2        # polynomial degree of the basis
ALPHA = 1e-3      # ridge regularisation strength (on standardised features)
N_STARTS = 20     # L-BFGS-B restarts when maximising the surrogate


def _basis(X, degree=DEGREE):
    X = np.atleast_2d(X)
    n, d = X.shape
    cols = [np.ones(n)]
    # all monomials x^a with total degree in 1..degree
    import itertools
    for deg in range(1, degree + 1):
        for combo in itertools.combinations_with_replacement(range(d), deg):
            cols.append(np.prod(X[:, list(combo)], axis=1))
    return np.column_stack(cols)


class _Surrogate:
    def __init__(self, degree=DEGREE, alpha=ALPHA):
        self.degree = degree
        self.alpha = alpha

    def fit(self, X, y):
        Phi = _basis(X, self.degree)
        self.mu_ = Phi[:, 1:].mean(axis=0)
        self.sd_ = Phi[:, 1:].std(axis=0)
        self.sd_[self.sd_ < 1e-12] = 1.0
        self.ym_ = y.mean()
        Z = self._z(Phi)
        # ridge: do not penalise intercept (Z column 0 is the constant term)
        n_feat = Z.shape[1]
        R = self.alpha * np.eye(n_feat)
        R[0, 0] = 0.0
        self.w_ = np.linalg.solve(Z.T @ Z + R, Z.T @ (y - self.ym_))
        return self

    def _z(self, Phi):
        Z = np.empty_like(Phi)
        Z[:, 0] = 1.0
        Z[:, 1:] = (Phi[:, 1:] - self.mu_) / self.sd_
        return Z

    def predict(self, X):
        return self._z(_basis(X, self.degree)) @ self.w_ + self.ym_


def _maximise(model, X, y, rng):
    best_x, best_v = None, np.inf

    def neg(x):
        return -float(model.predict(x[None, :])[0])

    starts = [X[np.argmax(y)]] + [rng.uniform(0, 1, X.shape[1]) for _ in range(N_STARTS)]
    for s in starts:
        r = minimize(neg, s, method="L-BFGS-B", bounds=[(0.0, 1.0)] * X.shape[1])
        if r.fun < best_v:
            best_v, best_x = r.fun, np.clip(r.x, 0.0, 1.0)
    return best_x


def strategy(probe, X0, y0, budget, seed):
    rng = np.random.default_rng(seed)
    X = np.array(X0, dtype=float)
    y = np.array(y0, dtype=float)
    for _ in range(budget):
        model = _Surrogate().fit(X, y)
        x_new = _maximise(model, X, y, rng)
        # avoid re-probing an existing point: if essentially a duplicate, explore randomly
        if np.min(np.linalg.norm(X - x_new, axis=1)) < 1e-6:
            x_new = rng.uniform(0, 1, X.shape[1])
        y_new = float(probe(x_new))
        X = np.vstack([X, x_new])
        y = np.append(y, y_new)
