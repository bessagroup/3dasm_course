import numpy as np
from scipy.optimize import minimize


def strategy(probe, X0, y0, budget, seed):
    """L-BFGS-B on the experiment (maximise y), finite-difference gradients.

    Starts from the best LHS points (ranked by y0). Each gradient costs up to
    3 probes (forward difference, backward at the upper bound). Probes are
    cached so no point is ever run twice. Restarts from the next-best LHS
    point if a run converges with budget left.
    """
    X0 = np.asarray(X0, dtype=float)
    y0 = np.asarray(y0, dtype=float)
    eps = 0.02  # finite-difference step

    cache = {}
    used = [0]
    best = {"y": -np.inf, "x": None}

    def key(x):
        return tuple(np.round(x, 12))

    def record(x, y):
        cache[key(x)] = y
        if y > best["y"]:
            best["y"], best["x"] = y, x.copy()

    for x, y in zip(X0, y0):
        record(x, float(y))

    class _OutOfBudget(Exception):
        pass

    def f_probe(x):
        x = np.clip(x, 0.0, 1.0)
        k = key(x)
        if k in cache:
            return cache[k]
        if used[0] >= budget:
            raise _OutOfBudget
        y = float(probe(x))
        used[0] += 1
        record(x, y)
        return y

    def fg(x):
        x = np.clip(x, 0.0, 1.0)
        f = f_probe(x)
        g = np.zeros(3)
        for i in range(3):
            h = eps if x[i] + eps <= 1.0 else -eps
            xp = x.copy()
            xp[i] += h
            g[i] = (f_probe(xp) - f) / h
        return -f, -g

    order = np.argsort(-y0)
    starts = [X0[i] for i in order]
    for s in starts:
        if used[0] >= budget:
            break
        try:
            minimize(fg, np.asarray(s, dtype=float), jac=True, method="L-BFGS-B",
                     bounds=[(0.0, 1.0)] * 3, options={"maxiter": 100})
        except _OutOfBudget:
            break
        # converged with budget left: restart from next-best LHS point
