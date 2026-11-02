import numpy as np
from scipy.optimize import minimize


class _BudgetExhausted(Exception):
    pass


def strategy(probe, X0, y0, budget, seed):
    """L-BFGS-B directly on the experiment, forward-difference gradients.

    Start point: best experiment already run (X0/y0).
    Gradient: forward finite differences with step H (backward at the upper bound).
    Each new point costs 1 probe (f) + 3 probes (gradient), cache avoids repeats.
    Stops cleanly when the probe budget is spent.
    """
    H = 0.05
    X0 = np.asarray(X0, dtype=float)
    y0 = np.asarray(y0, dtype=float)
    lb, ub = np.zeros(3), np.ones(3)

    cache = {}
    # seed the cache with the existing experiments (free)
    for x, y in zip(X0, y0):
        cache[tuple(np.round(x, 12))] = float(y)
    state = {"used": 0}

    def f_probe(x):
        key = tuple(np.round(x, 12))
        if key in cache:
            return cache[key]
        if state["used"] >= budget:
            raise _BudgetExhausted()
        state["used"] += 1
        y = float(probe(np.array(x, dtype=float)))
        cache[key] = y
        return y

    def neg_f(x):
        return -f_probe(x)

    def neg_grad(x):
        x = np.asarray(x, dtype=float)
        f0 = f_probe(x)
        g = np.zeros(3)
        for i in range(3):
            e = np.zeros(3)
            e[i] = H if x[i] + H <= ub[i] else -H  # stay inside [0,1]
            xp = np.clip(x + e, lb, ub)
            fi = f_probe(xp)
            g[i] = (fi - f0) / (xp[i] - x[i])
        return -g

    # start from the best experiment already run
    x_start = X0[int(np.argmax(y0))].copy()

    try:
        minimize(
            neg_f,
            x_start,
            jac=neg_grad,
            method="L-BFGS-B",
            bounds=list(zip(lb, ub)),
            options={"maxfun": 10**6, "eps": H},
        )
    except _BudgetExhausted:
        pass
