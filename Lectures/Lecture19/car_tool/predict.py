"""Stopping-distance model fitted from 30 Measure() calls.

mu(x) = A*x + B*x**2            (reaction-time term + braking term), metres
sd(x) = K * x**P               (heteroscedastic noise, grows with speed)

Fitted by Gaussian maximum likelihood on all 30 measurements (x in [3, 83] m/s).
"""
import numpy as np

A = 1.4976351502687222
B = 0.09854417464117382
K = 0.33553149819063793
P = 1.0512383222276787


def predict(x):
    """Return (mu, sd) of stopping distance y (m) at speed x (m/s)."""
    x = np.asarray(x, dtype=float)
    mu = A * x + B * x**2
    sd = K * x**P
    if mu.ndim == 0:
        return float(mu), float(sd)
    return mu, sd
