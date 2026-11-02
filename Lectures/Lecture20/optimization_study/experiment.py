"""The hidden experiment of Lecture 20 (do not open it before the end of the lecture!).

The capacity of a design x = (x1, x2, x3) in [0, 1]^3 is set by its weakest failure mode: y = min(m1, m2, m3, m4)."""
import numpy as np


def failure_modes(X):
    """The four failure modes (rows) at each design (columns)."""
    X = np.atleast_2d(X); a, b, c = X[:, 0], X[:, 1], X[:, 2]
    return np.stack([1.0 + 1.1*a + 0.3*c - 0.2*a*b,                     # m1: yielding
                     0.4 + 2.2*b**1.5 + 0.6*a*c,                         # m2: global buckling
                     2.6 - 1.5*a**2 + 0.6*np.sin(2.5*b) - 0.7*c,         # m3: local buckling
                     1.2 + 1.0*c + 0.8*(1 - b)**2 - 0.4*a])              # m4: a fourth mode


def experiment(X):
    """y: the weakest failure mode."""
    return failure_modes(X).min(axis=0)
