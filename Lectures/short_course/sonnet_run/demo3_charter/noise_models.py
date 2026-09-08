"""Candidate mean/noise models for the car stopping-distance record.

Every candidate shares the *same* mean model: an ordinary-least-squares
quadratic in x, y_mean(x) = b0 + b1*x + b2*x^2. This is the mean model the
user explicitly asked for, and it is not itself in question here -- what is
in question is how the scatter around that mean should be described.

The three candidates below differ only in how sd(x) is modelled, fit by
maximum likelihood under a Gaussian residual assumption:

1. ConstantNoiseQuadraticModel  -- sd(x) = sigma, a single constant. This is
   the model the user asked for by name, and it is the simplest possible
   noise assumption: homoscedastic Gaussian scatter around the quadratic
   mean.

2. LinearNoiseQuadraticModel    -- sd(x) = max(a + b*x, eps). Motivated by
   the physics of stopping distance: distance = v*t_reaction + v^2/(2*mu*g).
   Variability in the driver's reaction time contributes a term to the
   scatter that grows linearly with v, so a noise sd that grows linearly
   with x (not with the mean, which is quadratic) is a physically
   defensible alternative to a flat sd.

3. ProportionalNoiseQuadraticModel -- sd(x) = k * |y_mean(x)|, i.e. constant
   coefficient of variation. Motivated by the other physical source of
   scatter: variability in the road/tyre friction coefficient mu enters the
   braking term multiplicatively, so it produces scatter proportional to the
   predicted distance itself, not proportional to x.

All three are worth fitting side by side because the raw data show residual
spread that grows markedly with x (see fit_candidates.py output): a constant
sd is the simplest hypothesis but visibly optimistic at low speeds and
pessimistic at high speeds; the two heteroscedastic forms encode two
different, physically motivated guesses about *why* the scatter grows, and
they are not nested in a way that makes one obviously preferred over the
other without evaluation. That evaluation is not this module's job.

Interface every candidate implements:
    fit(x, y)          -- fits all parameters in place, returns self
    predict(x)          -- -> (mean, sd), both arrays broadcasting with x
    name                 -- short string identifier
    params               -- dict of fitted parameters, reachable on the instance
"""
from __future__ import annotations

import numpy as np
from scipy.optimize import minimize_scalar


class _QuadraticMeanBase:
    """Shared quadratic-mean fitting logic. Not used directly."""

    name = "base"

    def __init__(self):
        self.beta = None       # (b0, b1, b2) of the quadratic mean
        self.params = {}       # fitted noise-model parameters, filled by subclass
        self.n = None
        self.loglik = None     # Gaussian log-likelihood at the fitted parameters

    def _design(self, x):
        x = np.asarray(x, dtype=float)
        return np.c_[np.ones_like(x), x, x ** 2]

    def _fit_mean(self, x, y):
        X = self._design(x)
        beta, *_ = np.linalg.lstsq(X, y, rcond=None)
        self.beta = beta
        return X @ beta

    def mean(self, x):
        X = self._design(x)
        return X @ self.beta

    def predict(self, x):
        raise NotImplementedError

    def _gaussian_loglik(self, y, mu, sd):
        sd = np.asarray(sd, dtype=float)
        return float(np.sum(-0.5 * np.log(2 * np.pi) - np.log(sd)
                             - 0.5 * ((y - mu) / sd) ** 2))


class ConstantNoiseQuadraticModel(_QuadraticMeanBase):
    """Quadratic mean, single constant noise sd (Gaussian, homoscedastic).

    This is the model the user asked for by name.
    """

    name = "quad_const_sd"

    def fit(self, x, y):
        x = np.asarray(x, dtype=float)
        y = np.asarray(y, dtype=float)
        self.n = len(x)
        mu = self._fit_mean(x, y)
        resid = y - mu
        sigma = float(np.sqrt(np.mean(resid ** 2)))  # Gaussian MLE sd
        self.params = {"sigma": sigma}
        self.loglik = self._gaussian_loglik(y, mu, sigma)
        return self

    def predict(self, x):
        mu = self.mean(x)
        sd = np.full_like(np.asarray(x, dtype=float), self.params["sigma"])
        return mu, sd


class LinearNoiseQuadraticModel(_QuadraticMeanBase):
    """Quadratic mean, sd(x) = max(a + b*x, eps), fit by Gaussian MLE.

    Noise grows linearly with speed, motivated by reaction-time variability
    (which enters the stopping distance linearly in v).
    """

    name = "quad_linear_sd"
    _eps = 1e-3

    def fit(self, x, y):
        x = np.asarray(x, dtype=float)
        y = np.asarray(y, dtype=float)
        self.n = len(x)
        mu = self._fit_mean(x, y)
        resid = y - mu

        def neg_loglik(params):
            a, b = params
            sd = np.maximum(a + b * x, self._eps)
            return -self._gaussian_loglik(y, mu, sd)

        # initialise from a crude split-sample estimate of how sd scales with x
        a0 = float(np.std(resid))
        b0 = 0.0
        from scipy.optimize import minimize
        res = minimize(neg_loglik, x0=[a0, b0], method="Nelder-Mead")
        a, b = res.x
        self.params = {"a": float(a), "b": float(b)}
        sd = np.maximum(a + b * x, self._eps)
        self.loglik = self._gaussian_loglik(y, mu, sd)
        return self

    def predict(self, x):
        mu = self.mean(x)
        sd = np.maximum(self.params["a"] + self.params["b"] * np.asarray(x, dtype=float), self._eps)
        return mu, sd


class ProportionalNoiseQuadraticModel(_QuadraticMeanBase):
    """Quadratic mean, sd(x) = k * |mean(x)| (constant coefficient of variation).

    Noise grows with the predicted distance itself, motivated by
    variability in the friction coefficient (which enters the braking term
    of the stopping distance multiplicatively).
    """

    name = "quad_proportional_sd"
    _eps = 1e-3

    def fit(self, x, y):
        x = np.asarray(x, dtype=float)
        y = np.asarray(y, dtype=float)
        self.n = len(x)
        mu = self._fit_mean(x, y)
        resid = y - mu
        mu_safe = np.where(np.abs(mu) < self._eps, self._eps, mu)
        # Closed-form Gaussian MLE for k when sd_i = k * |mu_i|:
        k = float(np.sqrt(np.mean((resid / mu_safe) ** 2)))
        self.params = {"k": k}
        sd = np.maximum(k * np.abs(mu), self._eps)
        self.loglik = self._gaussian_loglik(y, mu, sd)
        return self

    def predict(self, x):
        mu = self.mean(x)
        sd = np.maximum(self.params["k"] * np.abs(mu), self._eps)
        return mu, sd


CANDIDATES = [
    ConstantNoiseQuadraticModel,
    LinearNoiseQuadraticModel,
    ProportionalNoiseQuadraticModel,
]
