"""
car_noise_models.py

One shared interface for "quadratic mean + noise model" candidates fit to the
car stopping-distance record (data/).  The mean function is always the same:

    mu(x) = a * x^2 + b * x + c

fit once by ordinary least squares, exactly as requested ("a quadratic
least-squares model for the mean of y given x").  What differs between
candidates is only how the residual spread sd(x) is modelled -- that is the
"noise model" half of the pairing.

This module started with three candidates (constant, powerlaw, proportional)
and now has two more (linear, additive-proportional) added to close a gap a
prior report flagged: a pure power-law/proportional sd(x) collapses towards
zero as x -> 0, but the in-sample tercile diagnostic below shows the lowest
tercile of speeds still carries real residual scatter (std ~ 7 m over
x in [3, 28]), not none. `linear` and `additive_proportional` both keep a
noise floor at low speed instead of letting it vanish.

Every candidate implements the same interface:

    model = SomeNoiseModel()
    model.fit(x, y)                # fits mean (OLS) + noise params (MLE)
    mean, sd = model.predict(x)    # arrays, same shape as x
    model.name                     # short string identifying the candidate
    model.beta_                    # fitted quadratic mean coefficients (a,b,c)
    model.params_                  # dict of fitted noise parameters
    model.loglik_                  # in-sample Gaussian log-likelihood at the fit

No candidate here is scored against another, and none is scored on held-out
data. That comparison is deliberately left to a separate script (`selector`'s
job), not this module.
"""
from __future__ import annotations

import numpy as np
from scipy import optimize
from scipy.stats import norm

_EPS = 1e-8  # floor to keep sd, and log-args, away from zero


class QuadraticNoiseModel:
    """Shared base: quadratic OLS mean + a pluggable noise model."""

    name = "base"

    def fit(self, x: np.ndarray, y: np.ndarray) -> "QuadraticNoiseModel":
        x = np.asarray(x, dtype=float)
        y = np.asarray(y, dtype=float)
        X = np.column_stack([x ** 2, x, np.ones_like(x)])
        beta, residuals_ss, rank, sv = np.linalg.lstsq(X, y, rcond=None)
        self.beta_ = beta  # (a, b, c) for a*x^2 + b*x + c
        self.x_train_ = x
        self.y_train_ = y
        resid = y - X @ beta
        self.residuals_ = resid
        self._fit_noise(x, resid)
        self.loglik_ = self._loglik(x, y)
        return self

    def mean(self, x) -> np.ndarray:
        x = np.asarray(x, dtype=float)
        a, b, c = self.beta_
        return a * x ** 2 + b * x + c

    def sd(self, x) -> np.ndarray:
        raise NotImplementedError

    def predict(self, x):
        x = np.asarray(x, dtype=float)
        mu = self.mean(x)
        sigma = np.maximum(self.sd(x), _EPS)
        return mu, sigma

    def _fit_noise(self, x: np.ndarray, resid: np.ndarray) -> None:
        raise NotImplementedError

    def _loglik(self, x: np.ndarray, y: np.ndarray) -> float:
        mu, sigma = self.predict(x)
        return float(np.sum(norm.logpdf(y, loc=mu, scale=sigma)))

    def __repr__(self):
        return f"<{type(self).__name__} name={self.name!r} params={getattr(self, 'params_', None)}>"


class ConstantNoiseModel(QuadraticNoiseModel):
    """Homoscedastic noise: sd(x) = sigma, one number for the whole range.

    This is the textbook default for ordinary least squares: it assumes the
    scatter of stopping distances around the quadratic mean does not depend
    on how fast the car was going. It is the model requested explicitly as a
    baseline candidate.
    """

    name = "constant"

    def _fit_noise(self, x, resid):
        # Gaussian MLE for a constant sigma is the RMS residual.
        sigma = float(np.sqrt(np.mean(resid ** 2)))
        self.sigma_ = sigma
        self.params_ = {"sigma": sigma}

    def sd(self, x):
        x = np.asarray(x, dtype=float)
        return np.full_like(x, self.sigma_)


class PowerLawNoiseModel(QuadraticNoiseModel):
    """Heteroscedastic noise: sd(x) = c * x^p.

    Physical motivation: braking distance itself grows like v^2, so if there
    are unmodelled per-driver/per-road effects (reaction time, tyre grip,
    surface) their absolute effect on stopping distance need not be constant
    across speeds; a power law lets the data pick whether spread grows,
    shrinks, or stays flat (p=0 recovers the constant model) as x grows.
    """

    name = "powerlaw"

    def _fit_noise(self, x, resid):
        # Rough starting point: assume something close to quadratic growth,
        # matched in scale to the RMS residual at the mean speed.
        rms = float(np.sqrt(np.mean(resid ** 2)))
        xbar = float(np.mean(x))
        p0 = np.array([np.log(max(rms / max(xbar, 1.0) ** 2, _EPS)), 2.0])

        def neg_loglik(theta):
            log_c, p = theta
            c = np.exp(log_c)
            sigma = np.maximum(c * x ** p, _EPS)
            return -np.sum(norm.logpdf(resid, loc=0.0, scale=sigma))

        res = optimize.minimize(neg_loglik, p0, method="Nelder-Mead",
                                 options={"xatol": 1e-10, "fatol": 1e-10,
                                          "maxiter": 5000})
        log_c, p = res.x
        self.c_ = float(np.exp(log_c))
        self.p_ = float(p)
        self.optimizer_success_ = bool(res.success)
        self.optimizer_message_ = str(res.message)
        self.params_ = {"c": self.c_, "p": self.p_}

    def sd(self, x):
        x = np.asarray(x, dtype=float)
        return self.c_ * x ** self.p_


class ProportionalNoiseModel(QuadraticNoiseModel):
    """Heteroscedastic noise: sd(x) = k * |mu(x)|  (constant coefficient of
    variation, i.e. relative/multiplicative noise).

    Physical motivation: stopping distance is dominated, at the speeds in
    this record, by the braking term v^2 / (2 * mu_friction * g). If the
    scatter across drivers/road surfaces comes mainly from variability in
    the friction coefficient mu_friction (a relative, multiplicative effect),
    then the *absolute* scatter of y should scale with the predicted mean
    itself, not with x directly. This is the standard "constant relative
    error" noise model for a quantity built from a multiplicative physical
    factor.
    """

    name = "proportional"

    def _fit_noise(self, x, resid):
        mu = self.mean(x)
        mu_safe = np.where(np.abs(mu) > _EPS, mu, _EPS)
        z = resid / mu_safe  # relative residuals
        k = float(np.sqrt(np.mean(z ** 2)))  # closed-form Gaussian MLE
        self.k_ = k
        self.params_ = {"k": k}

    def sd(self, x):
        mu = self.mean(x)
        return self.k_ * np.abs(mu)


class LinearNoiseModel(QuadraticNoiseModel):
    """Heteroscedastic noise: sd(x) = |c0 + c1 * x|.

    Physical motivation: part of the scatter in stopping distance is driven
    by variability in a driver's reaction time. Reaction *distance* is
    v * t_reaction, so if t_reaction varies from driver to driver, its
    contribution to the scatter of y grows linearly in x on top of whatever
    fixed (x-independent) measurement/road-surface floor c0 there is. This is
    the simplest possible heteroscedastic model -- a straight line in x --
    and unlike the power-law/proportional candidates it does not force sd to
    collapse to (near) zero as x -> 0.
    """

    name = "linear"

    def _fit_noise(self, x, resid):
        rms = float(np.sqrt(np.mean(resid ** 2)))
        p0 = np.array([rms, 0.0])

        def neg_loglik(theta):
            c0, c1 = theta
            sigma = np.maximum(np.abs(c0 + c1 * x), _EPS)
            return -np.sum(norm.logpdf(resid, loc=0.0, scale=sigma))

        res = optimize.minimize(neg_loglik, p0, method="Nelder-Mead",
                                 options={"xatol": 1e-10, "fatol": 1e-10,
                                          "maxiter": 5000})
        c0, c1 = res.x
        self.c0_ = float(c0)
        self.c1_ = float(c1)
        self.optimizer_success_ = bool(res.success)
        self.optimizer_message_ = str(res.message)
        self.params_ = {"c0": self.c0_, "c1": self.c1_}

    def sd(self, x):
        x = np.asarray(x, dtype=float)
        return np.abs(self.c0_ + self.c1_ * x)


class AdditiveProportionalNoiseModel(QuadraticNoiseModel):
    """Heteroscedastic noise: sd(x) = sqrt(sigma0^2 + (k * mu(x))^2).

    Physical motivation: a floor term sigma0 for effects that do not scale
    with speed (measurement/road-marking precision, fixed reaction-distance
    uncertainty) added in quadrature to a proportional/constant-CV term
    k * mu(x) for effects that plausibly do scale with the braking distance
    itself (variability in the friction coefficient that enters the v^2/2*mu*g
    braking term multiplicatively). This nests both `constant` (k -> 0) and
    `proportional` (sigma0 -> 0) as special cases and is the standard
    "floor + relative error" noise model used when a purely multiplicative
    model is suspected of under-predicting scatter at the low end of x.
    """

    name = "additive_proportional"

    def _fit_noise(self, x, resid):
        mu = self.mean(x)
        rms = float(np.sqrt(np.mean(resid ** 2)))
        p0 = np.array([rms / 2.0, 0.05])

        def neg_loglik(theta):
            log_sigma0, k = theta
            sigma0 = np.exp(log_sigma0)
            sigma = np.sqrt(sigma0 ** 2 + (k * mu) ** 2)
            sigma = np.maximum(sigma, _EPS)
            return -np.sum(norm.logpdf(resid, loc=0.0, scale=sigma))

        res = optimize.minimize(neg_loglik, np.array([np.log(max(p0[0], _EPS)), p0[1]]),
                                 method="Nelder-Mead",
                                 options={"xatol": 1e-10, "fatol": 1e-10,
                                          "maxiter": 5000})
        log_sigma0, k = res.x
        self.sigma0_ = float(np.exp(log_sigma0))
        self.k_ = float(k)
        self.optimizer_success_ = bool(res.success)
        self.optimizer_message_ = str(res.message)
        self.params_ = {"sigma0": self.sigma0_, "k": self.k_}

    def sd(self, x):
        mu = self.mean(x)
        return np.sqrt(self.sigma0_ ** 2 + (self.k_ * mu) ** 2)


CANDIDATES = {
    "constant": ConstantNoiseModel,
    "linear": LinearNoiseModel,
    "powerlaw": PowerLawNoiseModel,
    "proportional": ProportionalNoiseModel,
    "additive_proportional": AdditiveProportionalNoiseModel,
}


def fit_all(x: np.ndarray, y: np.ndarray) -> dict:
    """Fit every candidate on the same (x, y) and return {name: fitted model}."""
    fitted = {}
    for name, cls in CANDIDATES.items():
        model = cls()
        model.fit(x, y)
        fitted[name] = model
    return fitted
