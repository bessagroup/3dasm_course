"""Candidate conditional-density models for the car stopping-distance record.

Every model here is a Gaussian conditional density

    p(y | x) = Normal( mu(x), sd(x) )

and they differ only in the *shape* they allow for ``sd(x)`` (and, in one
case, in whether the mean is forced through the origin).  They all live
behind one interface, so a caller can fit and evaluate any of them without
editing this file:

    model = car_noise.make('power')      # or car_noise.CANDIDATES['power']()
    model.fit(x, y)
    mu, sd = model.predict(x)            # also model.mean(x) / model.std(x)
    model.name                           # 'power'
    model.params                         # dict of fitted parameters
    model.loglik(x, y)                   # Gaussian log-likelihood

``car_ml.QuadraticConstantNoise`` is included, wrapped and *unchanged*, as
the baseline: ``QuadConstOLS`` below delegates to it, so it keeps producing
exactly the numbers it produced before.  ``car_ml.py`` itself is not touched.

Why these shapes for sd(x)
--------------------------
Stopping distance is reaction distance plus braking distance,

    y = t_r * v  +  v^2 / (2 * mu_fric * g),

so the quadratic mean is physically motivated, and so are two of the noise
shapes: a driver-to-driver spread in reaction time t_r contributes a noise
term proportional to ``x``, and a spread in braking friction contributes one
proportional to ``x^2``.  Added in quadrature that is
``sd(x) = sqrt(a^2 x^2 + c^2 x^4)`` (``two_term`` / ``phys``).  The other
shapes -- affine, power law, exponential, proportional-to-the-mean -- are
generic ways of letting the spread grow with x, kept as candidates because
the physical story is a hypothesis, not a fact.

Fitting
-------
Every candidate except the OLS baseline is fitted by **joint maximum
likelihood**: the mean coefficients and the noise parameters are estimated
together by minimising the exact negative Gaussian log-likelihood

    NLL = 0.5 * sum_i [ log(2*pi*sd(x_i)^2) + (y_i - mu(x_i))^2 / sd(x_i)^2 ],

with ``scipy.optimize.minimize`` (L-BFGS-B, numerical gradients) from
several starting points, including a two-stage start (OLS mean, then a
regression of log|residual| on the relevant regressor).  Joint MLE is used
rather than the two-stage estimate alone because under heteroscedasticity
the mean coefficients themselves should be weighted by 1/sd(x)^2; the
two-stage fit is only the initial guess.  Positivity of scale parameters is
enforced by optimising their logarithms.
"""

from __future__ import annotations

import numpy as np
from scipy.optimize import minimize

import car_ml

LOG2PI = float(np.log(2.0 * np.pi))
_EPS = 1e-12


def _as_array(x) -> np.ndarray:
    return np.atleast_1d(np.asarray(x, dtype=float))


def _pexp(v):
    """exp of a log-parameter, clipped so a wandering optimiser cannot overflow."""
    return np.exp(np.clip(v, -60.0, 60.0))


def _scalar_out(x, values: np.ndarray):
    """Return a float if the caller passed a scalar, else the array."""
    if np.ndim(x) == 0:
        return float(values[0])
    return values


# --------------------------------------------------------------------------
# the common interface
# --------------------------------------------------------------------------

class ConditionalGaussian:
    """Interface shared by every candidate.

    Subclasses must provide ``name``, ``param_names``, ``fit``, ``mean``,
    ``std``.  Everything else (``predict``, ``loglik``, ``summary``, the
    in-sample diagnostics) is inherited.
    """

    name = 'base'
    description = ''
    param_names: tuple = ()

    def __init__(self):
        self.params: dict = {}
        self.fitted = False

    # --- to be provided by subclasses -------------------------------------
    def fit(self, x, y) -> 'ConditionalGaussian':
        raise NotImplementedError

    def mean(self, x):
        raise NotImplementedError

    def std(self, x):
        raise NotImplementedError

    # --- shared ------------------------------------------------------------
    def predict(self, x):
        """(mean, sd) at x.  Scalar in -> two floats; array in -> two arrays."""
        return self.mean(x), self.std(x)

    @property
    def n_params(self) -> int:
        return len(self.param_names)

    def loglik(self, x, y) -> float:
        """Gaussian log-likelihood of (x, y) under the *fitted* model."""
        mu = _as_array(self.mean(_as_array(x)))
        sd = _as_array(self.std(_as_array(x)))
        y = _as_array(y)
        z = (y - mu) / sd
        return float(-0.5 * np.sum(LOG2PI + 2.0 * np.log(sd) + z * z))

    def standardized_residuals(self, x, y) -> np.ndarray:
        mu = _as_array(self.mean(_as_array(x)))
        sd = _as_array(self.std(_as_array(x)))
        return (_as_array(y) - mu) / sd

    def standard_errors(self, x, y) -> dict:
        """Asymptotic MLE standard errors; ``{}`` when not available.

        Overridden by ``JointMLE``.  The OLS baseline does not provide them,
        so a generic caller gets an empty dict rather than an exception.
        """
        return {}

    def summary(self) -> str:
        bits = ', '.join(f'{k} = {self.params[k]:+.6g}'
                         for k in self.param_names)
        return f'{self.name:9s} {self.description}\n          {bits}'


# --------------------------------------------------------------------------
# baseline: car_ml.QuadraticConstantNoise, wrapped, behaviour unchanged
# --------------------------------------------------------------------------

class QuadConstOLS(ConditionalGaussian):
    """The previous round's model, delegating to ``car_ml`` untouched.

    mu(x) = b0 + b1 x + b2 x^2 by ordinary least squares;
    sd(x) = s, one constant, s^2 = RSS / (n - p)  (unbiased, 47 dof).
    """

    name = 'quad_ols'
    description = 'quadratic OLS mean, sd(x) = s (constant, /(n-p))'
    param_names = ('b0', 'b1', 'b2', 's')

    def __init__(self):
        super().__init__()
        self.inner = car_ml.QuadraticConstantNoise()

    def fit(self, x, y):
        self.inner.fit(x, y)
        b0, b1, b2 = self.inner.beta
        self.params = {'b0': float(b0), 'b1': float(b1), 'b2': float(b2),
                       's': float(self.inner.sd)}
        self.beta = np.asarray(self.inner.beta, dtype=float)
        self.sd_mle = float(self.inner.sd_mle)
        self.r2 = float(self.inner.r2)
        self.rss = float(self.inner.rss)
        self.fitted = True
        return self

    def mean(self, x):
        return _scalar_out(x, _as_array(self.inner.mean(_as_array(x))))

    def std(self, x):
        return _scalar_out(x, _as_array(self.inner.std(_as_array(x))))


# --------------------------------------------------------------------------
# joint-MLE machinery
# --------------------------------------------------------------------------

class JointMLE(ConditionalGaussian):
    """Base class: everything that is fitted by joint maximum likelihood.

    A subclass supplies ``_mu_sd(theta, x)`` (unconstrained parameter vector
    -> mean and sd arrays), ``_starts(x, y)`` (list of starting vectors) and
    ``_unpack(theta)`` (-> dict of interpretable parameters).
    """

    intercept = True          # quadratic mean includes b0 unless told otherwise
    log_params: tuple = ()    # entries of param_names that are exp(theta_j)

    def _design(self, x) -> np.ndarray:
        x = _as_array(x)
        if self.intercept:
            return np.column_stack([np.ones_like(x), x, x * x])
        return np.column_stack([x, x * x])

    # --- subclass hooks ----------------------------------------------------
    def _mu_sd(self, theta, x):
        raise NotImplementedError

    def _starts(self, x, y):
        raise NotImplementedError

    def _unpack(self, theta) -> dict:
        raise NotImplementedError

    # --- the likelihood ----------------------------------------------------
    def _nll(self, theta, x, y) -> float:
        try:
            mu, sd = self._mu_sd(theta, x)
        except FloatingPointError:
            return 1e12
        if not np.all(np.isfinite(mu)) or not np.all(np.isfinite(sd)):
            return 1e12
        sd = np.maximum(sd, _EPS)
        if np.any(sd > 1e8):
            return 1e12
        z = (y - mu) / sd
        val = 0.5 * float(np.sum(LOG2PI + 2.0 * np.log(sd) + z * z))
        return val if np.isfinite(val) else 1e12

    def fit(self, x, y):
        x = _as_array(x)
        y = _as_array(y)
        best = None
        self.optim_reports = []
        for theta0 in self._starts(x, y):
            res = minimize(self._nll, np.asarray(theta0, float),
                           args=(x, y), method='L-BFGS-B',
                           options={'maxiter': 20000, 'maxfun': 40000,
                                    'ftol': 1e-14, 'gtol': 1e-10})
            # polish with Nelder-Mead, then L-BFGS-B again
            res2 = minimize(self._nll, res.x, args=(x, y), method='Nelder-Mead',
                            options={'maxiter': 20000, 'xatol': 1e-10,
                                     'fatol': 1e-12})
            res3 = minimize(self._nll, res2.x, args=(x, y), method='L-BFGS-B',
                            options={'maxiter': 20000, 'ftol': 1e-15,
                                     'gtol': 1e-12})
            cand = min([res, res2, res3], key=lambda r: r.fun)
            self.optim_reports.append((float(cand.fun), bool(cand.success)))
            if best is None or cand.fun < best.fun:
                best = cand
        self.theta = np.asarray(best.x, dtype=float)
        self.nll = float(best.fun)
        self.converged = bool(best.success)
        self.n_starts = len(self.optim_reports)
        self.start_spread = float(
            max(v for v, _ in self.optim_reports)
            - min(v for v, _ in self.optim_reports))
        self.params = self._unpack(self.theta)
        self.fitted = True
        return self

    def mean(self, x):
        mu, _ = self._mu_sd(self.theta, _as_array(x))
        return _scalar_out(x, mu)

    def std(self, x):
        _, sd = self._mu_sd(self.theta, _as_array(x))
        return _scalar_out(x, sd)

    # --- asymptotic standard errors ---------------------------------------
    def standard_errors(self, x, y) -> dict:
        """MLE standard errors from the numerical Hessian of the NLL.

        The Hessian is taken in the unconstrained ``theta`` space by central
        differences and inverted; log-parametrised entries are mapped to the
        natural scale by the delta method (d exp(t)/dt = exp(t)).  Returns
        ``{}`` if the Hessian is not positive definite -- that failure is
        reported rather than hidden.
        """
        x, y = _as_array(x), _as_array(y)
        t = self.theta
        n = t.size
        h = np.maximum(1e-4 * np.abs(t), 1e-5)
        H = np.zeros((n, n))
        for i in range(n):
            for j in range(i, n):
                tpp, tpm, tmp, tmm = t.copy(), t.copy(), t.copy(), t.copy()
                tpp[i] += h[i]; tpp[j] += h[j]
                tpm[i] += h[i]; tpm[j] -= h[j]
                tmp[i] -= h[i]; tmp[j] += h[j]
                tmm[i] -= h[i]; tmm[j] -= h[j]
                val = ((self._nll(tpp, x, y) - self._nll(tpm, x, y)
                        - self._nll(tmp, x, y) + self._nll(tmm, x, y))
                       / (4.0 * h[i] * h[j]))
                H[i, j] = H[j, i] = val
        try:
            cov = np.linalg.inv(H)
        except np.linalg.LinAlgError:
            return {}
        var = np.diag(cov)
        if np.any(var <= 0) or not np.all(np.isfinite(var)):
            return {}
        se_theta = np.sqrt(var)
        out = {}
        for k, (nm, se) in enumerate(zip(self.param_names, se_theta)):
            out[nm] = float(se * self.params[nm]) if nm in self.log_params \
                else float(se)
        return out


class _QuadMeanMixin:
    """Quadratic mean; the first 2 or 3 entries of theta are the betas."""

    def _mu(self, theta, x):
        k = 3 if self.intercept else 2
        return self._design(x) @ np.asarray(theta[:k], float), k

    def _beta_start(self, x, y):
        X = self._design(x)
        beta, *_ = np.linalg.lstsq(X, y, rcond=None)
        return beta, y - X @ beta

    def _beta_dict(self, theta) -> dict:
        if self.intercept:
            return {'b0': float(theta[0]), 'b1': float(theta[1]),
                    'b2': float(theta[2])}
        return {'b1': float(theta[0]), 'b2': float(theta[1])}


# --------------------------------------------------------------------------
# the candidates
# --------------------------------------------------------------------------

class QuadConstMLE(_QuadMeanMixin, JointMLE):
    """sd(x) = s, constant -- the baseline's shape, but fitted by joint MLE.

    Here for a like-for-like likelihood: the MLE of a constant sd is
    sqrt(RSS/n), not sqrt(RSS/(n-p)), so this and ``quad_ols`` differ only in
    that divisor.
    """

    name = 'const'
    description = 'quadratic mean, sd(x) = s  (constant)'
    param_names = ('b0', 'b1', 'b2', 's')
    log_params = ('s',)

    def _mu_sd(self, theta, x):
        mu, k = self._mu(theta, x)
        sd = np.full_like(_as_array(x), _pexp(theta[k]))
        return mu, sd

    def _starts(self, x, y):
        beta, r = self._beta_start(x, y)
        s0 = np.log(max(float(np.std(r)), 1e-3))
        return [list(beta) + [s0], list(beta) + [s0 + 1.0]]

    def _unpack(self, theta):
        d = self._beta_dict(theta)
        d['s'] = float(_pexp(theta[3]))
        return d


class QuadAffineNoise(_QuadMeanMixin, JointMLE):
    """sd(x) = s0 + s1 * x, both terms positive."""

    name = 'affine'
    description = 'quadratic mean, sd(x) = s0 + s1*x'
    param_names = ('b0', 'b1', 'b2', 's0', 's1')
    log_params = ('s0', 's1')

    def _mu_sd(self, theta, x):
        mu, k = self._mu(theta, x)
        sd = _pexp(theta[k]) + _pexp(theta[k + 1]) * _as_array(x)
        return mu, sd

    def _starts(self, x, y):
        beta, r = self._beta_start(x, y)
        s = max(float(np.std(r)), 1e-3)
        return [list(beta) + [np.log(0.1 * s), np.log(s / max(x.mean(), 1.0))],
                list(beta) + [np.log(s), np.log(1e-3)],
                list(beta) + [np.log(1e-3), np.log(0.5)]]

    def _unpack(self, theta):
        d = self._beta_dict(theta)
        d['s0'] = float(_pexp(theta[3]))
        d['s1'] = float(_pexp(theta[4]))
        return d


class QuadPowerNoise(_QuadMeanMixin, JointMLE):
    """sd(x) = s0 * x^k -- a power law, the exponent free."""

    name = 'power'
    description = 'quadratic mean, sd(x) = s0 * x^k'
    param_names = ('b0', 'b1', 'b2', 's0', 'k')
    log_params = ('s0',)

    def _mu_sd(self, theta, x):
        mu, k = self._mu(theta, x)
        xx = np.maximum(_as_array(x), _EPS)
        sd = _pexp(theta[k]) * xx ** np.clip(theta[k + 1], -20.0, 20.0)
        return mu, sd

    def _starts(self, x, y):
        beta, r = self._beta_start(x, y)
        # two-stage start: regress log|residual| on log x
        lr = np.log(np.maximum(np.abs(r), 1e-6))
        A = np.column_stack([np.ones_like(x), np.log(x)])
        c, *_ = np.linalg.lstsq(A, lr, rcond=None)
        return [list(beta) + [float(c[0]), float(c[1])],
                list(beta) + [np.log(1e-2), 1.0],
                list(beta) + [np.log(1e-3), 2.0]]

    def _unpack(self, theta):
        d = self._beta_dict(theta)
        d['s0'] = float(_pexp(theta[3]))
        d['k'] = float(theta[4])
        return d


class QuadExpNoise(_QuadMeanMixin, JointMLE):
    """sd(x) = exp(g0 + g1 * x) -- log-linear noise, always positive."""

    name = 'exp'
    description = 'quadratic mean, sd(x) = exp(g0 + g1*x)'
    param_names = ('b0', 'b1', 'b2', 'g0', 'g1')

    def _mu_sd(self, theta, x):
        mu, k = self._mu(theta, x)
        sd = _pexp(theta[k] + theta[k + 1] * _as_array(x))
        return mu, sd

    def _starts(self, x, y):
        beta, r = self._beta_start(x, y)
        lr = np.log(np.maximum(np.abs(r), 1e-6))
        A = np.column_stack([np.ones_like(x), x])
        c, *_ = np.linalg.lstsq(A, lr, rcond=None)
        return [list(beta) + [float(c[0]), float(c[1])],
                list(beta) + [np.log(max(float(np.std(r)), 1e-3)), 0.0]]

    def _unpack(self, theta):
        d = self._beta_dict(theta)
        d['g0'] = float(theta[3])
        d['g1'] = float(theta[4])
        return d


class QuadCVNoise(_QuadMeanMixin, JointMLE):
    """sd(x) = cv * mu(x) -- one noise parameter, constant relative spread.

    The multiplicative-error story: every driver's stopping distance is the
    same curve times a random factor.
    """

    name = 'cv'
    description = 'quadratic mean, sd(x) = cv * mu(x)'
    param_names = ('b0', 'b1', 'b2', 'cv')
    log_params = ('cv',)

    def _mu_sd(self, theta, x):
        mu, k = self._mu(theta, x)
        sd = _pexp(theta[k]) * np.maximum(mu, 1e-6)
        return mu, sd

    def _starts(self, x, y):
        beta, r = self._beta_start(x, y)
        mu = self._design(x) @ beta
        cv = float(np.median(np.abs(r) / np.maximum(np.abs(mu), 1e-6)))
        return [list(beta) + [np.log(max(cv, 1e-3))],
                list(beta) + [np.log(0.1)]]

    def _unpack(self, theta):
        d = self._beta_dict(theta)
        d['cv'] = float(_pexp(theta[3]))
        return d


class QuadTwoTermNoise(_QuadMeanMixin, JointMLE):
    """sd(x) = sqrt(a^2 x^2 + c^2 x^4) -- reaction-time + friction spread.

    A driver-to-driver spread ``a`` in reaction time contributes ``a*x`` to
    the stopping distance; a spread ``c`` in the 1/(2*mu*g) braking constant
    contributes ``c*x^2``.  Independent, so they add in quadrature.
    """

    name = 'two_term'
    description = 'quadratic mean, sd(x) = sqrt(a^2 x^2 + c^2 x^4)'
    param_names = ('b0', 'b1', 'b2', 'a', 'c')
    log_params = ('a', 'c')

    def _mu_sd(self, theta, x):
        mu, k = self._mu(theta, x)
        xx = _as_array(x)
        a = _pexp(theta[k])
        c = _pexp(theta[k + 1])
        sd = np.sqrt((a * xx) ** 2 + (c * xx * xx) ** 2)
        return mu, sd

    def _starts(self, x, y):
        beta, r = self._beta_start(x, y)
        s = max(float(np.std(r)), 1e-3)
        return [list(beta) + [np.log(0.1), np.log(1e-2)],
                list(beta) + [np.log(s / max(x.mean(), 1.0)), np.log(1e-4)],
                list(beta) + [np.log(1e-3), np.log(s / max(x.mean() ** 2, 1.0))]]

    def _unpack(self, theta):
        d = self._beta_dict(theta)
        d['a'] = float(_pexp(theta[3]))
        d['c'] = float(_pexp(theta[4]))
        return d


class PhysTwoTermNoise(QuadTwoTermNoise):
    """The same, with the mean forced through the origin: mu(0) = 0.

    mu(x) = b1 x + b2 x^2, sd(x) = sqrt(a^2 x^2 + c^2 x^4).  A car that sees
    the obstacle at zero speed travels zero distance, so the intercept is not
    a free parameter physically; dropping it costs one parameter.
    """

    name = 'phys'
    description = ('mean b1*x + b2*x^2 (no intercept), '
                   'sd(x) = sqrt(a^2 x^2 + c^2 x^4)')
    param_names = ('b1', 'b2', 'a', 'c')
    log_params = ('a', 'c')
    intercept = False

    def _unpack(self, theta):
        d = self._beta_dict(theta)
        d['a'] = float(_pexp(theta[2]))
        d['c'] = float(_pexp(theta[3]))
        return d


# --------------------------------------------------------------------------
# registry -- the entry point a caller uses to get at every candidate
# --------------------------------------------------------------------------

CANDIDATES = {
    cls.name: cls for cls in (
        QuadConstOLS,        # 'quad_ols'  -- car_ml baseline, unchanged
        QuadConstMLE,        # 'const'
        QuadAffineNoise,     # 'affine'
        QuadPowerNoise,      # 'power'
        QuadExpNoise,        # 'exp'
        QuadCVNoise,         # 'cv'
        QuadTwoTermNoise,    # 'two_term'
        PhysTwoTermNoise,    # 'phys'
    )
}


def make(name: str) -> ConditionalGaussian:
    """Return a fresh, unfitted candidate by name."""
    if name not in CANDIDATES:
        raise KeyError(f'unknown candidate {name!r}; '
                       f'available: {sorted(CANDIDATES)}')
    return CANDIDATES[name]()


def fit_all(x, y, names=None) -> dict:
    """Fit every candidate (or the named subset) and return {name: model}."""
    names = list(CANDIDATES) if names is None else list(names)
    return {n: make(n).fit(x, y) for n in names}


def information_criteria(model: ConditionalGaussian, x, y) -> dict:
    """In-sample log-likelihood, AIC and BIC of a fitted model.

    These are *descriptions of the fit*, not a verdict: choosing between
    candidates on held-out data is the selector's job, not this module's.
    """
    ll = model.loglik(x, y)
    k = model.n_params
    n = len(_as_array(y))
    return {'loglik': ll, 'n_params': k,
            'aic': float(2 * k - 2 * ll),
            'bic': float(k * np.log(n) - 2 * ll)}
