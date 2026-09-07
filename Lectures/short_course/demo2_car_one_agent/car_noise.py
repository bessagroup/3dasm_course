"""Gaussian models for the car record whose noise is allowed to grow with x.

``car_ml.py`` fixed the noise at a single constant standard deviation, and its
residual panel showed that this is the weak assumption: the scatter is a few
metres at 3 m/s and tens of metres at 80 m/s. This module keeps the same
conditional-Gaussian shape

    p(y | x) = Normal( mu(x), sd(x) )

with mu(x) a polynomial, but makes ``sd(x)`` a *chosen function of x* instead
of a constant. Four forms are offered:

    'const'   sd(x) = s0                  the old assumption
    'prop'    sd(x) = s1 * x              spread proportional to velocity
    'affine'  sd(x) = s0 + s1 * x         a floor plus growth
    'power'   sd(x) = s0 * x**q           growth at a fitted rate q

'const' is deliberately one of the four and is fitted by exactly the same code
as the others, so a comparison between them changes the noise form and nothing
else -- not the fitter, not the degree of the mean, not the scoring.

Fitting is maximum likelihood. Least squares is not available any more: once
sd depends on x, the mean and the noise are coupled, because a point with a
small sd must be fitted more closely than a point with a large one. So for a
trial noise parameter the mean is solved in closed form by *weighted* least
squares with weights 1/sd(x)^2, and only the one or two noise parameters are
left to a numerical search (scipy Nelder-Mead over their logarithms, which
keeps sd positive without constraints). For 'const' the weights are equal, the
weighted solution collapses to ordinary least squares, and the fitted sd is the
maximum-likelihood sqrt(RSS/n) -- so the old model is recovered exactly.

Objects here
------------
``GaussianPolyModel``     the model: ``fit`` / ``mean`` / ``std`` / ``log_density``.
``leave_one_out``         refit 50 times, each time predicting the stop left out.
``kfold_log_density``     repeated K-fold, as a check that LOO is not a fluke.
``FitNoiseModel``         f3dasm ``Block``: fits on the record, writes columns.
``WriteLeaveOneOut``      f3dasm ``Block``: writes the held-out columns.
"""

from __future__ import annotations

import numpy as np
from scipy.optimize import minimize
from scipy.stats import norm

from f3dasm import Block, ExperimentData, datagenerator

DEGREE = 2                       # the mean stays quadratic throughout
WRITER = 'car_noise.FitNoiseModel'
WRITER_LOO = 'car_noise.WriteLeaveOneOut'

LOG2PI = float(np.log(2.0 * np.pi))


# --------------------------------------------------------------------------
# the noise forms
# --------------------------------------------------------------------------
# Each entry is (number of parameters, sd(theta, x), label). theta is
# unconstrained; the exponentials inside make sd positive for any theta the
# optimiser tries.

NOISE_FORMS = {
    'const':  (1, lambda t, x: np.full_like(x, np.exp(t[0])),
               'sd(x) = s0'),
    'prop':   (1, lambda t, x: np.exp(t[0]) * x,
               'sd(x) = s1*x'),
    'affine': (2, lambda t, x: np.exp(t[0]) + np.exp(t[1]) * x,
               'sd(x) = s0 + s1*x'),
    'power':  (2, lambda t, x: np.exp(t[0]) * x ** t[1],
               'sd(x) = s0*x^q'),
}


# --------------------------------------------------------------------------
# the model
# --------------------------------------------------------------------------

class GaussianPolyModel:
    """Polynomial mean, chosen noise form, both fitted by maximum likelihood."""

    def __init__(self, noise: str = 'prop', degree: int = DEGREE):
        if noise not in NOISE_FORMS:
            raise ValueError(f'unknown noise form {noise!r}; '
                             f'expected one of {sorted(NOISE_FORMS)}')
        self.noise = noise
        self.degree = degree
        self.n_noise, self._sd_of, self.noise_label = NOISE_FORMS[noise]
        self.beta = None
        self.theta = None

    # -- the pieces of the likelihood ---------------------------------------

    def design(self, x) -> np.ndarray:
        """The Vandermonde design matrix [1, x, x^2] for one or many x."""
        x = np.atleast_1d(np.asarray(x, dtype=float))
        return np.vander(x, self.degree + 1, increasing=True)

    def _sd(self, theta, x) -> np.ndarray:
        return self._sd_of(theta, np.atleast_1d(np.asarray(x, dtype=float)))

    @staticmethod
    def _weighted_ls(X, y, sd) -> np.ndarray:
        """Mean coefficients for a *given* sd: weighted least squares."""
        w = 1.0 / sd ** 2
        return np.linalg.solve(X.T @ (X * w[:, None]), X.T @ (w * y))

    def _nll(self, theta, x, y, X) -> float:
        """Negative log-likelihood with the mean profiled out."""
        sd = self._sd(theta, x)
        if not np.all(np.isfinite(sd)) or np.any(sd <= 0):
            return 1e12                      # push the optimiser back inside
        residual = y - X @ self._weighted_ls(X, y, sd)
        return 0.5 * float(np.sum(LOG2PI + 2.0 * np.log(sd)
                                  + (residual / sd) ** 2))

    def _starts(self, x, y, X) -> list:
        """Starting points for the search, from the ordinary-least-squares fit."""
        residual = y - X @ np.linalg.lstsq(X, y, rcond=None)[0]
        s = float(np.log(max(residual.std(), 1e-8)))     # log of a plain sd
        per_x = s - float(np.log(x.mean()))              # log of a plain sd/x
        return {
            'const':  [[s]],
            'prop':   [[per_x]],
            'affine': [[s - 1.0, per_x]],
            # two starts: growing like x, and flat -- so 'power' cannot be
            # trapped in whichever of the two it was started nearest to
            'power':  [[per_x, 1.0], [s, 0.0]],
        }[self.noise]

    # -- fitting -------------------------------------------------------------

    def fit(self, x, y) -> 'GaussianPolyModel':
        x = np.asarray(x, dtype=float)
        y = np.asarray(y, dtype=float)
        X = self.design(x)

        best = None
        for start in self._starts(x, y, X):
            trial = minimize(self._nll, np.array(start, dtype=float),
                             args=(x, y, X), method='Nelder-Mead',
                             options={'xatol': 1e-8, 'fatol': 1e-10,
                                      'maxiter': 4000})
            if best is None or trial.fun < best.fun:
                best = trial

        self.theta = best.x
        self.beta = self._weighted_ls(X, y, self._sd(best.x, x))
        self.nll = float(best.fun)
        self.n = int(x.size)
        self.n_params = self.degree + 1 + self.n_noise
        self.aic = 2.0 * self.nll + 2.0 * self.n_params
        self.bic = 2.0 * self.nll + self.n_params * float(np.log(self.n))
        return self

    # -- using the fitted model ----------------------------------------------

    def mean(self, x):
        mu = self.design(x) @ self.beta
        return float(mu[0]) if np.ndim(x) == 0 else mu

    def std(self, x):
        sd = self._sd(self.theta, x)
        return float(sd[0]) if np.ndim(x) == 0 else sd

    def log_density(self, x, y):
        """log p(y | x) under the fitted model -- the score used to compare."""
        return norm.logpdf(np.asarray(y, dtype=float),
                           self.design(x) @ self.beta,
                           self._sd(self.theta, x))

    def noise_parameters(self) -> str:
        """The fitted noise parameters in the units of their own form."""
        t = self.theta
        if self.noise == 'const':
            return f's0 = {np.exp(t[0]):.6g} m'
        if self.noise == 'prop':
            return f's1 = {np.exp(t[0]):.6g} s   (sd is this many seconds of travel)'
        if self.noise == 'affine':
            return f's0 = {np.exp(t[0]):.6g} m, s1 = {np.exp(t[1]):.6g} s'
        return f's0 = {np.exp(t[0]):.6g}, q = {t[1]:.6g}'

    def summary(self) -> str:
        terms = ' '.join(f'{b:+.6g}*x^{k}' if k else f'{b:+.6g}'
                         for k, b in enumerate(self.beta))
        return (
            f'noise form  {self.noise}   {self.noise_label}\n'
            f'mu(x) = {terms}\n'
            f'fitted noise: {self.noise_parameters()}\n'
            f'nll = {self.nll:.6g}   n_params = {self.n_params}   '
            f'AIC = {self.aic:.6g}   BIC = {self.bic:.6g}   n = {self.n}'
        )


# --------------------------------------------------------------------------
# scoring on stops the model did not see
# --------------------------------------------------------------------------

def leave_one_out(x, y, noise: str, degree: int = DEGREE) -> dict:
    """Refit without each stop in turn and predict that stop.

    Returns per-sample arrays -- ``mu``, ``sd`` and ``log_density`` -- in which
    every entry was produced by a model that never saw the stop it describes.
    """
    x = np.asarray(x, dtype=float)
    y = np.asarray(y, dtype=float)
    n = x.size
    mu = np.empty(n)
    sd = np.empty(n)

    for i in range(n):
        keep = np.ones(n, dtype=bool)
        keep[i] = False
        model = GaussianPolyModel(noise=noise, degree=degree).fit(x[keep], y[keep])
        mu[i] = model.mean(x[i:i + 1])[0]
        sd[i] = model.std(x[i:i + 1])[0]

    return {'mu': mu, 'sd': sd, 'log_density': norm.logpdf(y, mu, sd)}


def kfold_log_density(x, y, noise: str, degree: int = DEGREE,
                      k: int = 5, repeats: int = 20, seed: int = 0) -> dict:
    """Repeated K-fold mean held-out log density -- a check on leave-one-out.

    Leave-one-out trains on 49 stops; K-fold trains on 40. If the two agree,
    the comparison is not an artefact of how much data each fit was given.
    """
    x = np.asarray(x, dtype=float)
    y = np.asarray(y, dtype=float)
    n = x.size
    rng = np.random.default_rng(seed)
    per_repeat = []

    for _ in range(repeats):
        order = rng.permutation(n)
        total = 0.0
        for fold in range(k):
            test = order[fold::k]
            train = np.setdiff1d(np.arange(n), test)
            model = GaussianPolyModel(noise=noise, degree=degree).fit(x[train], y[train])
            total += float(np.sum(norm.logpdf(y[test], model.mean(x[test]),
                                              model.std(x[test]))))
        per_repeat.append(total / n)

    per_repeat = np.asarray(per_repeat)
    return {'mean': float(per_repeat.mean()),
            'sd': float(per_repeat.std(ddof=1)),
            'per_repeat': per_repeat}


# --------------------------------------------------------------------------
# the f3dasm blocks
# --------------------------------------------------------------------------

def _lookup(values: np.ndarray, table_x: np.ndarray, x: float) -> float:
    """Pick out the entry of ``values`` belonging to this x.

    A datagenerator is handed a sample's *inputs*, so a per-sample quantity
    that was computed outside it has to be found by x. The 50 velocities are
    distinct, so x identifies the stop; anything else is a bug and is raised
    rather than silently mismatched.
    """
    i = int(np.argmin(np.abs(table_x - x)))
    if abs(table_x[i] - x) > 1e-9:
        raise KeyError(f'x = {x!r} is not one of the record\'s velocities')
    return float(values[i])


class FitNoiseModel(Block):
    """Fit one noise form on the whole record and write its predictions back.

    Written columns, for a block named ``<name>``

    ``y_pred_<name>``   mu(x) at each sample's own x
    ``sd_<name>``       sd(x) at that x -- no longer the same number everywhere
    ``_source_<name>``  provenance: who wrote the columns beside it
    """

    def __init__(self, noise: str, name: str | None = None,
                 degree: int = DEGREE, verbose: bool = True):
        self.noise = noise
        self.name = name or noise
        self.verbose = verbose
        self.model = GaussianPolyModel(noise=noise, degree=degree)

    def arm(self, data: ExperimentData) -> None:
        input_df, output_df = data.to_pandas()
        x = input_df['x'].to_numpy(float)
        y = output_df['y'].to_numpy(float)
        self.model.fit(x, y)
        if self.verbose:
            print(self.model.summary())

    def call(self, data: ExperimentData, **kwargs) -> ExperimentData:
        if self.model.beta is None:
            self.arm(data)

        model, name = self.model, self.name

        @datagenerator(output_names=[f'y_pred_{name}', f'sd_{name}',
                                     f'_source_{name}'])
        def predict(x: float):
            return (float(model.mean(np.array([x]))[0]),
                    float(model.std(np.array([x]))[0]),
                    f'{WRITER}({model.noise})')

        data = data.mark_all('open')
        return predict.call(data, mode='sequential')


class WriteLeaveOneOut(Block):
    """Write the leave-one-out prediction of each stop into the record.

    Written columns, for a block named ``<name>``

    ``y_pred_loo_<name>``   mu(x_i) from the model fitted without stop i
    ``sd_loo_<name>``       sd(x_i) from that same model
    ``logpd_loo_<name>``    log p(y_i | x_i) under it -- the held-out score
    ``_source_loo_<name>``  provenance

    These are the columns the comparison is made from: no entry here was
    influenced by the measurement it is scored against.
    """

    def __init__(self, noise: str, name: str | None = None,
                 degree: int = DEGREE, verbose: bool = True):
        self.noise = noise
        self.name = name or noise
        self.degree = degree
        self.verbose = verbose
        self.result = None
        self.x = None

    def arm(self, data: ExperimentData) -> None:
        input_df, output_df = data.to_pandas()
        self.x = input_df['x'].to_numpy(float)
        y = output_df['y'].to_numpy(float)
        if np.unique(self.x).size != self.x.size:
            raise ValueError('velocities repeat; x cannot identify a stop')
        self.result = leave_one_out(self.x, y, self.noise, self.degree)
        if self.verbose:
            lpd = self.result['log_density']
            print(f'leave-one-out, noise={self.noise}: '
                  f'mean held-out log density {lpd.mean():.4f} '
                  f'over {lpd.size} stops')

    def call(self, data: ExperimentData, **kwargs) -> ExperimentData:
        if self.result is None:
            self.arm(data)

        table_x, result, name, noise = self.x, self.result, self.name, self.noise

        @datagenerator(output_names=[f'y_pred_loo_{name}', f'sd_loo_{name}',
                                     f'logpd_loo_{name}', f'_source_loo_{name}'])
        def write(x: float):
            return (_lookup(result['mu'], table_x, x),
                    _lookup(result['sd'], table_x, x),
                    _lookup(result['log_density'], table_x, x),
                    f'{WRITER_LOO}({noise})')

        data = data.mark_all('open')
        return write.call(data, mode='sequential')
