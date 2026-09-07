"""A machine-learning module for the car stopping-distance record.

The model is a Gaussian conditional density

    p(y | x) = Normal( mu(x), sd )

with two deliberately separate assumptions:

  * the **mean** mu(x) = b0 + b1*x + b2*x^2 is quadratic in x, fitted by
    ordinary least squares;
  * the **noise** is a *single constant* standard deviation sd, the same at
    every velocity.

Both assumptions are stated here so that they can be checked against the
record later; nothing in this file adapts them to the data.

Two objects live here:

  ``QuadraticConstantNoise``
      the model itself: plain numpy, ``fit`` / ``mean`` / ``std``.

  ``FitQuadraticConstantNoise``
      an f3dasm ``Block``. ``arm`` fits the model on the record's own x and y;
      ``call`` writes the per-sample predictions back into the record as
      output columns, stamped with a ``_source_`` provenance column.
"""

from __future__ import annotations

import numpy as np

from f3dasm import Block, ExperimentData, datagenerator

DEGREE = 2          # the mean is a quadratic
NAME = 'quad'       # column suffix: y_pred_quad, sd_quad, _source_quad
WRITER = 'car_ml.FitQuadraticConstantNoise'


# --------------------------------------------------------------------------
# the model
# --------------------------------------------------------------------------

class QuadraticConstantNoise:
    """Quadratic least-squares mean, one constant noise standard deviation."""

    def __init__(self, degree: int = DEGREE):
        self.degree = degree
        self.beta = None
        self.sd = None

    def design(self, x) -> np.ndarray:
        """The Vandermonde design matrix [1, x, x^2] for one or many x."""
        x = np.atleast_1d(np.asarray(x, dtype=float))
        return np.vander(x, self.degree + 1, increasing=True)

    def fit(self, x, y) -> 'QuadraticConstantNoise':
        """Least squares for the mean, then residual spread for the noise."""
        X = self.design(x)
        y = np.asarray(y, dtype=float)
        self.beta, *_ = np.linalg.lstsq(X, y, rcond=None)

        residual = y - X @ self.beta
        self.n, self.p = X.shape
        rss = float(residual @ residual)
        # unbiased estimate: n - p degrees of freedom are left after the fit
        self.sd = float(np.sqrt(rss / (self.n - self.p)))
        self.sd_mle = float(np.sqrt(rss / self.n))
        self.rss = rss
        self.r2 = float(1.0 - rss / np.sum((y - y.mean()) ** 2))
        return self

    def mean(self, x):
        """mu(x). Scalar in, scalar out."""
        mu = self.design(x) @ self.beta
        return float(mu[0]) if np.isscalar(x) or np.ndim(x) == 0 else mu

    def std(self, x):
        """sd(x) -- constant by construction, but asked for per x anyway."""
        if np.isscalar(x) or np.ndim(x) == 0:
            return float(self.sd)
        return np.full(np.shape(x), self.sd, dtype=float)

    def summary(self) -> str:
        b0, b1, b2 = self.beta
        return (
            f'mu(x) = {b0:+.6g} {b1:+.6g}*x {b2:+.6g}*x^2\n'
            f'sd    = {self.sd:.6g}   (constant; n-p = {self.n - self.p} dof)\n'
            f'sd_mle= {self.sd_mle:.6g}   (same residuals, /n instead of /(n-p))\n'
            f'RSS   = {self.rss:.6g}   R^2 = {self.r2:.6g}   n = {self.n}'
        )


# --------------------------------------------------------------------------
# the f3dasm block
# --------------------------------------------------------------------------

class FitQuadraticConstantNoise(Block):
    """Fit the model on a record, then write its predictions into that record.

    Written columns
    ---------------
    ``y_pred_quad``   mu(x) at each sample's own x
    ``sd_quad``       sd at that x (constant here)
    ``_source_quad``  provenance: who wrote the two columns beside it

    Residuals are not stored: ``y`` and ``y_pred_quad`` are both in the
    record, so ``y - y_pred_quad`` is already there. (A datagenerator only
    ever receives domain *inputs*, so it could not see ``y`` anyway.)
    """

    def __init__(self, name: str = NAME, verbose: bool = True):
        self.name = name
        self.verbose = verbose
        self.model = QuadraticConstantNoise()

    def arm(self, data: ExperimentData) -> None:
        """One-time setup: fit the model on the record's x and y."""
        input_df, output_df = data.to_pandas()
        x = input_df['x'].to_numpy(float)
        y = output_df['y'].to_numpy(float)
        self.model.fit(x, y)
        if self.verbose:
            print('fitted quadratic mean + constant noise')
            print(self.model.summary())

    def call(self, data: ExperimentData, **kwargs) -> ExperimentData:
        if self.model.beta is None:
            self.arm(data)

        model, name = self.model, self.name

        @datagenerator(output_names=[f'y_pred_{name}', f'sd_{name}',
                                     f'_source_{name}'])
        def predict(x: float):
            return float(model.mean(x)), float(model.std(x)), WRITER

        data = data.mark_all('open')
        return predict.call(data, mode='sequential')
