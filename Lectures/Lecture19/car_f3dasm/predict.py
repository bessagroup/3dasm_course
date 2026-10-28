"""Degree-2 polynomial surrogate for the car stopping distance y(x).

Fit on the stored ExperimentData in ./data (inputs: speed x, outputs: y).
The hidden per-driver reaction time z is not part of the data.
"""
from pathlib import Path

import numpy as np
from f3dasm import ExperimentData
from sklearn.linear_model import LinearRegression
from sklearn.pipeline import make_pipeline
from sklearn.preprocessing import PolynomialFeatures

_DATA_DIR = Path(__file__).resolve().parent / "data"

_din, _dout = ExperimentData.from_file(_DATA_DIR).to_pandas()
_X = _din[["x"]].to_numpy(dtype=float)
_y = _dout["y"].to_numpy(dtype=float)

_model = make_pipeline(PolynomialFeatures(degree=2), LinearRegression())
_model.fit(_X, _y)


def predict(x):
    """Predict stopping distance y (m) for speed x (m/s).

    Accepts a scalar or array-like of speeds; returns a float for a scalar
    input and a 1-D numpy array otherwise.
    """
    arr = np.asarray(x, dtype=float)
    scalar = arr.ndim == 0
    y_hat = _model.predict(arr.reshape(-1, 1)).ravel()
    return float(y_hat[0]) if scalar else y_hat


if __name__ == "__main__":
    print("coefficients (1, x, x^2):", _model[-1].intercept_, _model[-1].coef_)
    print("predict(10), predict(50), predict(80):", predict([10, 50, 80]))
