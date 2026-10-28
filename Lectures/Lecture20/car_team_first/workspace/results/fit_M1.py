"""Fit model M1: mean a*x + b*x**2 (no intercept), constant sd s. Writes M1.csv."""
import os
import numpy as np
import pandas as pd
from scipy.stats import norm
from sklearn.model_selection import train_test_split
from f3dasm import ExperimentData

HERE = os.path.dirname(os.path.abspath(__file__))
STUDY = os.path.abspath(os.path.join(HERE, "..", ".."))

ed = ExperimentData.from_file(os.path.join(STUDY, "data", "car_stopping_distance"))
inp, out = ed.to_pandas()
x = inp["x"].to_numpy(dtype=float)
y = out["y"].to_numpy(dtype=float)

idx = np.arange(len(x))
train_idx, test_idx = train_test_split(idx, train_size=0.75, random_state=1987)

xtr, ytr = x[train_idx], y[train_idx]
xte, yte = x[test_idx], y[test_idx]

A = np.column_stack([xtr, xtr ** 2])
(a, b), *_ = np.linalg.lstsq(A, ytr, rcond=None)

res_tr = ytr - (a * xtr + b * xtr ** 2)
s = np.std(res_tr)  # ddof=0, standard deviation of training residuals

mu_te = a * xte + b * xte ** 2
coverage = np.mean(np.abs(yte - mu_te) <= 1.96 * s)
nll = np.mean(-norm.logpdf(yte, loc=mu_te, scale=s))

row = pd.DataFrame([{
    "model": "M1",
    "a": a,
    "b": b,
    "sd_parameter": s,
    "coverage": coverage,
    "nll": nll,
    "n_train": len(train_idx),
    "n_test": len(test_idx),
}])
row.to_csv(os.path.join(HERE, "M1.csv"), index=False)
print(row.to_string(index=False))
