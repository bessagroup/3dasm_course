"""Fit M2 for the car stopping distance study (per PROBLEM_STATEMENT.md)."""
from pathlib import Path

import numpy as np
import pandas as pd
from f3dasm import ExperimentData
from sklearn.model_selection import train_test_split

STUDY = Path("/home/mbessa/Dropbox/GitHub/3dasm_course/Lectures/Lecture20/car_team_run1")
DATA = STUDY / "data" / "car_stopping_distance"
OUT = STUDY / "workspace" / "results"

data = ExperimentData.from_file(DATA)
inp, out = data.to_pandas()  # (input table, output table)
x_all = inp["x"].to_numpy(dtype=float)
y_all = out["y"].to_numpy(dtype=float)
assert len(x_all) == 50 and len(y_all) == 50

idx = np.arange(len(x_all))
train_idx, test_idx = train_test_split(idx, train_size=0.75, random_state=1987)
assert len(train_idx) == 37 and len(test_idx) == 13
x_tr, y_tr = x_all[train_idx], y_all[train_idx]
x_te, y_te = x_all[test_idx], y_all[test_idx]

# M2: lstsq of y/x on [1, x] over training rows
A_tr = np.column_stack([np.ones_like(x_tr), x_tr])
coef, *_ = np.linalg.lstsq(A_tr, y_tr / x_tr, rcond=None)
a, b = coef
resid = y_tr / x_tr - (a + b * x_tr)
c = np.std(resid, ddof=0)

# Prediction on test rows
mean_te = a * x_te + b * x_te**2
sd_te = c * x_te
coverage = float(np.mean(np.abs(y_te - mean_te) <= 1.96 * sd_te))
nll = float(np.mean(0.5 * np.log(2 * np.pi * sd_te**2) + 0.5 * ((y_te - mean_te) / sd_te) ** 2))

row = pd.DataFrame([{
    "model": "M2", "a": a, "b": b, "sd_parameter": c,
    "coverage": coverage, "nll": nll,
    "n_train": len(train_idx), "n_test": len(test_idx),
}])
OUT.mkdir(parents=True, exist_ok=True)
row.to_csv(OUT / "M2.csv", index=False)
print(row.to_string(index=False))
