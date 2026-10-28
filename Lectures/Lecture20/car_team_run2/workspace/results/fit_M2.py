"""Fit model M2 for the car stopping-distance study (per PROBLEM_STATEMENT.md).

M2: a, b by lstsq of y/x on [1, x] (training rows).
    mean = a*x + b*x**2 ; sd = c*x, c = std(training residuals of y/x, ddof=0).
Assessment on test rows: coverage (|y-mean| <= 1.96*sd), nll (mean of -log normal density of y).
"""
import os

import numpy as np
import pandas as pd
from sklearn.model_selection import train_test_split
from f3dasm import ExperimentData

STUDY_DIR = "/home/mbessa/Dropbox/GitHub/3dasm_course/Lectures/Lecture20/car_team_run2"
DATA_PATH = os.path.join(STUDY_DIR, "data", "car_stopping_distance")
OUT_PATH = os.path.join(STUDY_DIR, "workspace", "results", "M2.csv")

data = ExperimentData.from_file(DATA_PATH)
input_df = data.to_pandas()[0]
output_df = data.to_pandas()[1]

x_all = np.asarray(input_df["x"], dtype=float).ravel()
y_all = np.asarray(output_df["y"], dtype=float).ravel()
assert len(x_all) == len(y_all) == 50

idx = np.arange(len(x_all))
train_idx, test_idx = train_test_split(idx, train_size=0.75, random_state=1987)
assert len(train_idx) == 37 and len(test_idx) == 13

x_tr, y_tr = x_all[train_idx], y_all[train_idx]
x_te, y_te = x_all[test_idx], y_all[test_idx]

# Fit a, b: y/x = a + b*x  =>  y = a*x + b*x**2
X_tr = np.column_stack([np.ones_like(x_tr), x_tr])
coef, *_ = np.linalg.lstsq(X_tr, y_tr / x_tr, rcond=None)
a, b = float(coef[0]), float(coef[1])

# Residual sd of y/x on training rows
resid_tr = y_tr / x_tr - (a + b * x_tr)
c = float(np.std(resid_tr, ddof=0))

# Test-set assessment
mean_te = a * x_te + b * x_te**2
sd_te = c * x_te
coverage = float(np.mean(np.abs(y_te - mean_te) <= 1.96 * sd_te))
log_dens = -0.5 * np.log(2 * np.pi * sd_te**2) - 0.5 * ((y_te - mean_te) / sd_te) ** 2
nll = float(np.mean(-log_dens))

row = pd.DataFrame(
    [{
        "model": "M2",
        "a": a,
        "b": b,
        "sd_parameter": c,
        "coverage": coverage,
        "nll": nll,
        "n_train": len(train_idx),
        "n_test": len(test_idx),
    }],
    columns=["model", "a", "b", "sd_parameter", "coverage", "nll", "n_train", "n_test"],
)
row.to_csv(OUT_PATH, index=False)
print(row.to_string(index=False))
print("written:", OUT_PATH)
