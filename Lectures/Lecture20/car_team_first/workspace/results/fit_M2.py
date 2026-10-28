import sys
import numpy as np
import pandas as pd
from sklearn.model_selection import train_test_split
from f3dasm import ExperimentData

STUDY = "/home/mbessa/Dropbox/GitHub/3dasm_course/Lectures/Lecture20/car_team_first"
ed = ExperimentData.from_file(f"{STUDY}/data/car_stopping_distance")
inp = ed.to_pandas()[0]
out = ed.to_pandas()[1]
x = np.asarray(inp["x"], dtype=float).ravel()
y = np.asarray(out["y"], dtype=float).ravel()

idx = np.arange(len(x))
tr, te = train_test_split(idx, train_size=0.75, random_state=1987)
xtr, ytr, xte, yte = x[tr], y[tr], x[te], y[te]

# M2: a,b from lstsq of y/x on [1, x] (training rows)
A = np.column_stack([np.ones_like(xtr), xtr])
coef, *_ = np.linalg.lstsq(A, ytr / xtr, rcond=None)
a, b = float(coef[0]), float(coef[1])

# c = std of training residuals of y/x
resid_tr = ytr / xtr - (a + b * xtr)
c = float(np.std(resid_tr))  # population std (ddof=0), as in M1 convention

# Test assessment: mean a*x + b*x**2, sd = c*x
mu = a * xte + b * xte**2
sd = c * xte
coverage = float(np.mean(np.abs(yte - mu) <= 1.96 * sd))
nll = float(np.mean(0.5 * np.log(2 * np.pi * sd**2) + 0.5 * ((yte - mu) / sd) ** 2))

row = pd.DataFrame([{
    "model": "M2", "a": a, "b": b, "sd_parameter": c,
    "coverage": coverage, "nll": nll,
    "n_train": len(tr), "n_test": len(te),
}])
row.to_csv(f"{STUDY}/workspace/results/M2.csv", index=False)
print(row.to_string(index=False))
