import os
import numpy as np
import pandas as pd
from sklearn.model_selection import train_test_split
from scipy.stats import norm
from f3dasm import ExperimentData

STUDY = "/home/mbessa/Dropbox/GitHub/3dasm_course/Lectures/Lecture20/car_team_run1"
HERE = os.path.dirname(os.path.abspath(__file__))

ed = ExperimentData.from_file(os.path.join(STUDY, "data", "car_stopping_distance"))
inp, out = ed.to_pandas()
x = inp["x"].to_numpy(dtype=float)
y = out["y"].to_numpy(dtype=float)

idx = np.arange(len(x))
tr, te = train_test_split(idx, train_size=0.75, random_state=1987)
n_train, n_test = len(tr), len(te)

X = np.column_stack([x, x**2])
coef, *_ = np.linalg.lstsq(X[tr], y[tr], rcond=None)
a, b = float(coef[0]), float(coef[1])

mean_all = a * x + b * x**2
resid_tr = y[tr] - mean_all[tr]
s = float(np.std(resid_tr, ddof=0))

mu_te, y_te = mean_all[te], y[te]
coverage = float(np.mean(np.abs(y_te - mu_te) <= 1.96 * s))
nll = float(np.mean(-norm.logpdf(y_te, loc=mu_te, scale=s)))

row = pd.DataFrame([{
    "model": "M1", "a": a, "b": b, "sd_parameter": s,
    "coverage": coverage, "nll": nll, "n_train": n_train, "n_test": n_test,
}])
row.to_csv(os.path.join(HERE, "M1.csv"), index=False)
print(row.to_string(index=False))
