import numpy as np
import pandas as pd
from pathlib import Path
from sklearn.model_selection import train_test_split
from scipy.stats import norm
from f3dasm import ExperimentData

STUDY = Path("/home/mbessa/Dropbox/GitHub/3dasm_course/Lectures/Lecture20/car_team_run2")
OUT = STUDY / "workspace" / "results"

ed = ExperimentData.from_file(STUDY / "data" / "car_stopping_distance")
df_in, df_out = ed.to_pandas()
x_all = df_in["x"].to_numpy(dtype=float)
y_all = df_out["y"].to_numpy(dtype=float)
assert len(x_all) == len(y_all) == 50, (len(x_all), len(y_all))

idx = np.arange(len(x_all))
idx_train, idx_test = train_test_split(idx, train_size=0.75, random_state=1987)
assert len(idx_train) == 37 and len(idx_test) == 13

x_tr, y_tr = x_all[idx_train], y_all[idx_train]
x_te, y_te = x_all[idx_test], y_all[idx_test]

A = np.column_stack([x_tr, x_tr**2])
(a, b), *_ = np.linalg.lstsq(A, y_tr, rcond=None)
resid = y_tr - (a * x_tr + b * x_tr**2)
s = np.std(resid, ddof=0)

mu_te = a * x_te + b * x_te**2
coverage = float(np.mean(np.abs(y_te - mu_te) <= 1.96 * s))
nll = float(np.mean(-norm.logpdf(y_te, loc=mu_te, scale=s)))

row = pd.DataFrame([{
    "model": "M1", "a": a, "b": b, "sd_parameter": s,
    "coverage": coverage, "nll": nll,
    "n_train": len(idx_train), "n_test": len(idx_test),
}])
OUT.mkdir(parents=True, exist_ok=True)
row.to_csv(OUT / "M1.csv", index=False)
print(row.to_string(index=False))
