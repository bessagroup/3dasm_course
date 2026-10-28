import json
import sys
import numpy as np
import pandas as pd
from sklearn.model_selection import train_test_split
from f3dasm import ExperimentData

STUDY = "/home/mbessa/Dropbox/GitHub/3dasm_course/Lectures/Lecture20/car_team_run2"
data = ExperimentData.from_file(f"{STUDY}/data/car_stopping_distance")
inp = data.to_pandas()[0] if isinstance(data.to_pandas(), tuple) else None
dfs = data.to_pandas()
# to_pandas may return (input_df, output_df) or a single frame
if isinstance(dfs, tuple):
    input_df, output_df = dfs[0], dfs[1]
else:
    input_df = dfs
    output_df = dfs
x_all = np.asarray(input_df["x"], dtype=float).ravel()
y_all = np.asarray(output_df["y"], dtype=float).ravel()
print("N rows:", len(x_all), "input cols:", list(input_df.columns), "output cols:", list(output_df.columns))

idx = np.arange(len(x_all))
tr, te = train_test_split(idx, train_size=0.75, random_state=1987)
print("n_train", len(tr), "n_test", len(te))
x_tr, y_tr, x_te, y_te = x_all[tr], y_all[tr], x_all[te], y_all[te]

def nll_normal(y, mu, sd):
    return float(np.mean(0.5 * np.log(2 * np.pi * sd**2) + 0.5 * ((y - mu) / sd) ** 2))

# M1
A1 = np.column_stack([x_tr, x_tr**2])
a1, b1 = np.linalg.lstsq(A1, y_tr, rcond=None)[0]
res1 = y_tr - (a1 * x_tr + b1 * x_tr**2)
s1 = float(np.std(res1, ddof=0))
mu1 = a1 * x_te + b1 * x_te**2
cov1 = float(np.mean(np.abs(y_te - mu1) <= 1.96 * s1))
nll1 = nll_normal(y_te, mu1, s1)

# M2
A2 = np.column_stack([np.ones_like(x_tr), x_tr])
a2_, b2_ = np.linalg.lstsq(A2, y_tr / x_tr, rcond=None)[0]
# Spec: a,b fit of y/x on [1,x]; mean a*x + b*x**2
a2, b2 = a2_, b2_
res2 = y_tr / x_tr - (a2 + b2 * x_tr)
c2 = float(np.std(res2, ddof=0))
mu2 = a2 * x_te + b2 * x_te**2
sd2 = c2 * x_te
cov2 = float(np.mean(np.abs(y_te - mu2) <= 1.96 * sd2))
nll2 = nll_normal(y_te, mu2, sd2)

out = {
    "M1": dict(a=float(a1), b=float(b1), sd_parameter=s1, coverage=cov1, nll=nll1, n_train=len(tr), n_test=len(te)),
    "M2": dict(a=float(a2), b=float(b2), sd_parameter=c2, coverage=cov2, nll=nll2, n_train=len(tr), n_test=len(te)),
}
print(json.dumps(out, indent=1))
with open(sys.argv[1] if len(sys.argv) > 1 else "/dev/stdout", "w") as f:
    json.dump(out, f, indent=1)
