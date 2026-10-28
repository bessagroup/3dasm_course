import sys
import numpy as np
from scipy.stats import norm
from sklearn.model_selection import train_test_split
from f3dasm import ExperimentData
study = "/home/mbessa/Dropbox/GitHub/3dasm_course/Lectures/Lecture20/car_team_run2"
ed = ExperimentData.from_file(f"{study}/data/car_stopping_distance")
df = ed.to_pandas()
inp, out = df[0], df[1]
x = np.asarray(inp["x"] if "x" in inp.columns else inp.iloc[:, 0], dtype=float)
y = np.asarray(out["y"] if "y" in out.columns else out.iloc[:, 0], dtype=float)
print("n rows", len(x), "input cols", list(inp.columns), "output cols", list(out.columns))
idx = np.arange(len(x))
tr, te = train_test_split(idx, train_size=0.75, random_state=1987)
print("n_train", len(tr), "n_test", len(te))
def nll_cov(mean, sd, yt):
    cov = np.mean(np.abs(yt - mean) <= 1.96 * sd)
    nll = np.mean(-norm.logpdf(yt, loc=mean, scale=sd))
    return cov, nll
xtr, ytr, xte, yte = x[tr], y[tr], x[te], y[te]
# M1
A = np.column_stack([xtr, xtr**2])
a1, b1 = np.linalg.lstsq(A, ytr, rcond=None)[0]
s1 = np.std(ytr - (a1*xtr + b1*xtr**2))
m1 = a1*xte + b1*xte**2
c1, n1 = nll_cov(m1, s1, yte)
# M2
A2 = np.column_stack([np.ones_like(xtr), xtr])
a2, b2 = np.linalg.lstsq(A2, ytr/xtr, rcond=None)[0]
c2 = np.std(ytr/xtr - (a2 + b2*xtr))
m2 = a2*xte + b2*xte**2
sd2 = c2*xte
c2c, n2 = nll_cov(m2, sd2, yte)
r = {"M1": (a1, b1, s1, c1, n1, len(tr), len(te)), "M2": (a2, b2, c2, c2c, n2, len(tr), len(te))}
import csv
for k, v in r.items():
    print(k, v)
for name in ["M1", "M2"]:
    path = f"{study}/workspace/results/{name}.csv"
    with open(path) as f:
        rows = list(csv.DictReader(f))
    print(name, "csv header", list(rows[0].keys()) if rows else None, "rows", len(rows))
    row = rows[0]
    vals = r[name]
    keys = ["a", "b", "sd_parameter", "coverage", "nll", "n_train", "n_test"]
    for kk, v in zip(keys, vals):
        cv = float(row[kk]); rd = abs(cv - float(v)) / max(abs(float(v)), 1e-300)
        print(f"  {name} {kk}: csv={cv!r} mine={float(v)!r} reldiff={rd:.3e} {'PASS' if rd<=1e-6 else 'FAIL'}")
