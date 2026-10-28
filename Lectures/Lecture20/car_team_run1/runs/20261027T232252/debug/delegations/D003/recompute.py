import numpy as np
import pandas as pd
from sklearn.model_selection import train_test_split
from f3dasm import ExperimentData

STUDY = "/home/mbessa/Dropbox/GitHub/3dasm_course/Lectures/Lecture20/car_team_run1"
data = ExperimentData.from_file(f"{STUDY}/data/car_stopping_distance")
dfs = data.to_pandas()
inp, out = dfs[0], dfs[1]
df = pd.concat([inp, out], axis=1)
x_all = df["x"].to_numpy(dtype=float)
y_all = df["y"].to_numpy(dtype=float)
print("n rows:", len(df), "columns:", list(df.columns))

idx = np.arange(len(df))
tr, te = train_test_split(idx, train_size=0.75, random_state=1987)
print("n_train", len(tr), "n_test", len(te))

xtr, ytr = x_all[tr], y_all[tr]
xte, yte = x_all[te], y_all[te]

def nll_fn(y, mu, sd):
    return np.mean(0.5 * np.log(2 * np.pi * sd**2) + 0.5 * ((y - mu) / sd) ** 2)

# M1
A1 = np.column_stack([xtr, xtr**2])
(a1, b1), *_ = np.linalg.lstsq(A1, ytr, rcond=None)
res1 = ytr - (a1 * xtr + b1 * xtr**2)
s1 = np.std(res1, ddof=0)
mu1 = a1 * xte + b1 * xte**2
cov1 = np.mean(np.abs(yte - mu1) <= 1.96 * s1)
nll1 = nll_fn(yte, mu1, s1)

# M2
A2 = np.column_stack([np.ones_like(xtr), xtr])
(p2, q2), *_ = np.linalg.lstsq(A2, ytr / xtr, rcond=None)
a2, b2 = p2, q2
res2 = ytr / xtr - (p2 + q2 * xtr)
c2 = np.std(res2, ddof=0)
mu2 = a2 * xte + b2 * xte**2
sd2 = c2 * xte
cov2 = np.mean(np.abs(yte - mu2) <= 1.96 * sd2)
nll2 = nll_fn(yte, mu2, sd2)

mine = {
    "M1": dict(a=a1, b=b1, sd_parameter=s1, coverage=cov1, nll=nll1, n_train=len(tr), n_test=len(te)),
    "M2": dict(a=a2, b=b2, sd_parameter=c2, coverage=cov2, nll=nll2, n_train=len(tr), n_test=len(te)),
}
cols = ["model", "a", "b", "sd_parameter", "coverage", "nll", "n_train", "n_test"]
for m in ["M1", "M2"]:
    stored_df = pd.read_csv(f"{STUDY}/workspace/results/{m}.csv")
    print(m, "stored columns:", list(stored_df.columns), "rows:", len(stored_df))
    print("  schema match:", list(stored_df.columns) == cols, "row count ok:", len(stored_df) == 1)
    row = stored_df.iloc[0]
    for k, v in mine[m].items():
        sv = float(row[k])
        rel = abs(sv - float(v)) / max(abs(float(v)), 1e-300)
        ok = rel < 1e-6
        print(f"  {k}: stored={sv!r} recomputed={float(v)!r} rel={rel:.3e} {'PASS' if ok else 'FAIL'}")
