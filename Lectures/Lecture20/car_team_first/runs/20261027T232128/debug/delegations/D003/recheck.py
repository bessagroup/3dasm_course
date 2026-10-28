import numpy as np
from sklearn.model_selection import train_test_split
from f3dasm import ExperimentData
from scipy.stats import norm

STUDY = "/home/mbessa/Dropbox/GitHub/3dasm_course/Lectures/Lecture20/car_team_first"
ed = ExperimentData.from_file(f"{STUDY}/data/car_stopping_distance")
tables = ed.to_pandas()
print("n tables:", len(tables))
for i, t in enumerate(tables):
    print(i, list(t.columns), t.shape)
# identify by column names
tx = [t for t in tables if "x" in t.columns]
ty = [t for t in tables if "y" in t.columns]
assert len(tx) == 1 and len(ty) == 1
x = tx[0]["x"].to_numpy(float)
y = ty[0]["y"].to_numpy(float)
assert len(x) == len(y) == 50
print("x/y lengths", len(x), len(y), "x-table is y-table?", tx[0] is ty[0])

idx = np.arange(len(x))
tr, te = train_test_split(idx, train_size=0.75, random_state=1987)
print("n_train", len(tr), "n_test", len(te))

def nll_cov(mean, sd, yt):
    z = (yt - mean) / sd
    cov = np.mean(np.abs(yt - mean) <= 1.96 * sd)
    nll = np.mean(0.5 * np.log(2 * np.pi) + np.log(sd) + 0.5 * z**2)
    return cov, nll

# M1
X1 = np.column_stack([x, x**2])
a1, b1 = np.linalg.lstsq(X1[tr], y[tr], rcond=None)[0]
res1 = y[tr] - (a1 * x[tr] + b1 * x[tr]**2)
s1 = res1.std(ddof=0)
m1 = a1 * x[te] + b1 * x[te]**2
cov1, nll1 = nll_cov(m1, s1, y[te])

# M2
X2 = np.column_stack([np.ones_like(x), x])
a2, b2 = np.linalg.lstsq(X2[tr], (y / x)[tr], rcond=None)[0]
# residuals of y/x relative to fitted (a + b x)
res2 = (y / x)[tr] - (a2 + b2 * x[tr])
c2 = res2.std(ddof=0)
m2 = a2 * x[te] + b2 * x[te]**2
sd2 = c2 * x[te]
cov2, nll2 = nll_cov(m2, sd2, y[te])

print(f"M1 a={a1!r} b={b1!r} s={s1!r} cov={cov1!r} nll={nll1!r}")
print(f"M2 a={a2!r} b={b2!r} c={c2!r} cov={cov2!r} nll={nll2!r}")

import csv
def rel(p, q): return abs(p - q) / max(abs(q), 1e-300)
ref = {}
for name, f in [("M1", f"{STUDY}/workspace/results/M1.csv"), ("M2", f"{STUDY}/workspace/results/M2.csv")]:
    with open(f) as fh:
        r = csv.reader(fh)
        hdr = next(r); row = next(r)
        ref[name] = (hdr, dict(zip(hdr, row)), sum(1 for _ in r))
mine = {
 "M1": dict(a=a1, b=b1, sd_parameter=s1, coverage=cov1, nll=nll1),
 "M2": dict(a=a2, b=b2, sd_parameter=c2, coverage=cov2, nll=nll2),
}
want_hdr = ["model","a","b","sd_parameter","coverage","nll","n_train","n_test"]
for name in ["M1","M2"]:
    hdr, d, extra = ref[name]
    print(name, "schema ok:", hdr == want_hdr, "extra rows:", extra)
    print(name, "n_train csv", d["n_train"], "mine", len(tr), "| n_test csv", d["n_test"], "mine", len(te))
    for k, v in mine[name].items():
        rd = rel(v, float(d[k]))
        print(f"{name} {k}: csv={d[k]} mine={v!r} rel={rd:.2e} {'PASS' if rd < 1e-6 else 'FAIL'}")
