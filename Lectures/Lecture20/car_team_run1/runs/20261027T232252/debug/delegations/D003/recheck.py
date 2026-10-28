import numpy as np, pandas as pd, csv
from sklearn.model_selection import train_test_split
from f3dasm import ExperimentData
S="/home/mbessa/Dropbox/GitHub/3dasm_course/Lectures/Lecture20/car_team_run1"
ed = ExperimentData.from_file(f"{S}/data/car_stopping_distance")
df_in, df_out = ed.to_pandas()
x = np.asarray(df_in.iloc[:,0] if df_in.shape[1]==1 else df_in["x"], float).ravel()
y = np.asarray(df_out.iloc[:,0] if df_out.shape[1]==1 else df_out["y"], float).ravel()
print("N", len(x), df_in.columns.tolist(), df_out.columns.tolist())
idx = np.arange(len(x))
tr, te = train_test_split(idx, train_size=0.75, random_state=1987)
print("n_train", len(tr), "n_test", len(te))
xtr, ytr, xte, yte = x[tr], y[tr], x[te], y[te]
def nll(yv, mu, sd): return np.mean(0.5*np.log(2*np.pi*sd**2)+0.5*((yv-mu)/sd)**2)
# M1
A = np.column_stack([xtr, xtr**2])
a1, b1 = np.linalg.lstsq(A, ytr, rcond=None)[0]
s = np.std(ytr-(a1*xtr+b1*xtr**2), ddof=0)
mu = a1*xte+b1*xte**2
cov1 = np.mean(np.abs(yte-mu) <= 1.96*s)
nll1 = nll(yte, mu, s)
# M2: y/x = p + q*x  => y = p*x + q*x^2, so a=p, b=q
A2 = np.column_stack([np.ones_like(xtr), xtr])
p, q = np.linalg.lstsq(A2, ytr/xtr, rcond=None)[0]
c = np.std(ytr/xtr - (p+q*xtr), ddof=0)
mu2 = p*xte+q*xte**2
sd2 = c*xte
cov2 = np.mean(np.abs(yte-mu2) <= 1.96*sd2)
nll2 = nll(yte, mu2, sd2)
out = {"M1":dict(a=a1,b=b1,sd=s,coverage=cov1,nll=nll1,n_train=len(tr),n_test=len(te)),
       "M2":dict(a=p,b=q,sd=c,coverage=cov2,nll=nll2,n_train=len(tr),n_test=len(te))}
for k,v in out.items(): print(k, v)
import json; json.dump({k:{kk:float(vv) for kk,vv in v.items()} for k,v in out.items()}, open('recomputed.json','w'), indent=1)
