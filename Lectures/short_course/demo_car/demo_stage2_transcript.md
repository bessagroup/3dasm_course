# Stage 2 transcript — the same agent grades its own homework

**Command**

```bash
PATH="$ENV/bin:$PATH" claude --resume <stage-1 session id> -p "<prompt>" --allowedTools "Read Glob Grep Write Edit Bash Agent TodoWrite NotebookEdit" --output-format stream-json --verbose
```

**Prompt typed**

> Now select the model's hyperparameters properly and show me the record.

**Wall time:** 107 s

---

*session 86062f5b-ff20-4d9f-8d6b-022efda6eebd; agents available: claude, Explore, general-purpose, modeler, Plan, selector, statusline-setup*

**assistant**

This is model selection, so I'll do it the Lecture 18 way: create a held-out record that the model has never seen, enumerate the candidate grid as an f3dasm ExperimentData, score every candidate on the held-out data, then refit the winner on the training record and write it into `data/`. I will not edit the model class.

**tool: Bash**

```
ls; python make_data.py --test --seed 456
```

*result (Bash)*

```
baseline.py
blocks
CLAUDE.md
data
demo_stage1_transcript.md
figures
make_data.py
pipeline.ipynb
__pycache__
README.md
reset.sh
templates
data_test/: n = 200, sampler = random_sampler (seed 456), noise seed 457
  x in [3.73, 82.99], y in [7.12, 793.68], mean(y) = 281.252
  wrote data_test/experiment_data/{domain.json,input.csv,output.csv,jobs.csv}
```

**tool: Write**

```
/home/mbessa/Dropbox/GitHub/3dasm_course/Lectures/short_course/demo_car/select_model.py  (5760 chars)

"""Model selection for HeteroBLR, the Lecture 18 way.

The model (blocks/hetero_blr.py) has two hyperparameters: the degree of the
polynomial mean, d_mean, and the degree of the log-noise polynomial, d_noise.
This script chooses them on data the model has never seen.

    1. Held-out record:  data_test/   (python make_data.py --test --seed 456)
       -- created by the selector, never read by the model's fit.
    2. Candidate grid:   d_mean in {1,2,3,4} x d_noise in {0,1,2}, enumerated
       as an f3dasm ExperimentData so the whole study is itself a record.
    3. Score:            each candidate is fit on data/ only and scored on
       data_test/ by
           log_pred_density   mean log N(y_test | mu(x_test), sd(x_test)^2)
                              -- the proper score for a probabilistic model,
                                 it rewards the *noise* model as well as the mean
           mse                mean (y_test - mu(x_test))^2
                              -- the mean alone, for reference
       plus log_marginal (training evidence), for reference only.
    4. Winner:           argmax of held-out log_pred_density.  Refit on data/
       and written into the record as
           y_pred_selected, sd_selected, _source_selected = 'selected:...'
       The study is stored as study_selection/.

Run:  python select_model.py           (from demo_car/)
"""

from __future__ import annotations

import sys

import numpy as np
from f3dasm import ExperimentData, datagenerator
from f3dasm.design import Domain

sys.path.insert(0, "blocks")
from hetero_blr import HeteroBLR  # noqa: E402  (the model, used, not edited)

D_MEAN = (1, 2, 3, 4)
D_NOISE = (0, 1, 2)


def log_pred_density(y, mu, sd):
    return -0.5 * np.log(2.0 * np.pi * sd ** 2) - 0.5 * ((y - mu) / sd) ** 2


def main() -> None:
    train = ExperimentData.from_file("data")
    test = ExperimentData.from_file("data_test")
    xi, yo = train.to_pandas()
    x_tr, y_tr = xi["x"].to_numpy(float), yo["y"].to_numpy(float)
    xi, yo = test.to_pandas()
    x_te, y_te = xi["x"].to_numpy(float), yo["y"].to_numpy(float)
    print(f"train: data/ n={len(x_tr)}    held-out: data_test/ n={len(x_te)}")

    # --- the candidate grid as a record ------------------------------------
    domain = Domain()
    domain.add_int("d_mean", low=min(D_MEAN), high=max(D_MEAN))
    domain.add_int("d_noise", low=min(D_NOISE), high=max(D_NOISE))
    rows = [{"d_mean": a, "d_noise": b} for a in D_MEAN for b in D_NOISE]
    study = ExperimentData(domain=domain, input_data=rows)

    @datagenerator(output_names=["log_pred_density", "mse", "rmse",
                                 "log_marginal", "converged", "_source_selected"])
    def score(d_mean: int, d_noise: int):
        model = HeteroBLR(d_mean=int(d_mean), d_noise=int(d_noise)).fit(x_tr, y_tr)
        mu, sd = model.predict(x_te)
        lpd = float(np.mean(log_pred_density(y_te, mu, sd)))
        mse = float(np.mean((y_te - mu) ** 2))
        return (lpd, mse, float(np.sqrt(mse)), float(model.log_marginal),
                str(bool(model.opt_result.success)), "selected")

    study = score.call(study, mode="sequential")
    study.store("study_selection")
    si, so = study.to_pandas()
    table = si.join(so)
    print("\nSTUDY  (fit on data/, scored on data_test/)")
    print(table.to_string(float_format=lambda v: f"{v:.4f}"))

    # --- pick the winner ---------------------------------------------------
    best = table["log_pred_density"].idxmax()
    dm, dn = int(table.loc[best, "d_mean"]), int(table.loc[best, "d_noise"])
    best_mse = table["mse"].idxmin()
    print(f"\nWINNER by held-out log predictive density: "
          f"d_mean={dm}, d_noise={dn}  (row {best})")
    print(f"  held-out log_pred_density = {table.loc[best, 'log_pred_density']:.4f}")
    print(f"  held-out rmse             = {table.loc[best, 'rmse']:.4f}")
    print(f"  (row with lowest held-out mse: d_mean="
          f"{int(table.loc[best_mse, 'd_mean'])}, d_noise="
          f"{int(table.loc[best_mse, 'd_noise'])}, mse={table.loc[best_mse, 'mse']:.4f})")
    row1 = table[(table.d_mean == 2) & (table.d_noise == 1)].iloc[0]
    row0 = table[(table.d_mean == 2) & (table.d_noise == 0)].iloc[0]
    print(f"  for reference, stage-1 model d_mean=2,d_noise=1: "
          f"lpd={row1.log_pred_density:.4f}, rmse={row1.rmse:.4f}")
    print(f"  for reference, baseline-like d_mean=2,d_noise=0: "
          f"lpd={row0.log_pred_density:.4f}, rmse={row0.rmse:.4f}")

    # --- refit the winner on data/ and write it into the record -----------
    winner = HeteroBLR(d_mean=dm, d_noise=dn).fit(x_tr, y_tr)
    np.set_printoptions(precision=4, suppress=True)
    print(f"\nREFIT winner on data/: d_mean={dm}, d_noise={dn}")
    print(f"  converged                = {winner.opt_result.success}")
    print(f"  tau                      = {winner.tau:.4f}")
    print(f"  mean coefs (in t=x/83)   = {winner.m}")
    print(f"  log-noise coefs c (in t) = {winner.c}")
    for xv in (3.0, 43.0, 83.0):
        print(f"  fitted sd(x={xv:g}) = {winner.noise_sd([xv])[0]:.4f}"
              f"   (truth 0.5x = {0.5 * xv:.2f})")

    tag = f"selected:d_mean={dm},d_noise={dn}"

    @datagenerator(output_names=["y_pred_selected", "sd_selected",
                                 "_source_selected"])
    def predict_selected(x: float):
        mu, sd = winner.predict(np.array([x]))
        return float(mu[0]), float(sd[0]), tag

    train = train.mark_all("open")
    train = predict_selected.call(train, mode="sequential")
    train.store("data")
    print(f"\n  wrote y_pred_selected, sd_selected, _source_selected='{tag}' into data/")
    print("  wrote study_selection/ (the 12-row study record)")


if __name__ == "__main__":
    main()
```

*result (Write)*

```
File created successfully at: /home/mbessa/Dropbox/GitHub/3dasm_course/Lectures/short_course/demo_car/select_model.py (file state is current in your context — no need to Read it back)
```

**tool: Bash**

```
python select_model.py 2>&1 | grep -v "^\["
```

*result (Bash)*

```
train: data/ n=60    held-out: data_test/ n=200

STUDY  (fit on data/, scored on data_test/)
    d_mean  d_noise  log_pred_density       mse    rmse  log_marginal converged _source_selected
0        1        0           -5.2978 2219.8477 47.1153     -333.7103      True         selected
1        1        1           -5.2376 8448.0328 91.9132     -329.2543      True         selected
2        1        2           -5.0479 3537.2750 59.4750     -315.2971      True         selected
3        2        0           -4.5764  505.5947 22.4854     -294.3791      True         selected
4        2        1           -4.2604  510.3446 22.5908     -273.8905      True         selected
5        2        2           -4.2395  508.7490 22.5555     -273.1568      True         selected
6        3        0           -4.5965  533.0693 23.0883     -294.7227      True         selected
7        3        1           -4.2744  551.2915 23.4796     -274.7327      True         selected
8        3        2           -4.2553  553.2209 23.5206     -273.9556      True         selected
9        4        0           -4.5922  525.4983 22.9237     -295.3347      True         selected
10       4        1           -4.2689  527.2607 22.9622     -275.2512      True         selected
11       4        2           -6.4179  854.2188 29.2270     -389.4079      True         selected

WINNER by held-out log predictive density: d_mean=2, d_noise=2  (row 5)
  held-out log_pred_density = -4.2395
  held-out rmse             = 22.5555
  (row with lowest held-out mse: d_mean=2, d_noise=0, mse=505.5947)
  for reference, stage-1 model d_mean=2,d_noise=1: lpd=-4.2604, rmse=22.5908
  for reference, baseline-like d_mean=2,d_noise=0: lpd=-4.5764, rmse=22.4854

REFIT winner on data/: d_mean=2, d_noise=2
  converged                = True
  tau                      = 406.2562
  mean coefs (in t=x/83)   = [ -0.3263 118.5092 692.3022]
  log-noise coefs c (in t) = [ 1.2831  4.3337 -1.7138]
  fitted sd(x=3) = 4.2102   (truth 0.5x = 1.50)
  fitted sd(x=43) = 21.5052   (truth 0.5x = 21.50)
  fitted sd(x=83) = 49.5517   (truth 0.5x = 41.50)

  wrote y_pred_selected, sd_selected, _source_selected='selected:d_mean=2,d_noise=2' into data/
  wrote study_selection/ (the 12-row study record)
```

**assistant**

All 12 fits converged and the winner is degree 2 for the mean with a degree-2 log-noise polynomial. Now I'll append a selection cell to the notebook, execute it end to end, and print the record.

**tool: Bash**

```
python - <<'EOF'
import nbformat
nb = nbformat.read('pipeline.ipynb', 4)
idx = next(i for i, c in enumerate(nb.cells)
           if c.cell_type == 'markdown' and 'What the record remembers' in c.source)

md = nbformat.v4.new_markdown_cell(
"## Selected model (written by `select_model.py`)\n\n"
"Hyperparameters `d_mean`, `d_noise` chosen on a held-out record the model never\n"
"saw during fitting (`data_test/`, seed 456), by held-out mean log predictive\n"
"density over the 12-row grid `d_mean in {1,2,3,4} x d_noise in {0,1,2}`. The winner\n"
"was refit on `data/` and stored as `y_pred_selected`, `sd_selected`; its stamp\n"
"`_source_selected` records which hyperparameters won. Plotted from those columns\n"
"only — this notebook does not read `data_test/`.")

code = nbformat.v4.new_code_cell(
'''mu_s = output_df["y_pred_selected"].to_numpy(float)[order]
sd_s = output_df["sd_selected"].to_numpy(float)[order]
tag = output_df["_source_selected"].iloc[0]

fig, ax = plt.subplots(figsize=(7, 4.5))
ax.scatter(x, y, s=18, color="#333333", zorder=3, label="training data")
ax.plot(x[order], mu_h + 2 * sd_h, color="#d62728", lw=1.2, alpha=0.7,
        label="stage-1 hetero-BLR $\\\\pm2$ sd (d_noise = 1)")
ax.plot(x[order], mu_h - 2 * sd_h, color="#d62728", lw=1.2, alpha=0.7)
ax.plot(x[order], mu_s, color="#2ca02c", lw=2, label=f"{tag} mean")
ax.fill_between(x[order], mu_s - 2 * sd_s, mu_s + 2 * sd_s, color="#2ca02c",
                alpha=0.25, label=f"{tag} $\\\\pm2$ sd")
ax.plot(grid, true_mean(grid) + 2 * true_sd(grid), "k--", lw=1.4,
        label="true $\\\\pm2$ sd (sd = 0.5 x)")
ax.plot(grid, true_mean(grid) - 2 * true_sd(grid), "k--", lw=1.4)
ax.set_xlabel("speed x [m/s]"); ax.set_ylabel("stopping distance y [m]")
ax.set_title("Selected on held-out data: closer to the true fan")
ax.legend(loc="upper left", fontsize=8)
fig.tight_layout(); fig.savefig("figures/selection.png", dpi=150)

print("redrew figures/selection.png from the record")
print(f"stamp: {tag}")
print(f"sd_selected at the slowest point : {sd_s[0]:.2f}   (x = {x[order][0]:.1f})")
print(f"sd_selected at the fastest point : {sd_s[-1]:.2f}   (x = {x[order][-1]:.1f})")''')

nb.cells[idx:idx] = [md, code]
nbformat.write(nb, 'pipeline.ipynb')
EOF
jupyter nbconvert --to notebook --execute --inplace pipeline.ipynb 2>&1 | grep -v Kernel | tail -2
python -c "
import nbformat
nb=nbformat.read('pipeline.ipynb',4)
for c in nb.cells:
    if c.cell_type=='code':
        for o in c.get('outputs',[]):
            if o.get('output_type')=='stream': print(o['text'])
            if o.get('output_type')=='error': print('ERROR', o['ename'], o['evalue'])
"
echo "=== data/experiment_data/output.csv (all 60 rows, joined with input) ==="
python -c "
import pandas as pd
from f3dasm import ExperimentData
i,o = ExperimentData.from_file('data').to_pandas()
pd.set_option('display.width', 250)
print(i.join(o).to_string(float_format=lambda v: f'{v:.3f}'))
"
ls figures data_test study_selection
```

*result (Bash)*

```
[NbConvertApp] Converting notebook pipeline.ipynb to notebook
[NbConvertApp] Writing 12204 bytes to pipeline.ipynb
record data/: 60 rows
output columns: ['y', 'y_pred_baseline', 'sd_baseline', '_source_baseline', 'y_pred_hblr', 'sd_hblr', '_source_modeler', 'y_pred_selected', 'sd_selected', '_source_selected']

redrew figures/baseline.png from the record

redrew figures/model.png from the record
sd_hblr at the slowest point : 6.60   (x = 3.0)
sd_hblr at the fastest point : 62.60   (x = 81.8)
sd_baseline everywhere       : 27.56

redrew figures/selection.png from the record
stamp: selected:d_mean=2,d_noise=2
sd_selected at the slowest point : 4.91   (x = 3.0)
sd_selected at the fastest point : 50.60   (x = 81.8)

rows: 60
input columns : ['x']
output columns: ['y', 'y_pred_baseline', 'sd_baseline', '_source_baseline', 'y_pred_hblr', 'sd_hblr', '_source_modeler', 'y_pred_selected', 'sd_selected', '_source_selected']

provenance -- who wrote what:
  _source_baseline       baseline
  _source_modeler        modeler
  _source_selected       selected:d_mean=2,d_noise=2

=== data/experiment_data/output.csv (all 60 rows, joined with input) ===
        x       y  y_pred_baseline  sd_baseline _source_baseline  y_pred_hblr  sd_hblr _source_modeler  y_pred_selected  sd_selected             _source_selected
0   3.000   5.231            4.732       27.563         baseline        4.927    6.603         modeler            4.862        4.907  selected:d_mean=2,d_noise=2
1  43.000 277.273          246.975       27.563         baseline      246.809   19.430         modeler          246.883       21.759  selected:d_mean=2,d_noise=2
2  63.000 462.586          489.048       27.563         baseline      488.688   35.411         modeler          488.487       36.588  selected:d_mean=2,d_noise=2
3  23.000  75.837           85.537       27.563         baseline       85.556   10.789         modeler           85.675       10.756  selected:d_mean=2,d_noise=2
4  33.000 144.306          156.177       27.563         baseline      156.104   14.495         modeler          156.230       15.682  selected:d_mean=2,d_noise=2
5  73.000 616.859          640.322       27.563         baseline      639.861   48.003         modeler          639.436       44.346  selected:d_mean=2,d_noise=2
6  53.000 333.519          357.933       27.563         baseline      357.670   26.167         modeler          357.636       28.857  selected:d_mean=2,d_noise=2
7  13.000  34.242           35.055       27.563         baseline       35.163    8.023         modeler           35.219        6.997  selected:d_mean=2,d_noise=2
8  18.000  63.780           57.776       27.563         baseline       57.840    9.287         modeler           57.935        8.724  selected:d_mean=2,d_noise=2
9  58.000 413.700          420.971       27.563         baseline      420.660   30.427         modeler          420.549       32.677  selected:d_mean=2,d_noise=2
10 78.000 722.720          723.519       27.563         baseline      723.007   55.875         modeler          722.448       48.013  selected:d_mean=2,d_noise=2
11 38.000 199.672          199.056       27.563         baseline      198.937   16.777         modeler          199.044       18.579  selected:d_mean=2,d_noise=2
12 28.000 119.252          118.337       27.563         baseline      118.310   12.517         modeler          118.440       13.075  selected:d_mean=2,d_noise=2
13 68.000 597.234          562.165       27.563         baseline      561.755   41.229         modeler          561.449       40.508  selected:d_mean=2,d_noise=2
14 48.000 309.377          299.934       27.563         baseline      299.720   22.532         modeler          299.747       25.198  selected:d_mean=2,d_noise=2
15  8.000  13.850           17.374       27.563         baseline       17.526    7.078         modeler           17.528        5.660  selected:d_mean=2,d_noise=2
16 10.500  35.286           25.584       27.563         baseline       25.715    7.503       
... [truncated]
```

**tool: Read**

```
{"file_path": "/home/mbessa/Dropbox/GitHub/3dasm_course/Lectures/short_course/demo_car/figures/selection.png"}
```

*result (Read)*

```

```

**assistant**

Selection is done and the record now has three authors. The winner is a degree-2 mean with a degree-2 log-noise polynomial, chosen on held-out data the model never saw during fitting.

**How I selected.** I created a held-out record of 200 points with the data generator's `--test --seed 456` option, and the model class was fit on `data/` only. The 12-candidate grid, mean degree 1 to 4 times noise degree 0 to 2, is itself an f3dasm record stored as `study_selection/`. The score is mean log predictive density on the held-out set, because that is the proper score for a probabilistic model and it rewards getting the noise right, not just the mean. Held-out MSE and the training evidence are stored alongside for reference. The winner was refit on the training record and written into `data/` as `y_pred_selected`, `sd_selected`, with a stamp that records the chosen hyperparameters. I did not edit the model class.

**The study, fit on data/, scored on data_test/:**

| d_mean | d_noise | held-out log pred density | held-out RMSE |
|---|---|---|---|
| 1 | 0 | -5.2978 | 47.12 |
| 1 | 1 | -5.2376 | 91.91 |
| 1 | 2 | -5.0479 | 59.48 |
| 2 | 0 | -4.5764 | 22.49 |
| 2 | 1 | -4.2604 | 22.59 |
| **2** | **2** | **-4.2395** | **22.56** |
| 3 | 0 | -4.5965 | 23.09 |
| 3 | 1 | -4.2744 | 23.48 |
| 3 | 2 | -4.2553 | 23.52 |
| 4 | 0 | -4.5922 | 22.92 |
| 4 | 1 | -4.2689 | 22.96 |
| 4 | 2 | -6.4179 | 29.23 |

All 12 optimisations reported convergence. Three things stand out.

- **The noise model is what separates candidates.** Every degree-2 mean has nearly the same held-out RMSE, but constant noise scores -4.58 and fanning noise scores -4.24. MSE alone would have picked constant noise, which is the baseline's mistake.
- **Degree 2 for the noise beats degree 1 by a small margin.** The refit winner gives standard deviation 4.2 at speed 3, 21.5 at speed 43, and 49.6 at speed 83, against a truth of 1.5, 21.5 and 41.5. The stage-1 log-linear model overshot at both ends, 5.8 and 63.3.
- **The (4, 2) corner is a bad local optimum.** Its training evidence is far below its neighbours despite the optimizer reporting success. It would not have won either way, but it is a real caveat about type-II maximum likelihood with a single start.

**The record.** Each of the 60 rows in `data/experiment_data/output.csv` now carries the observation, the baseline's mean and constant standard deviation, the stage-1 model's mean and standard deviation, the selected model's mean and standard deviation, and three provenance stamps reading `baseline`, `modeler` and `selected:d_mean=2,d_noise=2`. The full table printed above in the session output.

**Reproduction gate.** I appended two cells to the notebook that draw `figures/selection.png` from the selected columns only. The notebook executed cleanly with nbconvert, redrew all three figures, and does not read the held-out record.

**One thing worth saying plainly.** In this session I built the model, chose the score, created the held-out set, and picked the winner. Every step is in the record, but nobody other than me checked any of them.

Files touched: `select_model.py` (new), `data_test/` (new, 200 points), `study_selection/` (new, 12 rows), `data/` (three columns appended), `pipeline.ipynb` (two cells appended), `figures/selection.png` (new). The model class, the baseline, the data generator and git are untouched.

---

## Final answer

Selection is done and the record now has three authors. The winner is a degree-2 mean with a degree-2 log-noise polynomial, chosen on held-out data the model never saw during fitting.

**How I selected.** I created a held-out record of 200 points with the data generator's `--test --seed 456` option, and the model class was fit on `data/` only. The 12-candidate grid, mean degree 1 to 4 times noise degree 0 to 2, is itself an f3dasm record stored as `study_selection/`. The score is mean log predictive density on the held-out set, because that is the proper score for a probabilistic model and it rewards getting the noise right, not just the mean. Held-out MSE and the training evidence are stored alongside for reference. The winner was refit on the training record and written into `data/` as `y_pred_selected`, `sd_selected`, with a stamp that records the chosen hyperparameters. I did not edit the model class.

**The study, fit on data/, scored on data_test/:**

| d_mean | d_noise | held-out log pred density | held-out RMSE |
|---|---|---|---|
| 1 | 0 | -5.2978 | 47.12 |
| 1 | 1 | -5.2376 | 91.91 |
| 1 | 2 | -5.0479 | 59.48 |
| 2 | 0 | -4.5764 | 22.49 |
| 2 | 1 | -4.2604 | 22.59 |
| **2** | **2** | **-4.2395** | **22.56** |
| 3 | 0 | -4.5965 | 23.09 |
| 3 | 1 | -4.2744 | 23.48 |
| 3 | 2 | -4.2553 | 23.52 |
| 4 | 0 | -4.5922 | 22.92 |
| 4 | 1 | -4.2689 | 22.96 |
| 4 | 2 | -6.4179 | 29.23 |

All 12 optimisations reported convergence. Three things stand out.

- **The noise model is what separates candidates.** Every degree-2 mean has nearly the same held-out RMSE, but constant noise scores -4.58 and fanning noise scores -4.24. MSE alone would have picked constant noise, which is the baseline's mistake.
- **Degree 2 for the noise beats degree 1 by a small margin.** The refit winner gives standard deviation 4.2 at speed 3, 21.5 at speed 43, and 49.6 at speed 83, against a truth of 1.5, 21.5 and 41.5. The stage-1 log-linear model overshot at both ends, 5.8 and 63.3.
- **The (4, 2) corner is a bad local optimum.** Its training evidence is far below its neighbours despite the optimizer reporting success. It would not have won either way, but it is a real caveat about type-II maximum likelihood with a single start.

**The record.** Each of the 60 rows in `data/experiment_data/output.csv` now carries the observation, the baseline's mean and constant standard deviation, the stage-1 model's mean and standard deviation, the selected model's mean and standard deviation, and three provenance stamps reading `baseline`, `modeler` and `selected:d_mean=2,d_noise=2`. The full table printed above in the session output.

**Reproduction gate.** I appended two cells to the notebook that draw `figures/selection.png` from the selected columns only. The notebook executed cleanly with nbconvert, redrew all three figures, and does not read the held-out record.

**One thing worth saying plainly.** In this session I built the model, chose the score, created the held-out set, and picked the winner. Every step is in the record, but nobody other than me checked any of them.

Files touched: `select_model.py` (new), `data_test/` (new, 200 points), `study_selection/` (new, 12 rows), `data/` (three columns appended), `pipeline.ipynb` (two cells appended), `figures/selection.png` (new). The model class, the baseline, the data generator and git are untouched.

*duration: 106.2 s, turns: 6, cost: $0.666606*

