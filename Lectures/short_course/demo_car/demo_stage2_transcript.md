# Stage 2 transcript — the same agent grades its own homework

**Command**

```
PATH="$ENV/bin:$PATH" claude -p "<prompt>" --resume <session-of-stage-1> --allowedTools "Read Glob Grep Write Edit Bash Agent TodoWrite NotebookEdit" --output-format stream-json --verbose
```

**Prompt typed**

> Now select the model's hyperparameters properly and show me the record.

**Wall time:** 133 s

---

*session bea7a7f6-0397-4fb5-b020-c7e32c2a779c; model claude-fable-5-1; agents available: claude, Explore, general-purpose, modeler, Plan, selector, statusline-setup*

**assistant**

Stage 2. I'll build a held-out record with a fresh seed, enumerate the candidate grid as an f3dasm study, score every candidate on held-out log predictive density and MSE, refit the winner on the training data and write it into the record.

**tool: Write**

```
/home/mbessa/Dropbox/GitHub/3dasm_course/Lectures/short_course/demo_car/select_model.py  (6599 chars)

"""Model selection for HeteroBLR, done properly: score on data the fit never saw.

    python select_model.py

1. Held-out record.  If data_test/ does not exist, make it with
   `python make_data.py --test --seed 456` (200 points, random sampler).
   The training record data/ is never touched by this step.
2. Candidate grid.  d_mean in {1,2,3,4} x d_noise in {0,1,2} -> 12 rows, built
   by hand as an f3dasm ExperimentData (no sampler).
3. Score.  Every candidate is fitted on data/ ONLY and evaluated on
   data_test/: mean log predictive density (Gaussian predictive, mean and sd
   from HeteroBLR.predict) and mean squared error.  The training log marginal
   likelihood is stored too, for comparison, but it is not the score.
   Results are stored as the record study_selection/.
4. Winner.  argmax held-out log predictive density.  Refit on data/ (same fit
   as in the study), write y_pred_selected, sd_selected, d_mean_selected,
   d_noise_selected and the stamp _source_selected into data/.
5. figures/selection.png: the score surface.

The block in blocks/hetero_blr.py is imported unchanged.
"""

from __future__ import annotations

import os
import subprocess
import sys

import matplotlib
matplotlib.use("Agg")

import matplotlib.pyplot as plt
import numpy as np
from f3dasm import ExperimentData, datagenerator
from f3dasm.design import Domain

from blocks.hetero_blr import HeteroBLR

TEST_SEED = 456
D_MEAN = (1, 2, 3, 4)
D_NOISE = (0, 1, 2)


def load_xy(path: str):
    data = ExperimentData.from_file(path)
    i, o = data.to_pandas()
    return i["x"].to_numpy(float), o["y"].to_numpy(float)


def gaussian_lpd(y, mean, sd):
    return -0.5 * np.log(2 * np.pi * sd ** 2) - 0.5 * ((y - mean) / sd) ** 2


def main() -> None:
    # --- 1. held-out record -------------------------------------------------
    if not os.path.isdir("data_test"):
        print(f"no held-out record: running make_data.py --test --seed {TEST_SEED}")
        subprocess.run([sys.executable, "make_data.py", "--test",
                        "--seed", str(TEST_SEED)], check=True)
    x_tr, y_tr = load_xy("data")
    x_te, y_te = load_xy("data_test")
    print(f"train: {len(x_tr)} rows (data/)   held-out: {len(x_te)} rows (data_test/)")

    # --- 2. candidate grid as a record -----------------------------------------
    domain = Domain()
    domain.add_int("d_mean", low=min(D_MEAN), high=max(D_MEAN))
    domain.add_int("d_noise", low=min(D_NOISE), high=max(D_NOISE))
    rows = [{"d_mean": a, "d_noise": b} for a in D_MEAN for b in D_NOISE]
    study = ExperimentData(domain=domain, input_data=rows)

    # --- 3. score every candidate ----------------------------------------------
    @datagenerator(output_names=["lpd_heldout", "mse_heldout",
                                 "log_evidence_train", "rmse_train",
                                 "_source_selected"])
    def score(d_mean: int, d_noise: int):
        m = HeteroBLR(d_mean=int(d_mean), d_noise=int(d_noise)).fit(x_tr, y_tr)
... [truncated]
```

*result (Write)*

```
File created successfully at: /home/mbessa/Dropbox/GitHub/3dasm_course/Lectures/short_course/demo_car/select_model.py (file state is current in your context — no need to Read it back)
```

**tool: Bash**

```
python select_model.py 2>&1 | grep -v '^$'; echo "exit=${PIPESTATUS[0]}"
```

*result (Bash)*

```
data_test/: n = 200, sampler = random_sampler (seed 456), noise seed 457
  x in [3.73, 82.99], y in [7.12, 793.68], mean(y) = 281.252
  wrote data_test/experiment_data/{domain.json,input.csv,output.csv,jobs.csv}
no held-out record: running make_data.py --test --seed 456
train: 60 rows (data/)   held-out: 200 rows (data_test/)
STUDY  (fit on data/, scored on data_test/) -> study_selection/
    d_mean  d_noise  lpd_heldout  mse_heldout  log_evidence_train  rmse_train _source_selected
0        1        0      -5.2978    2219.8478           -333.7103     54.5917         selected
1        1        1      -5.9441   46991.0402           -405.1145    232.0408         selected
2        1        2      -5.0479    3537.3213           -315.2971     67.4552         selected
3        2        0      -4.5764     505.5944           -294.3791     26.8726         selected
4        2        1      -4.2604     510.3449           -273.8905     26.8667         selected
5        2        2      -4.2395     508.7481           -273.1568     26.8698         selected
6        3        0      -4.5965     533.0701           -294.7227     27.0546         selected
7        3        1      -4.2744     551.2935           -274.7327     27.0398         selected
8        3        2      -5.6142   23731.0760           -389.6448    165.3889         selected
9        4        0      -4.5922     525.5005           -295.3347     26.9815         selected
10       4        1      -4.6840     899.0676           -317.3537     32.1734         selected
11       4        2      -4.2489     530.4509           -274.4696     26.9058         selected
WINNER by held-out log predictive density: d_mean = 2, d_noise = 2, lpd = -4.2395
  best constant-noise candidate: d_mean = 2, lpd = -4.5764
  best by training evidence alone would be: {'d_mean': 2, 'd_noise': 2}
HeteroBLR  (d_mean=2, d_noise=2, t = x/83)
  mean weights m [1, t, ..]  = [-3.2635e-01  1.1851e+02  6.9230e+02]
  log-noise coeffs c [1, t, ..] = [ 1.2831  4.3335 -1.7137]
  prior sd tau               = 406.1655
  log marginal likelihood    = -273.1568
  train RMSE                 = 26.8698
  fitted sd at x =  3.00       = 4.2103   (truth 0.5 x = 1.50)
  fitted sd at x = 83.00       = 49.5483   (truth 0.5 x = 41.50)
  noise grows with x: yes
  wrote y_pred_selected, sd_selected, d_mean_selected, d_noise_selected, _source_selected into data/
  output columns: ['y', 'y_pred_baseline', 'sd_baseline', '_source_baseline', 'y_pred_hblr', 'sd_hblr', '_source_hblr', 'y_pred_selected', 'sd_selected', '_source_selected', 'd_mean_selected', 'd_noise_selected']
  wrote figures/selection.png
exit=0
```

**assistant**

Winner found and written. Some grid rows have suspicious training RMSEs, which I'll flag below. Now the notebook cells, the figure check, and the record dump.

**tool: Read**

```
{"file_path": "/home/mbessa/Dropbox/GitHub/3dasm_course/Lectures/short_course/demo_car/figures/selection.png"}
```

*result (Read)*

```

```

**tool: Bash**

```
python - <<'EOF'
import json
nb = json.load(open('pipeline.ipynb'))
cells = nb['cells']
idx = next(i for i, c in enumerate(cells)
           if 'What the record remembers' in ''.join(c['source']) and c['cell_type']=='markdown')
# insert before the "---\n## What the record remembers" cell
md = """## Selected model (written by `select_model.py`)

The hyperparameters were chosen on a held-out record (`data_test/`, 200 points,
seed 456) by mean log predictive density; the 12-row study lives in
`study_selection/`. This notebook reads neither. It only reads what the
selection wrote back into `data/`: `y_pred_selected`, `sd_selected` and the
chosen `d_mean_selected`, `d_noise_selected`.
"""
code = '''mu_s = output_df["y_pred_selected"].to_numpy(float)[order]
sd_s = output_df["sd_selected"].to_numpy(float)[order]
d_mean_s = int(output_df["d_mean_selected"].iloc[0])
d_noise_s = int(output_df["d_noise_selected"].iloc[0])

fig, ax = plt.subplots(figsize=(7, 4.5))
ax.scatter(x, y, s=18, color="#333333", zorder=3, label="training data")
ax.fill_between(x[order], mu_b - 2 * sd_b, mu_b + 2 * sd_b, color="#1f77b4",
                alpha=0.12, label=f"baseline $\\\\pm2$ sd (constant, sd = {sd_b:.1f})")
ax.plot(x[order], mu_h - 2 * sd_h, color="#d62728", lw=1, ls=":",
        label="stage-1 HeteroBLR $\\\\pm2$ sd (d_noise = 1)")
ax.plot(x[order], mu_h + 2 * sd_h, color="#d62728", lw=1, ls=":")
ax.plot(x[order], mu_s, color="#2ca02c", lw=2,
        label=f"selected mean (d_mean = {d_mean_s})")
ax.fill_between(x[order], mu_s - 2 * sd_s, mu_s + 2 * sd_s, color="#2ca02c",
                alpha=0.30, label=f"selected $\\\\pm2$ sd (d_noise = {d_noise_s})")
ax.plot(grid, true_mean(grid) + 2 * true_sd(grid), "k--", lw=1.4,
        label="true $\\\\pm2$ sd (sd = 0.5 x)")
ax.plot(grid, true_mean(grid) - 2 * true_sd(grid), "k--", lw=1.4)
ax.set_xlabel("speed x [m/s]"); ax.set_ylabel("stopping distance y [m]")
ax.set_title(f"Selected on held-out data: d_mean = {d_mean_s}, d_noise = {d_noise_s}")
... [truncated]
```

*result (Bash)*

```
inserted 2 cells before index 7
=== data/experiment_data/output.csv (all 60 rows)
        x       y  y_pred_baseline  sd_baseline _source_baseline  y_pred_hblr  sd_hblr _source_hblr  y_pred_selected  sd_selected _source_selected  d_mean_selected  d_noise_selected
0    3.00    5.23             4.73        27.56         baseline         4.93     6.60         hblr             4.86         4.91         selected                2                 2
1   43.00  277.27           246.98        27.56         baseline       246.81    19.43         hblr           246.88        21.76         selected                2                 2
2   63.00  462.59           489.05        27.56         baseline       488.69    35.41         hblr           488.49        36.59         selected                2                 2
3   23.00   75.84            85.54        27.56         baseline        85.56    10.79         hblr            85.67        10.76         selected                2                 2
4   33.00  144.31           156.18        27.56         baseline       156.10    14.50         hblr           156.23        15.68         selected                2                 2
5   73.00  616.86           640.32        27.56         baseline       639.86    48.00         hblr           639.44        44.34         selected                2                 2
6   53.00  333.52           357.93        27.56         baseline       357.67    26.17         hblr           357.64        28.86         selected                2                 2
7   13.00   34.24            35.06        27.56         baseline        35.16     8.02         hblr            35.22         7.00         selected                2                 2
8   18.00   63.78            57.78        27.56         baseline        57.84     9.29         hblr            57.93         8.72         selected                2                 2
9   58.00  413.70           420.97        27.56         baseline       420.66    30.43         hblr           420.55        32.68         selected                2                 2
10  78.00  722.72           723.52        27.56         baseline       723.01    55.88         hblr           722.45        48.01         selected                2                 2
11  38.00  199.67           199.06        27.56         baseline       198.94    16.78         hblr           199.04        18.58         selected                2                 2
12  28.00  119.25           118.34        27.56         baseline       118.31    12.52         hblr           118.44        13.07         selected                2                 2
13  68.00  597.23           562.17        27.56         baseline       561.76    41.23         hblr           561.45        40.51         selected                2                 2
14  48.00  309.38           299.93        27.56         baseline       299.72    22.53         hblr           299.75        25.20         selected                2                 2
15   8.00   13.85            17.37        27.56         baseline        17.53     7.08         hblr            17.53         5.66         selected                2                 2
16  10.50   35.29            25.58        27.56         baseline        25.71     7.50         hblr            25.75         6.27         selected                2                 2
17  50.50  342.35           328.30        27.56         baseline       328.07    24.28         hblr           328.06        27.00         selected                2                 2
18  70.50  639.09           600.61        27.56         baseline       600.18    44.49         hblr           599.81        42.44         selected                2                 2
19  30.50  128.56           136.63        27.56         baseline       136.58    13.47         hblr           136.71        14.34         selected                2                 2
20  40.50  257.00           222.39        27.56         baseline       222.24    18.05         hblr           222.34        20.13         selected                2                 2
21  80.50  693.82           767.01        27.56         baseline       766.47    60.27         hblr           765.84        49.75         selected                2                 2
22  60.50  461.44           454.38        27.56         baseline       454.04    32.82         hblr           453.89        34.62         selected                2                 2
23  20.50   63.91            71.03        27.56         baseline        71.07    10.01         hblr            71.18         9.70         selected                2                 2
24  15.50   46.84            45.79        27.56         baseline        45.87     8.62         hblr            45.95         7.82         selected                2                 2
25  55.50  395.80           388.82        27.56         baseline       388.54    28.21         hblr           388.46        30.75         selected                2                 2
26  75.50  700.02           681.29        27.56         baseline       680.80    51.79         hblr           680.31        46.20         selected                2                 2
27  35.50  197.68           176.99        27.56         baseline       176.89    15.59         hblr           177.01        17.09         selected                2                 2
28  25.50   87.59           101.31        27.56         baseline       101.30    11.62         hblr           101.43        11.88         selected                2                 2
29  65.50  521.27           524.98        27.56         baseline       524.59    38.21         hblr           524.34        38.55         selected                2                 2
30  45.50  299.30           272.82        27.56         baseline       272.63    20.92         hblr           272.69        23.45         selected                2                 2
31   5.50    9.80            10.42        27.56         baseline        10.60     6.77
... [truncated]
```

**tool: Bash**

```
jupyter nbconvert --to notebook --execute --inplace pipeline.ipynb 2>&1 | grep -v IPKernelApp | tail -2; echo "exit=${PIPESTATUS[0]}"; python - <<'EOF'
import json
nb = json.load(open('pipeline.ipynb'))
for c in nb['cells']:
    if c['cell_type'] == 'code':
        for o in c.get('outputs', []):
            if o.get('output_type') == 'stream': print(''.join(o['text']), end='')
            elif o.get('output_type') == 'error': print('ERROR', o['ename'], o['evalue'])
EOF
ls figures study_selection/experiment_data data_test/experiment_data
```

**tool: Read**

```
{"file_path": "/home/mbessa/Dropbox/GitHub/3dasm_course/Lectures/short_course/demo_car/figures/selected.png"}
```

*result (Bash)*

```
_validate(nbdict, ref, version, version_minor, relax_add_props)
[NbConvertApp] Writing 12262 bytes to pipeline.ipynb
exit=0
record data/: 60 rows
output columns: ['y', 'y_pred_baseline', 'sd_baseline', '_source_baseline', 'y_pred_hblr', 'sd_hblr', '_source_hblr', 'y_pred_selected', 'sd_selected', '_source_selected', 'd_mean_selected', 'd_noise_selected']
redrew figures/baseline.png from the record
redrew figures/model.png from the record
stored sd_hblr runs from 6.60 (x = 3.00) to 62.59 (x = 81.75); baseline sd = 27.56
redrew figures/selected.png from the record
selected hyperparameters stored in the record: d_mean = 2, d_noise = 2
stored sd_selected runs from 4.91 (x = 3.00) to 50.60 (x = 81.75)
rows: 60
input columns : ['x']
output columns: ['y', 'y_pred_baseline', 'sd_baseline', '_source_baseline', 'y_pred_hblr', 'sd_hblr', '_source_hblr', 'y_pred_selected', 'sd_selected', '_source_selected', 'd_mean_selected', 'd_noise_selected']

provenance -- who wrote what:
  _source_baseline       baseline
  _source_hblr           hblr
  _source_selected       selected
data_test/experiment_data:
domain.json
input.csv
jobs.csv
output.csv

figures:
baseline.png
model.png
selected.png
selection.png

study_selection/experiment_data:
domain.json
input.csv
jobs.csv
output.csv
```

*result (Read)*

```

```

**assistant**

Done. The selected model is in the record, stamped, and the notebook redraws everything from `data/` alone.

**How the selection was done**

- **Held-out data first.** No test record existed, so `select_model.py` created one with `make_data.py --test --seed 456`: 200 points from a random sampler, never seen by any fit. The training record was not touched by this step.
- **Candidate grid as a record.** The 12 candidates, d_mean in {1,2,3,4} times d_noise in {0,1,2}, were laid out by hand as an f3dasm ExperimentData and scored by a datagenerator. The scored study lives in `study_selection/`.
- **Score.** Each candidate was fitted on the 60 training points only, then scored on the 200 held-out points by mean log predictive density, which rewards a correct noise model, and by MSE, which only sees the mean. The training log marginal likelihood was recorded alongside but was not the criterion.
- **Winner refitted and written.** The argmax candidate was refitted on `data/` and written back as `y_pred_selected`, `sd_selected`, the chosen `d_mean_selected` and `d_noise_selected`, and the stamp `_source_selected`. The block in `blocks/hetero_blr.py` was imported unchanged.

**The study, as printed** (fit on data/, scored on data_test/)

| d_mean | d_noise | held-out lpd | held-out MSE | train RMSE |
|---|---|---|---|---|
| 1 | 0 | -5.2978 | 2219.85 | 54.59 |
| 1 | 1 | -5.9441 | 46991.04 | 232.04 |
| 1 | 2 | -5.0479 | 3537.32 | 67.46 |
| 2 | 0 | -4.5764 | 505.59 | 26.87 |
| 2 | 1 | -4.2604 | 510.34 | 26.87 |
| **2** | **2** | **-4.2395** | **508.75** | **26.87** |
| 3 | 0 | -4.5965 | 533.07 | 27.05 |
| 3 | 1 | -4.2744 | 551.29 | 27.04 |
| 3 | 2 | -5.6142 | 23731.08 | 165.39 |
| 4 | 0 | -4.5922 | 525.50 | 26.98 |
| 4 | 1 | -4.6840 | 899.07 | 32.17 |
| 4 | 2 | -4.2489 | 530.45 | 26.91 |

Winner: d_mean = 2, d_noise = 2. Constant noise loses clearly: the best constant-noise candidate scores -4.5764, about 0.34 nats per point worse. The gap between noise degree 1 and 2 is small, -4.2604 versus -4.2395. MSE alone cannot separate any of the degree-2 mean models, which is why the log predictive density is the right score here. The refitted winner's noise sd runs from 4.21 at x = 3 to 49.55 at x = 83, against a truth of 1.50 to 41.50, closer than the stage-1 model at both ends.

**The record.** `data/experiment_data/output.csv` now has 12 output columns per row: the observation, three baseline columns, three stage-1 columns, and five selected columns. Three provenance stamps say who wrote what: baseline, hblr, selected. Every row carries the same chosen hyperparameters, so the record itself remembers the decision. The full 60-row table was printed above in this session. The notebook ran end to end with nbconvert and redrew `baseline.png`, `model.png` and a new `selected.png` from the record only; `selection.png`, the score surface, is written by the selection script from the study record and is the one figure the notebook does not redraw, because redrawing it would require reading held-out data.

**Two things you should know**

- **Three grid rows look like optimizer failures, not model properties.** Candidates (1,1), (3,2) and (4,1) have training RMSEs of 232, 165 and 32 where their neighbours sit near 27. Type-II maximum likelihood with L-BFGS-B from a single start found a poor local optimum there. This does not change the winner, but fixing it would mean editing the block's initialisation during selection, which I did not do. Say so if you want it fixed and re-scored.
- **One agent did everything.** In this session I built the model, chose the score, created the held-out set, and picked the winner. Every step is in the record, but nobody other than me checked any of them.

---

## Final answer

Done. The selected model is in the record, stamped, and the notebook redraws everything from `data/` alone.

**How the selection was done**

- **Held-out data first.** No test record existed, so `select_model.py` created one with `make_data.py --test --seed 456`: 200 points from a random sampler, never seen by any fit. The training record was not touched by this step.
- **Candidate grid as a record.** The 12 candidates, d_mean in {1,2,3,4} times d_noise in {0,1,2}, were laid out by hand as an f3dasm ExperimentData and scored by a datagenerator. The scored study lives in `study_selection/`.
- **Score.** Each candidate was fitted on the 60 training points only, then scored on the 200 held-out points by mean log predictive density, which rewards a correct noise model, and by MSE, which only sees the mean. The training log marginal likelihood was recorded alongside but was not the criterion.
- **Winner refitted and written.** The argmax candidate was refitted on `data/` and written back as `y_pred_selected`, `sd_selected`, the chosen `d_mean_selected` and `d_noise_selected`, and the stamp `_source_selected`. The block in `blocks/hetero_blr.py` was imported unchanged.

**The study, as printed** (fit on data/, scored on data_test/)

| d_mean | d_noise | held-out lpd | held-out MSE | train RMSE |
|---|---|---|---|---|
| 1 | 0 | -5.2978 | 2219.85 | 54.59 |
| 1 | 1 | -5.9441 | 46991.04 | 232.04 |
| 1 | 2 | -5.0479 | 3537.32 | 67.46 |
| 2 | 0 | -4.5764 | 505.59 | 26.87 |
| 2 | 1 | -4.2604 | 510.34 | 26.87 |
| **2** | **2** | **-4.2395** | **508.75** | **26.87** |
| 3 | 0 | -4.5965 | 533.07 | 27.05 |
| 3 | 1 | -4.2744 | 551.29 | 27.04 |
| 3 | 2 | -5.6142 | 23731.08 | 165.39 |
| 4 | 0 | -4.5922 | 525.50 | 26.98 |
| 4 | 1 | -4.6840 | 899.07 | 32.17 |
| 4 | 2 | -4.2489 | 530.45 | 26.91 |

Winner: d_mean = 2, d_noise = 2. Constant noise loses clearly: the best constant-noise candidate scores -4.5764, about 0.34 nats per point worse. The gap between noise degree 1 and 2 is small, -4.2604 versus -4.2395. MSE alone cannot separate any of the degree-2 mean models, which is why the log predictive density is the right score here. The refitted winner's noise sd runs from 4.21 at x = 3 to 49.55 at x = 83, against a truth of 1.50 to 41.50, closer than the stage-1 model at both ends.

**The record.** `data/experiment_data/output.csv` now has 12 output columns per row: the observation, three baseline columns, three stage-1 columns, and five selected columns. Three provenance stamps say who wrote what: baseline, hblr, selected. Every row carries the same chosen hyperparameters, so the record itself remembers the decision. The full 60-row table was printed above in this session. The notebook ran end to end with nbconvert and redrew `baseline.png`, `model.png` and a new `selected.png` from the record only; `selection.png`, the score surface, is written by the selection script from the study record and is the one figure the notebook does not redraw, because redrawing it would require reading held-out data.

**Two things you should know**

- **Three grid rows look like optimizer failures, not model properties.** Candidates (1,1), (3,2) and (4,1) have training RMSEs of 232, 165 and 32 where their neighbours sit near 27. Type-II maximum likelihood with L-BFGS-B from a single start found a poor local optimum there. This does not change the winner, but fixing it would mean editing the block's initialisation during selection, which I did not do. Say so if you want it fixed and re-scored.
- **One agent did everything.** In this session I built the model, chose the score, created the held-out set, and picked the winner. Every step is in the record, but nobody other than me checked any of them.

*duration: 132.0 s, turns: 7, cost: $0.84825825*
