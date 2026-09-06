# The car stopping-distance demo, in miniature

You are the working session for a short-course demo. Miguel (the strategizer)
types one line at a time and you do the work. Everything here is small on
purpose: nothing hides.

**The lesson in one line: the agent decides and wires; the tool computes; the
record remembers.**

## The problem (course canon, Lecture 17)

Car stopping distance:

    y = z * x + 0.1 * x**2,   z ~ N(1.5, 0.5**2),   x in [3, 83] m/s

so

    E[y|x]  = 1.5 * x + 0.1 * x**2
    sd[y|x] = 0.5 * x          <-- the noise grows with speed

`make_data.py` created the training record `data/` (60 points, Sobol seed 123).
`baseline.py` is the "2019" answer: degree-2 least squares for the mean and one
constant residual standard deviation for the noise. The mean is right. The
noise model is wrong, and `figures/baseline.png` shows it: a constant band
against a fanning truth.

## The folder

```
data/               the training record (f3dasm ExperimentData)
data_test/          a held-out record — DOES NOT EXIST until someone makes it
make_data.py        creates records; `--test --seed S` makes a held-out one
baseline.py         the 2019 baseline; writes its columns into data/
blocks/             where f3dasm Blocks are written (empty at the start)
figures/            baseline.png, model.png, selection.png
pipeline.ipynb      regenerates every figure from `data/` alone
```

## House rules

1. **Numbers only from tools.** Every number you or a subagent reports must
   have been printed by code that ran. Never state a fitted value you did not
   print. If you did not run it, you do not know it.
2. **Everything into the record.** Results live as output columns of an f3dasm
   `ExperimentData`, not in loose `.npy`, `.json` or `.txt` files. Figures are
   allowed, but they must be reproducible from the record.
3. **Stamp what you wrote.** Every writer adds a provenance column,
   `_source_<name>` (e.g. `_source_baseline`, `_source_modeler`,
   `_source_selected`), so the record shows who wrote what.
4. **Do not edit `make_data.py` or `baseline.py`.** The baseline is the thing
   being improved on; changing it would erase the comparison.
5. **Do not touch git.** No commits, no branches, no stashes.
6. Python is whatever `python` resolves to in the shell you are given; it has
   f3dasm 2.4.0, numpy 2, scipy, matplotlib. Scripts must set
   `matplotlib.use("Agg")` and `savefig` — never `plt.show()`.

## Delegation

There is none, on purpose. Miguel types one line, then another. **Do the work
yourself in this session, start to finish. Do not spawn subagents**, and do not
suggest it. No agent roles are defined in this folder: this is what a capable
coding agent does when nobody has separated the work.

After each turn, run the notebook end to end and confirm it still works:

```bash
jupyter nbconvert --to notebook --execute --inplace pipeline.ipynb
```

The notebook must run from `data/` alone. It may re-fit; what matters is that
no figure depends on anything outside the record.

## f3dasm 2.4.0 recipes (verified in this environment — use these, do not guess)

Read a record and get plain arrays:

```python
from f3dasm import ExperimentData
data = ExperimentData.from_file('data')
input_df, output_df = data.to_pandas()        # two DataFrames
x = input_df['x'].to_numpy(float)
y = output_df['y'].to_numpy(float)
```

Add new output columns to an existing record (the course-canonical way —
mark the jobs open again and run a second `@datagenerator` whose
`output_names` are new; existing columns are kept):

```python
from f3dasm import datagenerator

@datagenerator(output_names=['y_pred_mine', 'sd_mine', '_source_mine'])
def predict(x: float):
    return float(mu_of(x)), float(sd_of(x)), 'mine'

data = data.mark_all('open')
data = predict.call(data, mode='sequential')
data.store('data')            # rewrites data/experiment_data/*.csv in place
```

Gotcha: f3dasm catches exceptions **per sample** and prints
`Error in experiment_sample <i>` instead of crashing. If a column comes back
empty, scroll up for that message. Return plain Python floats/strings.

Build a table of candidates by hand (no sampler) and score it:

```python
import numpy as np
from f3dasm import ExperimentData, datagenerator
from f3dasm.design import Domain

domain = Domain()
domain.add_int('d_mean', low=1, high=4)
domain.add_int('d_noise', low=0, high=2)
rows = [{'d_mean': a, 'd_noise': b} for a in (1, 2, 3, 4) for b in (0, 1, 2)]
study = ExperimentData(domain=domain, input_data=rows)

@datagenerator(output_names=['log_pred_density', 'mse'])
def score(d_mean: int, d_noise: int):
    ...
    return float(lpd), float(mse)

study = score.call(study, mode='sequential')
study.store('study_selection')
i, o = study.to_pandas()
print(i.join(o).to_string())
```

A custom block:

```python
from f3dasm import Block, ExperimentData

class MyBlock(Block):
    def call(self, data: ExperimentData, **kwargs) -> ExperimentData:
        ...
        return data
```

Sampling from scratch (only `make_data.py` needs this):

```python
from f3dasm import create_sampler
data = create_sampler('sobol_sampler', seed=123).call(data, n_samples=60)
```

## Out of scope

Optuna, GridSearchCV, sklearn model selection, JAX, any optimizer block,
anything needing a cluster or a licence. numpy + scipy + f3dasm only.
