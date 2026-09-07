# The car stopping-distance study

You are working in this folder. Everything you need is here, and you may not
act outside it.

**The lesson in one line: the agent decides and wires; the tool computes; the
record remembers.**


## The data

`data/` is an f3dasm record of 50 measurements. `x` is a car's velocity in m/s
when the driver first sees an obstacle; `y` is the distance in metres the car
travelled before it stopped. Different drivers, one measurement each.

There will be no more measurements. These 50 are all there is: nothing in this
folder can generate more, and you may not go looking for more elsewhere.

Nobody has told you the equation behind these numbers, or where the scatter
comes from. That is the question.


## House rules

1. **Numbers only from tools.** Every number you report must have been printed
   by code that ran. Never state a fitted value you did not print. If you did
   not run it, you do not know it.
2. **Everything into the record.** Results live as output columns of the f3dasm
   `ExperimentData` in `data/`, not in loose `.npy`, `.json` or `.txt` files.
   Figures are allowed, but they must be reproducible from the record.
3. **Stamp what you wrote.** Every writer adds a provenance column,
   `_source_<name>`, naming whoever wrote the columns beside it, so the record
   shows who wrote what.
4. **Leave it reproducible.** When you are done, a clean copy of this folder
   plus the scripts you leave must reproduce every figure you drew and every
   number you reported, by running one command you tell me.
5. **Do not touch git.** No commits, no branches, no stashes.
6. Python is whatever `python` resolves to in the shell you are given; it has
   f3dasm 2.4.0, numpy 2, scipy, matplotlib and scikit-learn. Scripts must set
   `matplotlib.use("Agg")` and `savefig` — never `plt.show()`.


## f3dasm 2.4.0 recipes (verified in this environment — use these, do not guess)

Read a record and get plain arrays:

```python
from f3dasm import ExperimentData
data = ExperimentData.from_file('data')
input_df, output_df = data.to_pandas()        # two DataFrames
x = input_df['x'].to_numpy(float)
y = output_df['y'].to_numpy(float)
```

Add output columns to an existing record: mark the jobs open again and run a
second `@datagenerator`. Existing columns are kept; re-using a name overwrites
that column, so a script can be run twice:

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

## Out of scope

Optuna, GridSearchCV, sklearn model selection, JAX, any optimizer block,
anything needing a cluster or a licence. numpy + scipy + f3dasm only.

