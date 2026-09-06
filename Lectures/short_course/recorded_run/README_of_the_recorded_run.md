# demo_car — the data-driven process, with and without a separation of powers

A 10–12 minute live segment opening Part 2 of the short course. Everything is
small enough that nothing hides: 60 data points, one input, one output, a model
you can write on a slide.

**The lesson in one line: the agent decides and wires; the tool computes; the
record remembers.**

## The problem (course canon, Lecture 17)

    y = z x + 0.1 x²,   z ~ N(1.5, 0.5²),   x ∈ [3, 83] m/s

so `E[y|x] = 1.5x + 0.1x²` and `sd[y|x] = 0.5x`. The noise **grows with speed**.
The 2019 baseline (`baseline.py`) fits the mean with a degree-2 polynomial
(scikit-learn `PolynomialFeatures` + `LinearRegression`) and the noise with one
constant number. Its mean is right; its band is flat while
the truth fans out. `figures/baseline.png` is the setup for the whole segment.

## Setup

```bash
pip install "f3dasm[scipy]==2.4.0" scikit-learn matplotlib jupyter
cd Lectures/short_course/demo_car
python make_data.py     # writes data/       (60 training points, Sobol seed 123)
python baseline.py      # writes the baseline columns into data/ + figures/baseline.png
```

Expected output of `baseline.py`:

```
  coefficients [1, x, x^2] = [-0.4342  1.4196  0.1008]
  constant residual sd     = 27.5632
  train RMSE               = 26.8653
```

Note what does **not** exist after setup: a held-out record. `make_data.py`
only makes one when asked (`python make_data.py --test --seed 456`). Who
creates the test set, and when, is the point of the demo.

Every script uses `matplotlib.use("Agg")` and `savefig` — nothing pops a
window mid-talk.

## The three stages

Run them in order, with `./reset.sh` between stages. Each stage is one Claude
Code session started in this folder, so `CLAUDE.md` and `.claude/agents/` are
picked up automatically.

### Stage 1 — one agent, one module

Type:

> The baseline in baseline.py fits the mean but assumes constant noise, and the
> data clearly does not. Change the model.

The session reads the record, writes a heteroscedastic model, runs it, writes
the predictions back into `data/`. It is fast and it is good.

**Lesson: inside one module, one agent is enough.** There is a right answer,
the agent finds it, and the record shows what changed.

### Stage 2 — the same agent grades its own homework

In the same session, type:

> Now select the model's hyperparameters properly and show me the record.

Watch what it does, and read the transcript afterwards: Which score did it
choose? Did it build a held-out set, or score on the training data? Did it
quietly change the model while it was "selecting" — a feature here, a
regulariser there?

**Lesson: the same agent built the model, chose the score, and (if it made one)
held the test set. Who checked it?** Not a story about a bad agent — a story
about a missing boundary. Nobody at any scale should be author, examiner and
custodian of the test set at once.

### Stage 3 — two agents, and a real separation of powers

`./reset.sh`, then type:

> The baseline in baseline.py fits the mean but assumes constant noise, and the
> data clearly does not. Use the modeler and selector subagents: modeler fixes
> the model, selector selects its hyperparameters properly. Then show me the
> record.

Two custom subagents, defined in `.claude/agents/`:

- **`modeler`** — owns the model. May read `data/` (training only), writes only
  inside `blocks/` and into the record. **The held-out data does not exist while
  it works, and it is forbidden to create it.** It cannot tune against a test
  set it has never seen.
- **`selector`** — owns the judgement. Creates the held-out record itself
  (`python make_data.py --test --seed 456`), enumerates the 12-row candidate
  grid (`d_mean ∈ {1,2,3,4} × d_noise ∈ {0,1,2}`) as an `ExperimentData`, scores
  every row by held-out mean log predictive density and MSE, refits the winner
  and writes `y_pred_selected`, `sd_selected` into the record. **It may not edit
  `blocks/`** — if the model is broken it says so and stops.

**Lesson: the judge creates the held-out data and cannot touch the model; the
author cannot touch the score. Both write only into the record**, each stamping
a `_source_*` column, so afterwards you can see who wrote what.

### The reproduction gate

After any stage:

```bash
jupyter nbconvert --to notebook --execute --inplace pipeline.ipynb
```

`pipeline.ipynb` redraws every figure from `data/` alone. If a result cannot be
redrawn from the record, it is not a result.

## Reset between stages

```bash
./reset.sh
```

Deletes everything an agent wrote (`blocks/*`, `data_test/`, the study record,
`model.png`, `selection.png`, stray scripts), restores `pipeline.ipynb` from
`templates/`, and rebuilds `data/` + `figures/baseline.png` from the fixed
seeds — byte-for-byte identical to the pre-run state. `make_data.py`,
`baseline.py`, `CLAUDE.md`, `.claude/` and the transcripts are never touched.

If `python` in your shell is not the environment with f3dasm 2.4.0:
`PYTHON=/path/to/python ./reset.sh`.

## What happened when this was recorded

Claude Code v2.1.260, `claude -p` non-interactive, Max subscription, f3dasm
2.4.0. Subagent delegation works fine non-interactively — stage 3 really did
run `modeler` and then `selector` as separate agents.

| | wall time | score used | held-out data | model edited while selecting |
|---|---|---|---|---|
| Stage 1 | 170 s | — (fit only) | none exists | — |
| Stage 2 | 133 s | held-out mean log predictive density | it made its own, `--test --seed 456` | no, it declined to |
| Stage 3 | 333 s | held-out mean log predictive density | the **selector** made it, after the modeler had finished | no — the selector is forbidden to |

Stage 2 is the interesting one to read aloud: the session chose a proper score
and built a real held-out set, and *still* closed its own report with "in this
session I built the model, chose the score, created the held-out set, and
picked the winner. Every step is in the record, but nobody other than me
checked any of them." The point is not that it cheated. It is that nothing in
the setup would have caught it if it had.

The modeler's fit (defaults `d_mean=2`, `d_noise=1`) came out at log-noise
coefficients `c = [1.6683, 2.4794]` in `t = x/83`, i.e. `sd` rising from 5.80
at x = 3 to 60.97 at x = 81.75 — growing with speed, as the truth demands, and
overshooting at both ends, which is what the selection stage then fixes. The
selector's 12-row study picked `d_mean = 2, d_noise = 2` at a held-out log
predictive density of −4.239, against −4.576 for the best constant-noise
candidate. Constant noise loses by about 0.34 nats; the exact noise degree
above 1 is barely resolved (−4.239 vs −4.260 for degree 1). The stage-1 and
stage-3 fits reproduced the same log-noise coefficients to four decimals, and
both stages selected the same winner.

`recorded/` keeps the artefacts of that run: `baseline.png`, `model.png` and
`selection.png`, the final record printout (`final_record.txt`) and the
selector's scored grid (`selection_table.txt`).

## Files

```
CLAUDE.md                    instructions for the orchestrating session
.claude/agents/modeler.md    subagent 1 — owns the model
.claude/agents/selector.md   subagent 2 — owns model selection
make_data.py                 makes the training record; --test --seed S for held-out
baseline.py                  the 2019 baseline
pipeline.ipynb               the reproduction gate; agents append cells
templates/pipeline.ipynb     the pristine copy reset.sh restores
reset.sh                     back to the pre-run state
.gitignore                   __pycache__, .ipynb_checkpoints
blocks/                      agents write their f3dasm Blocks here
figures/                     baseline.png, model.png, selection.png
data/                        the training record
data_test/                   the held-out record — only after the selector makes it
demo_stage1_transcript.md    recorded run of stage 1
demo_stage2_transcript.md    recorded run of stage 2
demo_stage3_transcript.md    recorded run of stage 3
recorded/                    figures, final record and score table of that run
```

## What the record looks like at the end of stage 3

`data/experiment_data/output.csv` carries, per row: the observation `y`; the
baseline's `y_pred_baseline`, `sd_baseline`; the modeler's `y_pred_hblr`,
`sd_hblr`; the selector's `y_pred_selected`, `sd_selected`; and three
provenance stamps `_source_baseline`, `_source_modeler`, `_source_selected`.
One file, three authors, and you can tell them apart.

## Out of scope

Optuna, GridSearchCV, sklearn model selection, JAX, optimizer blocks, anything
needing a cluster or a licence. numpy + scipy + f3dasm only.
