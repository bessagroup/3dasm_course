"""Lecture 20, round 1 (the tournament), exactly as it ran: one agent, one assigned strategy written as code; a tool runs it
ONCE on the hidden experiment (LHS seed 0, 15 probes). The hidden experiment is written inline as _truth (the same
function as ../experiment.py). Usage: python run_round1.py <strategy 0-3> <replicate> <study_folder>
(0: L-BFGS on the experiment, 1: GP + L-BFGS, 2: linear regression + L-BFGS, 3: free, with a literature reviewer)"""
import csv, importlib.util, json, logging, sys, traceback
from pathlib import Path
import numpy as np
from scipy.stats import qmc
from adda import Agent, Graph, Edge, AgenticRun, LiteratureReviewAgent
logging.getLogger("adda").setLevel(logging.ERROR)

M, REP, study = int(sys.argv[1]), int(sys.argv[2]), Path(sys.argv[3]).resolve()
METHODS = [
 "L-BFGS directly on the experiment, with no model: scipy.optimize.minimize with method='L-BFGS-B' and bounds [0, 1], its gradients estimated by finite differences (each costs probes). You choose the starting point(s), the finite-difference step, and any restarts",
 "a Gaussian-process surrogate optimized with L-BFGS: fit a Gaussian process (scikit-learn) to all the experiments, find the maximum of its mean with scipy.optimize.minimize (method='L-BFGS-B', bounds [0, 1]), probe that point, and repeat. You choose the kernel, its hyperparameters and the restarts; do not use the GP's uncertainty to choose points",
 "a linear-regression surrogate optimized with L-BFGS: fit a linear model on basis functions of your choice (e.g. polynomials) to all the experiments, find its maximum with scipy.optimize.minimize (method='L-BFGS-B', bounds [0, 1]), probe that point, and repeat. You choose the basis, its degree and the regularization",
 "any method you choose: before choosing, ask the reviewer (a literature-review agent) which methods suit this problem, and follow its recommendation as you judge best"]


def _truth(X):                                          # the hidden experiment (revealed at the end)
    X = np.atleast_2d(X); a, b, c = X[:, 0], X[:, 1], X[:, 2]
    return np.min(np.stack([1.0 + 1.1*a + 0.3*c - 0.2*a*b, 0.4 + 2.2*b**1.5 + 0.6*a*c,
                            2.6 - 1.5*a**2 + 0.6*np.sin(2.5*b) - 0.7*c, 1.2 + 1.0*c + 0.8*(1-b)**2 - 0.4*a]), 0)


study.mkdir(parents=True, exist_ok=True)
X0 = qmc.LatinHypercube(3, seed=0).random(15); y0 = _truth(X0)
with open(study / "record.csv", "w", newline="") as f:
    w = csv.writer(f); w.writerow(["n", "source", "x1", "x2", "x3", "y"])
    for i, (x, y) in enumerate(zip(X0, y0)): w.writerow([i + 1, "LHS", *x, y])
(study / "PROBLEM_STATEMENT.md").write_text(f"""# Find the highest y, with a strategy written as code

An experiment returns one number y for an input x = (x1, x2, x3), each in [0, 1]. Its formula is unknown, and each
experiment is expensive. The goal: the input with the highest y.

- 15 experiments were already run (a Latin hypercube): `record.csv`.
- **Your method** (assigned): {METHODS[M]}. Stay with this method: you may change only its settings and the structure of its model.\n- **The result is the best experiment actually run.** A model's predicted maximum counts only after you probe it, and that probe is one of the 15.
- Write it as a Python function in `strategy.py`, next to this file:

      def strategy(probe, X0, y0, budget, seed):
          # probe(x) runs one experiment at x (a numpy array of 3 numbers in [0, 1]) and returns its y.
          # X0 (15 x 3) and y0 (15,) are the experiments already run; budget = 15 probes; seed: for any randomness.
          # Return nothing: every call to probe is recorded.

  It must use only numpy, scipy and scikit-learn, call probe at most `budget` times, and finish in under 2 minutes.
- Test it as much as you like on functions of your own. Then call the tool `RunStrategy` **once**: it runs your
  strategy on the real experiment with seed 0 and adds the 15 probes to `record.csv`. It cannot be run twice.
- Do not search the web yourself. End with a short report: your strategy, its settings, and its best experiment.
""")
state = {"done": False}


def RunStrategy() -> str:
    """Run strategy.py (in the study folder) once on the real experiment: 15 probes, seed 0. It cannot be run twice."""
    if state["done"]:
        return "REFUSED: the strategy has already been run."
    path = study / "strategy.py"
    if not path.exists():
        return "ERROR: strategy.py not found next to PROBLEM_STATEMENT.md."
    state["done"] = True
    spec = importlib.util.spec_from_file_location("strategy_mod", path); mod = importlib.util.module_from_spec(spec)
    calls = []
    def probe(x):
        if len(calls) >= 15:
            raise RuntimeError("budget exceeded")
        x = np.clip(np.asarray(x, float).ravel()[:3], 0, 1); y = float(_truth(x)[0]); calls.append((x, y)); return y
    try:
        spec.loader.exec_module(mod); mod.strategy(probe, X0.copy(), y0.copy(), 15, 0)
        note = ""
    except Exception:
        note = " The strategy raised an error after these probes: " + traceback.format_exc(limit=1).splitlines()[-1]
    with open(study / "record.csv", "a", newline="") as f:
        w = csv.writer(f)
        for i, (x, y) in enumerate(calls): w.writerow([16 + i, "strategy", *x, y])
    best = max([y for _, y in calls], default=float("nan"))
    return f"Ran {len(calls)} probes; best y among them = {best:.6f}; all added to record.csv.{note}"


agent = Agent(model="haiku", description="Writes and runs one optimization strategy.",
              tools={"Read", "Glob", "Grep", "Edit", "Bash", RunStrategy, "Done"},
              system_prompt=(f"Follow PROBLEM_STATEMENT.md in the study folder {study}: write strategy.py with the assigned"
                             f" strategy, test it on your own functions, call RunStrategy once, report, and call Done."
                             f" Run Python with {sys.executable}."))
LIGHT_RUNTIME = dict(pipeline_deliverable=False, reproduction_gate=False, hypothesis_ledger=False,
                     milestones_enabled=False, science_monitor=False, peer_interaction=False)
graph = Graph(nodes={"agent": agent}, edges=(), entry="agent") if M < 3 else Graph(
    nodes={"agent": agent, "reviewer": LiteratureReviewAgent(model="haiku")},
    edges=(Edge("agent", "reviewer", preamble="Search arXiv only: call SearchPapers with sources='arxiv'."),), entry="agent")
run = AgenticRun(study, graph=graph, budget=1500,
                 required_deliverables=["strategy.py"], runtime=LIGHT_RUNTIME, interactive=False, review_statement=False)
(study / "REPORT.md").write_text(str(run.execute()))
print(json.dumps({"method": M, "rep": REP, "ran": state["done"]}))
