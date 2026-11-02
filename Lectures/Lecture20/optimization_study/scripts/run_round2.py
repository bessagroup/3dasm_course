"""Lecture 20, round 2 (a team that talks), exactly as it ran: N agents x k probes each (N*k = 15) on the hidden
experiment, from a fixed LHS of 15 points (seed = the second argument), in lockstep rounds, with a coordinator and a
shared literature reviewer. The hidden experiment is written inline as _truth (the same function as ../experiment.py).
Usage: python run_round2.py 3 <lhs_seed> <study_folder> haiku"""
import csv, json, logging, sys, threading
from pathlib import Path
import numpy as np
from scipy.stats import qmc
from f3dasm import ExperimentData
from f3dasm.design import Domain
from adda import Agent, Graph, Edge, AgenticRun, LiteratureReviewAgent
logging.getLogger("adda").setLevel(logging.ERROR)

N, SEED, study = int(sys.argv[1]), int(sys.argv[2]), Path(sys.argv[3]).resolve()
MODEL = sys.argv[4] if len(sys.argv) > 4 else "haiku"
CHARTER = len(sys.argv) > 5 and sys.argv[5] == "charter"
METHODS = ["Bayesian optimization: fit a Gaussian process to all the experiments and probe where the expected improvement is largest (choose the kernel yourself)",
           "a quadratic response surface: fit a quadratic model to the experiments near the best one and probe its maximum inside a trust region",
           "local pattern search: probe a step from the best experiment along one direction, and shrink the step when no direction improves",
           "exploration: probe where the experiments are thinnest (the point farthest from every experiment so far)",
           "a random-forest surrogate: fit a random forest to all the experiments and probe where its expected improvement is largest"]
K = 15 // N                                             # probes per agent
names = [f"agent{i + 1}" for i in range(N)]


def _truth(X):                                          # the hidden experiment (revealed at the end)
    X = np.atleast_2d(X); a, b, c = X[:, 0], X[:, 1], X[:, 2]
    return np.min(np.stack([1.0 + 1.1*a + 0.3*c - 0.2*a*b, 0.4 + 2.2*b**1.5 + 0.6*a*c,
                            2.6 - 1.5*a**2 + 0.6*np.sin(2.5*b) - 0.7*c, 1.2 + 1.0*c + 0.8*(1-b)**2 - 0.4*a]), 0)


# ---- the study folder: the statement and the record (15 LHS experiments)
study.mkdir(parents=True, exist_ok=True)
domain = Domain()
for k in range(1, 4):
    domain.add_float(f"x{k}", 0.0, 1.0)
X0 = qmc.LatinHypercube(3, seed=SEED).random(15)
rows = [dict(n=i + 1, agent="LHS", round=0, x1=x[0], x2=x[1], x3=x[2], y=float(_truth(x)[0]), reason="initial Latin hypercube")
        for i, x in enumerate(X0)]
FIELDS = ["n", "agent", "round", "x1", "x2", "x3", "y", "reason"]
lock = threading.Lock()
count = {n: 0 for n in names}


def save():
    with open(study / "record.csv", "w", newline="") as f:
        w = csv.DictWriter(f, FIELDS); w.writeheader(); w.writerows(rows)
    data = ExperimentData(domain=domain, input_data=np.array([[r["x1"], r["x2"], r["x3"]] for r in rows]),
                          output_data=np.array([[r["y"]] for r in rows]))
    data.store(study / "experiments")                   # the same experiments as an f3dasm ExperimentData


save()
plural = "s" if K > 1 else ""
team_text = (f"The team has {N} agents ({', '.join(names)}), and each one has its own probe tool with {K} probe{plural}."
             f" The probes go in rounds: in each round every agent makes one probe, and no agent can probe again until all"
             f" the others have probed in that round ({K} round{plural} in total).") if N > 1 else \
            "You are the only agent: you have one probe tool with all 15 probes, and you may use them in any order."
(study / "PROBLEM_STATEMENT.md").write_text(f"""# Find the highest y

An experiment returns one number y for an input x = (x1, x2, x3), each in [0, 1]. Its formula is unknown, and each
experiment is expensive. The goal: the input with the **highest y** that you can find.

- 15 experiments were already run: a Latin hypercube, in `record.csv` (agent "LHS", round 0), and the same
  experiments as an f3dasm ExperimentData in `experiments/`.
- **Budget: 15 more experiments**, run only through the probe tools. {team_text}
- Every probe is added to `record.csv` with who made it, in which round, its y, and the reason given for it. Every
  agent reads `record.csv`: it is how the team shares what it has learned.
- Choose your strategy freely, and change it whenever the record suggests it. Any agent may ask the reviewer (a
  literature-review agent) which methods suit the problem. Do not search the web yourself; you may write and run
  code in your own folder.
- **The result is the best experiment actually run.** A model's predicted maximum counts only after it is probed.
- When the budget is spent, write `best.json` next to this file: {{"x1": ..., "x2": ..., "x3": ..., "y": ...}}, the
  best experiment in `record.csv` (it must be one that was run).
""")


def make_probe(name):
    def probe(x1: float, x2: float, x3: float, reason: str) -> str:
        with lock:
            if count[name] >= K:
                return f"REFUSED: {name} has used its {K} probes."
            if count[name] > min(count.values()):
                return "REFUSED: wait until every other agent has probed in this round."
            if not all(0.0 <= v <= 1.0 for v in (x1, x2, x3)):
                return "REFUSED: each input must be in [0, 1]."
            count[name] += 1
            y = float(_truth([x1, x2, x3])[0])
            rows.append(dict(n=len(rows) + 1, agent=name, round=count[name], x1=x1, x2=x2, x3=x3, y=y, reason=reason))
            save()
            return f"y = {y:.6f} at ({x1}, {x2}, {x3}); added to record.csv ({name}: {count[name]} of {K} probes used)."
    probe.__name__ = f"Probe_{name}"
    probe.__doc__ = (f"Run one experiment at (x1, x2, x3), each in [0, 1], and return its y. `reason`: one sentence on why"
                     f" this point. {name} has {K} probe{plural}, one per round. The result is added to record.csv.")
    return probe


TOOLS = {"Read", "Glob", "Grep", "Edit", "Bash"}
rules = f" Read PROBLEM_STATEMENT.md and record.csv in the study folder {study} first. Run Python with {sys.executable}."
if N == 1:
    solo = Agent(model=MODEL, description="Finds the highest y with 15 probes.", tools=TOOLS | {make_probe("agent1"), "Done"},
                 system_prompt="Find the input with the highest y, as stated. Spend the 15 probes, write best.json,"
                               " then end with a short report (your strategy, and the best experiment) and call Done." + rules)
    team = Graph(nodes={"agent1": solo}, edges=(), entry="agent1")
else:
    coordinator = Agent(model=MODEL, description="Runs the rounds and writes best.json.",
        tools={"Read", "Glob", "Grep", "Write", "Wait", "Done"},
        system_prompt=(f"You make no probes and write no code. Run {K} round{plural}: in each round, delegate to every agent"
                       f" ({', '.join(names)}) at the same time (wait=False) the task 'Make your probe for round r', with the"
                       f" full path of the study folder, then Wait for all of them. After the last round, write best.json from"
                       f" record.csv (its best experiment), end with a short report (what each agent tried, and the best"
                       f" experiment) and call Done." + rules))
    member = lambda n: Agent(model=MODEL, description=f"{n}: makes one probe per round.", tools=TOOLS | {make_probe(n)},
        system_prompt=(f"You are {n}, one agent of a team that looks for the input with the highest y. In each task, read"
                       f" record.csv (every experiment so far, with the reason of each teammate), decide your next point,"
                       f" call your probe tool once with a one-sentence reason, and report what you probed and why. You may ask the reviewer about methods first." + (f" Your method, which you apply in every round: {METHODS[names.index(n)]}. Write and run the code it needs in your own folder." if CHARTER else "") + rules))
    pre = "Search arXiv only: call SearchPapers with sources='arxiv'."
    team = Graph(nodes={"coordinator": coordinator, **{n: member(n) for n in names}, "reviewer": LiteratureReviewAgent(model="haiku")},
                 edges=tuple(Edge("coordinator", n) for n in names) + tuple(Edge(n, "reviewer", preamble=pre) for n in names),
                 entry="coordinator")

LIGHT_RUNTIME = dict(pipeline_deliverable=False, reproduction_gate=False, hypothesis_ledger=False,
                     milestones_enabled=False, science_monitor=False, peer_interaction=False)
run = AgenticRun(study, graph=team, budget=1500, required_deliverables=[],
                 runtime=LIGHT_RUNTIME, interactive=False, review_statement=False)
(study / "REPORT.md").write_text(str(run.execute()))
best = max(r["y"] for r in rows[15:]) if len(rows) > 15 else float("nan")
print(json.dumps({"N": N, "seed": SEED, "probes": len(rows) - 15, "best_probe_y": best, "best_lhs_y": max(r["y"] for r in rows[:15]),
                  "counts": count}))
