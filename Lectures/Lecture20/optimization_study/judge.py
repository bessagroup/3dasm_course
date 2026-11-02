"""The judge of Lecture 20: run each strategy of round 1 on many Latin hypercubes (15 points), with 15 probes each.

The judge is an f3dasm experiment: each row is (strategy, seed); the data generator runs that strategy once and
returns the best y it probed. Run it from the folder of Lecture 20:  python optimization_study/judge.py"""
import importlib.util
import os
import sys
from pathlib import Path

import numpy as np
from scipy.stats import qmc
from f3dasm import ExperimentData, datagenerator
from f3dasm.design import Domain

HERE = Path(__file__).resolve().parent
sys.path.insert(0, str(HERE))
from experiment import experiment  # noqa: E402

STRATEGIES = sorted(p.name for p in (HERE / "round1").iterdir() if (p / "strategy.py").exists())
SEEDS, BUDGET = 40, 15


@datagenerator(output_names=["best_y", "n_probes"])
def run_strategy(strategy, seed):
    """Run one strategy from a Latin hypercube with this seed; return the best y among its probes."""
    os.environ["OMP_NUM_THREADS"] = "1"
    seed = int(seed)
    X0 = qmc.LatinHypercube(3, seed=seed).random(15); y0 = experiment(X0); ys = []
    def probe(x):
        if len(ys) >= BUDGET:
            raise RuntimeError("budget exceeded")
        x = np.clip(np.asarray(x, float).ravel()[:3], 0, 1); ys.append(float(experiment(x)[0])); return ys[-1]
    spec = importlib.util.spec_from_file_location(strategy, HERE / "round1" / strategy / "strategy.py")
    module = importlib.util.module_from_spec(spec); spec.loader.exec_module(module)
    try:
        module.strategy(probe, X0.copy(), y0.copy(), BUDGET, seed)
    except Exception:
        pass                                            # a strategy that fails keeps the probes it made
    return (max(ys) if ys else float("nan")), len(ys)


if __name__ == "__main__":
    domain = Domain()
    domain.add_category("strategy", STRATEGIES)
    domain.add_int("seed", 0, SEEDS - 1)
    rows = [{"strategy": s, "seed": k} for s in STRATEGIES for k in range(SEEDS)]
    data = ExperimentData(domain=domain, input_data=rows)
    data = run_strategy.call(data=data, mode="parallel")
    data.store(HERE / "judge")
    print(data.to_pandas()[1].describe())
