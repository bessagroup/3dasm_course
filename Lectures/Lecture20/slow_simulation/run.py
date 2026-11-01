import json
import time
from collections import Counter

import numpy as np

from f3dasm import ExperimentData, create_sampler, datagenerator
from f3dasm.design import Domain


@datagenerator(output_names=["y"])
def simulation(x1, x2, x3, x4, x5, x6, x7):
    time.sleep(2)
    if x1 + x2 > 1.7:
        raise RuntimeError("the simulation did not converge")
    return np.sin(5 * x1) + (2 * x2 - 0.6) ** 2


@datagenerator(output_names=["y"])
def simulation_fixed(x1, x2, x3, x4, x5, x6, x7):
    time.sleep(2)
    return np.sin(5 * x1) + (2 * x2 - 0.6) ** 2


def counts(data):
    return dict(Counter(es.job_status.name.lower() for _, es in data))


if __name__ == "__main__":
    domain = Domain()
    for i in range(1, 8):
        domain.add_float(f"x{i}", 0.0, 1.0)
    domain.add_output("y")

    design = create_sampler("sobol").call(ExperimentData(domain=domain), n_samples=64)

    t = time.perf_counter()
    seq = simulation.call(design, mode="sequential")
    t_seq = time.perf_counter() - t

    t = time.perf_counter()
    par = simulation.call(design, mode="parallel", nodes=8)
    t_par = time.perf_counter() - t
    par.store("stored_first")
    first = counts(par)

    failed = [i for i, es in par if es.is_status("error")]
    todo = par.mark(failed, "open")
    t = time.perf_counter()
    final = simulation_fixed.call(todo, mode="parallel", nodes=8)
    t_rerun = time.perf_counter() - t
    final.store("stored_final")

    json.dump(
        {
            "wall_time_sequential_s": t_seq,
            "wall_time_parallel_s": t_par,
            "status_counts_first": first,
            "n_rerun": len(failed),
            "wall_time_rerun_s": t_rerun,
            "status_counts_final": counts(final),
        },
        open("results.json", "w"),
        indent=2,
    )
