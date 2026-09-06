"""Create the car stopping-distance dataset, the 2019 way.

Course canon (Lecture 17):

    y = z * x + 0.1 * x**2,     z ~ N(1.5, 0.5**2),     x in [3, 83] m/s

so the conditional mean and standard deviation are

    E[y|x]  = 1.5 * x + 0.1 * x**2
    sd[y|x] = 0.5 * x            <-- heteroscedastic, linear in x

Two ways to call it:

    python make_data.py                      -> data/       60 training points
                                                (Sobol, seed 123, noise seed 2019)

    python make_data.py --test --seed 456     -> data_test/  200 held-out points
                                                (random sampler, seed 456)

The held-out record is DELIBERATELY not created by default.  Whoever needs a
held-out set has to make it, and the record shows that they did.

Options: --test, --seed S, --n N, --out DIR
"""

from __future__ import annotations

import argparse

import numpy as np
from f3dasm import ExperimentData, create_sampler, datagenerator
from f3dasm.design import Domain

# ----------------------------------------------------------------------------
# Problem constants -- the single source of truth for the whole demo.
# ----------------------------------------------------------------------------
X_LOW, X_HIGH = 3.0, 83.0
MU_Z, SD_Z = 1.5, 0.5

N_TRAIN, N_TEST = 60, 200
SEED_TRAIN = 123          # sampler seed for the training record
SEED_TRAIN_NOISE = 2019   # generator seed for the training record


def true_mean(x):
    """E[y|x] = 1.5 x + 0.1 x^2."""
    return MU_Z * np.asarray(x) + 0.1 * np.asarray(x) ** 2


def true_sd(x):
    """sd[y|x] = 0.5 x."""
    return SD_Z * np.asarray(x)


def make_record(project_dir: str, n_samples: int, sampler_name: str,
                sampler_seed: int, noise_seed: int) -> ExperimentData:
    """Sample x, run the car data generator on it, store the record."""
    domain = Domain()
    domain.add_float("x", low=X_LOW, high=X_HIGH)

    data = ExperimentData(domain=domain)
    sampler = create_sampler(sampler_name, seed=sampler_seed)
    data = sampler.call(data, n_samples=n_samples)

    # One draw of z per experiment.  mode='sequential' fixes the order, so the
    # record is reproducible for a fixed seed.
    rng = np.random.default_rng(noise_seed)

    @datagenerator(output_names=["y"])
    def car(x: float) -> float:
        z = rng.normal(MU_Z, SD_Z)
        return float(z * x + 0.1 * x ** 2)

    data = car.call(data, mode="sequential")
    data.store(project_dir)

    xi, yo = data.to_pandas()
    xv, yv = xi["x"].to_numpy(), yo["y"].to_numpy()
    print(f"{project_dir}/: n = {len(data)}, sampler = {sampler_name} "
          f"(seed {sampler_seed}), noise seed {noise_seed}")
    print(f"  x in [{xv.min():.2f}, {xv.max():.2f}], "
          f"y in [{yv.min():.2f}, {yv.max():.2f}], mean(y) = {yv.mean():.3f}")
    print(f"  wrote {project_dir}/experiment_data/"
          "{domain.json,input.csv,output.csv,jobs.csv}")
    return data


def main() -> None:
    p = argparse.ArgumentParser(description=__doc__)
    p.add_argument("--test", action="store_true",
                   help="make a held-out record instead of the training one")
    p.add_argument("--seed", type=int, default=None,
                   help="seed for the held-out record (required with --test)")
    p.add_argument("--n", type=int, default=None, help="number of points")
    p.add_argument("--out", type=str, default=None, help="output directory")
    args = p.parse_args()

    if args.test:
        if args.seed is None:
            p.error("--test needs an explicit --seed (e.g. --seed 456)")
        make_record(args.out or "data_test", args.n or N_TEST,
                    "random_sampler", args.seed, args.seed + 1)
    else:
        make_record(args.out or "data", args.n or N_TRAIN,
                    "sobol_sampler", SEED_TRAIN, SEED_TRAIN_NOISE)


if __name__ == "__main__":
    main()
