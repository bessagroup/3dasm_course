"""
fit_noise_models.py

Fits every candidate in car_noise_models.py on the record in data/, prints
the numbers the fits produced (mean coefficients, noise parameters,
in-sample log-likelihood, and an exploratory tercile-residual check used only
to motivate which noise model is physically defensible -- not to score or
rank the candidates against each other).

Writes the predictions of ONE candidate (chosen for physical/diagnostic
reasons, not by comparing fit quality) into the f3dasm record as new output
columns, stamped with a _source_ column. This choice is provisional: it has
not been judged against the alternatives. That judgement belongs to a
separate selection step.

Run:
    python fit_noise_models.py
"""
import numpy as np
from f3dasm import ExperimentData, datagenerator

from car_noise_models import fit_all

CHOSEN = "additive_proportional"  # see printed diagnostics + report for the reasoning


def main():
    data = ExperimentData.from_file("data")
    input_df, output_df = data.to_pandas()
    x = input_df["x"].to_numpy(float)
    y = output_df["y"].to_numpy(float)

    print(f"Loaded record: n = {len(x)} samples, x in [{x.min():.3f}, {x.max():.3f}]")
    print()

    models = fit_all(x, y)

    print("=== Shared quadratic mean fit (OLS, identical for every candidate) ===")
    any_model = next(iter(models.values()))
    a, b, c = any_model.beta_
    print(f"mu(x) = {a:.6f} * x^2 + {b:.6f} * x + {c:.6f}")
    resid = any_model.residuals_
    print(f"residual RMS (in-sample) = {np.sqrt(np.mean(resid**2)):.6f}")
    print()

    print("=== Exploratory heteroscedasticity check (in-sample, informational only) ===")
    order = np.argsort(x)
    xs, rs = x[order], resid[order]
    n = len(xs)
    edges = np.array_split(np.arange(n), 3)
    for i, idx in enumerate(edges):
        lo, hi = xs[idx].min(), xs[idx].max()
        print(f"  x-tercile {i+1} [{lo:6.2f}, {hi:6.2f}]: "
              f"residual std = {np.std(rs[idx]):9.4f}, "
              f"|residual| mean = {np.mean(np.abs(rs[idx])):9.4f}")
    print("  (used only to motivate which noise model is physically sensible;")
    print("   not used to score or rank the candidates against each other)")
    print()

    print("=== Candidate noise models ===")
    for name, model in models.items():
        print(f"--- {name} ---")
        print(f"  params: {model.params_}")
        if hasattr(model, "optimizer_success_"):
            print(f"  optimizer success: {model.optimizer_success_} ({model.optimizer_message_})")
        print(f"  in-sample Gaussian log-likelihood: {model.loglik_:.4f}")
        mu_lo, sd_lo = model.predict(np.array([x.min()]))
        mu_hi, sd_hi = model.predict(np.array([x.max()]))
        print(f"  sd at x={x.min():.2f}: {sd_lo[0]:.4f}   sd at x={x.max():.2f}: {sd_hi[0]:.4f}")
        print()

    print(f"Chosen candidate for the record (provisional, unjudged): '{CHOSEN}'")
    chosen_model = models[CHOSEN]

    @datagenerator(output_names=[f"y_pred_{CHOSEN}", f"sd_{CHOSEN}", f"_source_{CHOSEN}"])
    def predict(x: float):
        mu, sd = chosen_model.predict(np.array([x]))
        return float(mu[0]), float(sd[0]), "modeler"

    data = data.mark_all("open")
    data = predict.call(data, mode="sequential")
    data.store("data")
    print("Wrote columns "
          f"y_pred_{CHOSEN}, sd_{CHOSEN}, _source_{CHOSEN} into data/.")


if __name__ == "__main__":
    main()
