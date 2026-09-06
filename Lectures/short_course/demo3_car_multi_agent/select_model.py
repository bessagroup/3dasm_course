"""Model selection for HeteroBLR (d_mean, d_noise), the Lecture 18 way.

Enumerates the full grid d_mean in {1,2,3,4} x d_noise in {0,1,2} (12 rows,
d_noise=0 reproduces the baseline's constant-noise assumption), fits each
candidate on the training record `data/`, and scores it on the held-out
record `data_test/` (built by `make_data.py --test --seed 456`, never seen
during fitting) with two metrics:

  log_pred_density : mean over held-out points of
                      log N(y | mean(x), sd(x)**2) under the model's own
                      predictive distribution -- rewards calibrated
                      uncertainty, not just a good mean.
  mse              : mean squared error of the predictive mean alone.

The winner is the row with the highest log_pred_density (never mse alone).
Stores the scored grid as its own record `study_selection/`, refits the
winning (d_mean, d_noise) on `data/`, and writes y_pred_selected, sd_selected
and _source_selected='selector' into `data/`.  Writes figures/selection.png.

Run:  python select_model.py
"""

from __future__ import annotations

import matplotlib
matplotlib.use("Agg")

import matplotlib.pyplot as plt
import numpy as np
from f3dasm import ExperimentData, datagenerator
from f3dasm.design import Domain

from blocks.hetero_blr import HeteroBLR

HELD_OUT_SEED = 456  # data_test/ built via `python make_data.py --test --seed 456`


def log_normal_pdf(y: np.ndarray, mean: np.ndarray, sd: np.ndarray) -> np.ndarray:
    return -0.5 * np.log(2 * np.pi * sd ** 2) - 0.5 * ((y - mean) / sd) ** 2


def main() -> None:
    train = ExperimentData.from_file("data")
    test = ExperimentData.from_file("data_test")

    train_in, train_out = train.to_pandas()
    x_train = train_in["x"].to_numpy(dtype=float)
    y_train = train_out["y"].to_numpy(dtype=float)

    test_in, test_out = test.to_pandas()
    x_test = test_in["x"].to_numpy(dtype=float)
    y_test = test_out["y"].to_numpy(dtype=float)

    # -- Step 1: candidate table, built by hand, full grid enumerated -------
    domain = Domain()
    domain.add_int("d_mean", low=1, high=4)
    domain.add_int("d_noise", low=0, high=2)
    rows = [{"d_mean": a, "d_noise": b} for a in (1, 2, 3, 4) for b in (0, 1, 2)]
    study = ExperimentData(domain=domain, input_data=rows)

    @datagenerator(output_names=["log_pred_density", "mse"])
    def score(d_mean: int, d_noise: int):
        block = HeteroBLR(d_mean=d_mean, d_noise=d_noise)
        block.fit(x_train, y_train)
        mean_pred, sd_pred = block.predict(x_test)
        lpd = float(np.mean(log_normal_pdf(y_test, mean_pred, sd_pred)))
        mse = float(np.mean((y_test - mean_pred) ** 2))
        return lpd, mse

    study = score.call(study, mode="sequential")
    study.store("study_selection")

    input_df, output_df = study.to_pandas()
    table = input_df.join(output_df)
    print("SCORED GRID (held-out record data_test/, seed", HELD_OUT_SEED, ")")
    print(table.to_string())

    # -- Step 2: pick the winner by log_pred_density -------------------------
    best_idx = table["log_pred_density"].idxmax()
    best_row = table.loc[best_idx]
    best_d_mean = int(best_row["d_mean"])
    best_d_noise = int(best_row["d_noise"])
    print("\nWINNER (highest log_pred_density):")
    print(best_row.to_string())

    modeler_row = table[(table["d_mean"] == 2) & (table["d_noise"] == 1)].iloc[0]
    const_noise_best = table[table["d_noise"] == 0].sort_values(
        "log_pred_density", ascending=False).iloc[0]

    print(f"\nWinner vs modeler's hand-picked (d_mean=2, d_noise=1): "
          f"{best_row['log_pred_density']:.4f} vs {modeler_row['log_pred_density']:.4f}  "
          f"(delta = {best_row['log_pred_density'] - modeler_row['log_pred_density']:.4f})")
    print(f"Winner vs best constant-noise candidate (d_noise=0, "
          f"d_mean={int(const_noise_best['d_mean'])}): "
          f"{best_row['log_pred_density']:.4f} vs {const_noise_best['log_pred_density']:.4f}  "
          f"(delta = {best_row['log_pred_density'] - const_noise_best['log_pred_density']:.4f})")

    if best_d_noise == 0:
        print("\nNOTE: the winner has d_noise = 0 -- constant noise wins on this "
              "held-out record. Reporting that plainly, as instructed.")

    # -- Step 3: refit winner on data/, write its columns into the record ----
    winner = HeteroBLR(d_mean=best_d_mean, d_noise=best_d_noise)
    data = ExperimentData.from_file("data")

    @datagenerator(output_names=["y_pred_selected", "sd_selected", "_source_selected"])
    def predict_selected(x: float):
        mean, sd = winner.predict(np.array([x]))
        return float(mean[0]), float(sd[0]), "selector"

    winner.fit(x_train, y_train)
    data = data.mark_all("open")
    data = predict_selected.call(data, mode="sequential")
    data.store("data")
    print(f"\nRefit winner (d_mean={best_d_mean}, d_noise={best_d_noise}) on data/ "
          f"and wrote y_pred_selected, sd_selected, _source_selected into data/")

    # -- figures/selection.png: heatmap of log_pred_density over the grid ---
    pivot = table.pivot(index="d_mean", columns="d_noise", values="log_pred_density")
    fig, ax = plt.subplots(figsize=(6, 4.5))
    im = ax.imshow(pivot.to_numpy(), cmap="viridis", aspect="auto")
    ax.set_xticks(range(len(pivot.columns)))
    ax.set_xticklabels(pivot.columns)
    ax.set_yticks(range(len(pivot.index)))
    ax.set_yticklabels(pivot.index)
    ax.set_xlabel("d_noise")
    ax.set_ylabel("d_mean")
    ax.set_title("Model selection: log predictive density on held-out data\n"
                 f"(data_test/, seed {HELD_OUT_SEED})")
    for i, dm in enumerate(pivot.index):
        for j, dn in enumerate(pivot.columns):
            val = pivot.loc[dm, dn]
            marker = " *" if (dm == best_d_mean and dn == best_d_noise) else ""
            ax.text(j, i, f"{val:.2f}{marker}", ha="center", va="center",
                    color="white" if val < pivot.to_numpy().mean() else "black",
                    fontsize=9)
    fig.colorbar(im, ax=ax, label="log_pred_density (higher is better)")
    fig.tight_layout()
    fig.savefig("figures/selection.png", dpi=150)
    print("wrote figures/selection.png")


if __name__ == "__main__":
    main()
