"""Model selection (Lecture 18) for the HeteroBLR block.

The `modeler` picked d_mean=2, d_noise=1 by hand and wrote y_pred_hblr,
sd_hblr into `data/`. This script does not trust that choice: it enumerates
the full (d_mean, d_noise) grid, fits each candidate on the training record
`data/`, and scores it on a held-out record `data_test/` that the modeler
never saw and could not have tuned against.

Score:
    log_pred_density : mean over held-out points of
                        log N(y | mean(x), sd(x)**2) under the candidate's
                        own predictive distribution -- rewards calibrated
                        uncertainty, not just a good mean.
    mse              : mean squared error of the predictive mean.

The winner is the row with the highest log_pred_density (NOT the lowest
mse -- those can disagree, and log_pred_density is the score that respects
uncertainty).

Writes:
    study_selection/           the 12-row scored study (ExperimentData)
    data/  (in place)          y_pred_selected, sd_selected, _source_selected
                                for the refit winner
    figures/selection.png      heatmap of log_pred_density over the grid

Run:  python select_model.py
"""

from __future__ import annotations

import numpy as np
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt

from f3dasm import ExperimentData, datagenerator
from f3dasm.design import Domain

from blocks.hetero_blr import HeteroBLR

D_MEAN_GRID = (1, 2, 3, 4)
D_NOISE_GRID = (0, 1, 2)


def log_normal_density(y, mean, sd):
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

    # --- build the candidate table by hand: the full grid, enumerated ---
    domain = Domain()
    domain.add_int("d_mean", low=1, high=4)
    domain.add_int("d_noise", low=0, high=2)
    rows = [{"d_mean": a, "d_noise": b}
            for a in D_MEAN_GRID for b in D_NOISE_GRID]
    study = ExperimentData(domain=domain, input_data=rows)

    @datagenerator(output_names=["log_pred_density", "mse"])
    def score(d_mean: int, d_noise: int):
        block = HeteroBLR(d_mean=int(d_mean), d_noise=int(d_noise))
        block.fit(x_train, y_train)
        mean_pred, sd_pred = block.predict(x_test)
        lpd = float(np.mean(log_normal_density(y_test, mean_pred, sd_pred)))
        mse = float(np.mean((y_test - mean_pred) ** 2))
        return lpd, mse

    study = score.call(study, mode="sequential")
    study.store("study_selection")

    input_df, output_df = study.to_pandas()
    table = input_df.join(output_df)
    print("\n===== model-selection study (scored on data_test, seed 456) =====")
    print(table.to_string())

    best_idx = output_df["log_pred_density"].idxmax()
    best_row = table.loc[best_idx]
    best_d_mean = int(best_row["d_mean"])
    best_d_noise = int(best_row["d_noise"])
    print("\n===== winner (highest log_pred_density) =====")
    print(best_row.to_string())

    if best_d_noise == 0:
        print("\n  NOTE: d_noise = 0 won -- the study selected a CONSTANT "
              "noise model, i.e. no better than the baseline's noise "
              "assumption. Reporting this plainly, not arguing with it.")

    # --- refit the winner on data/ and write it back into the record ---
    winner = HeteroBLR(d_mean=best_d_mean, d_noise=best_d_noise)
    winner.fit(x_train, y_train)
    mean_train, sd_train = winner.predict(x_train)
    print(f"\nRefit winner on data/: d_mean={best_d_mean}, d_noise={best_d_noise}")
    print(f"  fitted tau = {winner.tau:.6f}")
    print(f"  fitted noise-log-poly c = {np.array2string(winner.c, precision=6)}")

    @datagenerator(output_names=["y_pred_selected", "sd_selected",
                                 "_source_selected"])
    def predict_selected(x: float):
        mu, sd = winner.predict(np.array([x]))
        return float(mu[0]), float(sd[0]), "selector"

    train = train.mark_all("open")
    train = predict_selected.call(train, mode="sequential")
    train.store("data")
    print("  wrote y_pred_selected, sd_selected, _source_selected into "
          "data/ (baseline + modeler columns kept)")

    # --- figure: heatmap of log_pred_density over the (d_mean, d_noise) grid
    grid_lpd = np.full((len(D_MEAN_GRID), len(D_NOISE_GRID)), np.nan)
    for _, r in table.iterrows():
        i = D_MEAN_GRID.index(int(r["d_mean"]))
        j = D_NOISE_GRID.index(int(r["d_noise"]))
        grid_lpd[i, j] = r["log_pred_density"]

    fig, ax = plt.subplots(figsize=(5.5, 4.5))
    im = ax.imshow(grid_lpd, cmap="viridis", aspect="auto")
    ax.set_xticks(range(len(D_NOISE_GRID)))
    ax.set_xticklabels(D_NOISE_GRID)
    ax.set_yticks(range(len(D_MEAN_GRID)))
    ax.set_yticklabels(D_MEAN_GRID)
    ax.set_xlabel("d_noise")
    ax.set_ylabel("d_mean")
    ax.set_title("log_pred_density on held-out data_test (higher = better)")
    for i in range(len(D_MEAN_GRID)):
        for j in range(len(D_NOISE_GRID)):
            ax.text(j, i, f"{grid_lpd[i, j]:.2f}", ha="center", va="center",
                     color="white", fontsize=9)
    ax.scatter([D_NOISE_GRID.index(best_d_noise)],
               [D_MEAN_GRID.index(best_d_mean)],
               s=300, facecolors="none", edgecolors="red", linewidths=2,
               label="winner")
    ax.legend(loc="upper right", fontsize=8)
    fig.colorbar(im, ax=ax, label="log_pred_density")
    fig.tight_layout()
    fig.savefig("figures/selection.png", dpi=150)
    print("  wrote figures/selection.png")


if __name__ == "__main__":
    main()
