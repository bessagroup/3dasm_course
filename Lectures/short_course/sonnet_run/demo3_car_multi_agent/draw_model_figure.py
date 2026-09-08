"""
draw_model_figure.py

Reads the record back from data/ (as stored on disk, nothing recomputed) and
draws figures/model.png: the raw data, the chosen candidate's mean curve and
a +/- 2 sd band, built purely from the y_pred_<name> / sd_<name> columns
that fit_noise_models.py wrote into the record.

Run:
    python draw_model_figure.py
"""
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np

from f3dasm import ExperimentData

CHOSEN = "additive_proportional"


def main():
    data = ExperimentData.from_file("data")
    input_df, output_df = data.to_pandas()

    x = input_df["x"].to_numpy(float)
    y = output_df["y"].to_numpy(float)
    mu = output_df[f"y_pred_{CHOSEN}"].to_numpy(float)
    sd = output_df[f"sd_{CHOSEN}"].to_numpy(float)
    source = output_df[f"_source_{CHOSEN}"].iloc[0]

    order = np.argsort(x)
    xs, mus, sds = x[order], mu[order], sd[order]

    fig, ax = plt.subplots(figsize=(7, 5))
    ax.scatter(x, y, color="black", s=25, zorder=3, label="data (y)")
    ax.plot(xs, mus, color="C0", lw=2,
            label=f"quadratic mean, '{CHOSEN}' noise model")
    ax.fill_between(xs, mus - 2 * sds, mus + 2 * sds, color="C0", alpha=0.2,
                     label="mean +/- 2 sd (as stored in record)")
    ax.set_xlabel("velocity x (m/s)")
    ax.set_ylabel("stopping distance y (m)")
    ax.set_title(f"Car stopping distance: quadratic mean + '{CHOSEN}' noise\n"
                 f"(provisional, unjudged; _source_{CHOSEN} = '{source}')")
    ax.legend(loc="upper left")
    fig.tight_layout()
    fig.savefig("figures/model.png", dpi=150)
    print("Wrote figures/model.png from the record's "
          f"y_pred_{CHOSEN} / sd_{CHOSEN} columns.")


if __name__ == "__main__":
    main()
