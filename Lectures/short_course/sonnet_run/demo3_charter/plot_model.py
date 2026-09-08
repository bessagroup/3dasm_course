"""Draw figures/model.png from the record in data/, as it stands on disk.

Run after fit_candidates.py has written y_pred_mean / y_pred_sd:
    python plot_model.py
"""
from __future__ import annotations

import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np

from f3dasm import ExperimentData


def main():
    data = ExperimentData.from_file("data")
    input_df, output_df = data.to_pandas()
    x = input_df["x"].to_numpy(float)
    y = output_df["y"].to_numpy(float)
    y_pred_mean = output_df["y_pred_mean"].to_numpy(float)
    y_pred_sd = output_df["y_pred_sd"].to_numpy(float)
    source = output_df["_source_modeler"].iloc[0]
    model_name = (output_df["model_name"].iloc[0]
                  if "model_name" in output_df.columns else "unknown")

    order = np.argsort(x)
    xs = x[order]
    mean_s = y_pred_mean[order]
    sd_s = y_pred_sd[order]

    fig, ax = plt.subplots(figsize=(7, 5))
    ax.scatter(x, y, color="black", s=25, zorder=3, label="measurements (n=50)")
    ax.plot(xs, mean_s, color="C0", lw=2, label="fitted mean (quadratic OLS)")
    ax.fill_between(xs, mean_s - 2 * sd_s, mean_s + 2 * sd_s,
                     color="C0", alpha=0.2, label="mean +/- 2 sd (fitted noise model)")
    ax.set_xlabel("velocity x (m/s)")
    ax.set_ylabel("stopping distance y (m)")
    ax.set_title(f"Provisional model written to record (_source_{source})\n"
                  f"quadratic mean, {model_name} noise sd -- not yet judged")
    ax.legend(loc="upper left")
    fig.tight_layout()
    fig.savefig("figures/model.png", dpi=150)
    print("Wrote figures/model.png from data/ as read from disk.")


if __name__ == "__main__":
    main()
