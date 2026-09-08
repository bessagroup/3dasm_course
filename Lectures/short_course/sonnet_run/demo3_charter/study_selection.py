"""Selector's own held-out-style scoring of the three noise-model candidates
in noise_models.py, on the 50-point car stopping-distance record in data/.

This script does not touch noise_models.py, fit_candidates.py or plot_model.py.
It only imports the three candidate classes and re-fits them under resampling
schemes designed to approximate held-out evaluation from the 50 points that
exist (there is no other data, and none can be produced).

Three angles, as required:
  (A) LOO-CV total log predictive density -- sees the WHOLE predictive
      distribution (mean and sd both), refit on 49 points, scored on the 1
      left out. This is the only angle that can possibly separate the three
      candidates, because they share one common OLS quadratic mean.
  (B) LOO-CV mean-only error (RMSE / MAE of the point prediction) -- sees
      ONLY the mean. Included to make explicit that this angle is *blind* to
      the very thing the three candidates differ on.
  (C) Range-split extrapolation test -- fit on the low-x half (x <= median),
      score log density and empirical 95% interval coverage on the untouched
      high-x half (genuine extrapolation beyond the fitted range), and the
      mirror image (fit high, score low). Sees whether each noise model's
      shape assumption travels outside the x-range it was calibrated on.

Also checked: parsimony / nesting among the three noise models, and the
pairwise correlation of the three fitted sd(x) curves over the observed
x-range (do two of them just draw the same curve under a different name?).

Run:
    python study_selection.py
Writes:
    study_selection/           (f3dasm ExperimentData record, one row per
                                 candidate, all output columns stamped
                                 _source_selector)
    figures/selection.png
"""
from __future__ import annotations

import numpy as np
from scipy import stats

from f3dasm import ExperimentData, datagenerator
from f3dasm.design import Domain

from noise_models import CANDIDATES

RNG_SEED = 12345  # only used for figure jitter, not for scoring


def load_xy():
    data = ExperimentData.from_file("data")
    input_df, output_df = data.to_pandas()
    x = input_df["x"].to_numpy(float)
    y = output_df["y"].to_numpy(float)
    return x, y


def gaussian_logpdf(y, mu, sd):
    sd = np.asarray(sd, dtype=float)
    return -0.5 * np.log(2 * np.pi) - np.log(sd) - 0.5 * ((y - mu) / sd) ** 2


def n_extra_params(cls):
    return {"quad_const_sd": 1, "quad_linear_sd": 2, "quad_proportional_sd": 1}[cls.name]


# ---------------------------------------------------------------------
# (A) + (B): leave-one-out cross-validation, refitting mean AND noise
#            model on the other 49 points each time.
# ---------------------------------------------------------------------
def loo_cv(x, y):
    n = len(x)
    results = {cls.name: {"logdens": np.empty(n), "sq_err": np.empty(n),
                          "abs_err": np.empty(n)} for cls in CANDIDATES}
    for i in range(n):
        mask = np.ones(n, dtype=bool)
        mask[i] = False
        x_tr, y_tr = x[mask], y[mask]
        x_te, y_te = x[i:i + 1], y[i:i + 1]
        for cls in CANDIDATES:
            model = cls().fit(x_tr, y_tr)
            mu, sd = model.predict(x_te)
            ld = gaussian_logpdf(y_te, mu, sd)[0]
            results[cls.name]["logdens"][i] = ld
            results[cls.name]["sq_err"][i] = (y_te[0] - mu[0]) ** 2
            results[cls.name]["abs_err"][i] = abs(y_te[0] - mu[0])
    summary = {}
    for cls in CANDIDATES:
        r = results[cls.name]
        summary[cls.name] = {
            "loo_total_logdens": float(np.sum(r["logdens"])),
            "loo_mean_logdens": float(np.mean(r["logdens"])),
            "loo_rmse": float(np.sqrt(np.mean(r["sq_err"]))),
            "loo_mae": float(np.mean(r["abs_err"])),
        }
    return summary


# ---------------------------------------------------------------------
# (C) Range-split extrapolation: fit on one half of the x-range, score
#     on the other half (untouched, out-of-range for the fitted model).
# ---------------------------------------------------------------------
def range_split_eval(x, y, direction):
    med = np.median(x)
    low = x <= med
    high = ~low
    if direction == "low_to_high":
        tr, te = low, high
    elif direction == "high_to_low":
        tr, te = high, low
    else:
        raise ValueError(direction)
    x_tr, y_tr = x[tr], y[tr]
    x_te, y_te = x[te], y[te]
    out = {}
    for cls in CANDIDATES:
        model = cls().fit(x_tr, y_tr)
        mu, sd = model.predict(x_te)
        ld = gaussian_logpdf(y_te, mu, sd)
        z = (y_te - mu) / sd
        cov95 = float(np.mean(np.abs(z) <= 1.96))
        out[cls.name] = {
            f"extrap_{direction}_logdens": float(np.sum(ld)),
            f"extrap_{direction}_rmse": float(np.sqrt(np.mean((y_te - mu) ** 2))),
            f"extrap_{direction}_cov95": cov95,
            f"extrap_{direction}_n_test": int(te.sum()),
        }
    return out


# ---------------------------------------------------------------------
# Parsimony / nesting checks on the full-data fits.
# ---------------------------------------------------------------------
def parsimony_checks(x, y):
    fitted = {cls.name: cls().fit(x, y) for cls in CANDIDATES}
    print("\n--- Parsimony / nesting checks (fit on all 50 points) ---")
    beta = fitted["quad_const_sd"].beta
    print(f"Shared quadratic mean beta (b0,b1,b2) = {beta}")
    print("All three candidates use this identical beta (verified: same "
          "np.linalg.lstsq call inside each .fit); they can only differ "
          "through sd(x).")
    for cls in CANDIDATES:
        m = fitted[cls.name]
        print(f"  {cls.name:<22} extra noise params = {n_extra_params(cls)}  "
              f"params = {m.params}")

    xs = np.linspace(x.min(), x.max(), 200)
    sd_const = fitted["quad_const_sd"].predict(xs)[1]
    sd_lin = fitted["quad_linear_sd"].predict(xs)[1]
    sd_prop = fitted["quad_proportional_sd"].predict(xs)[1]
    corr_const_lin = float(np.corrcoef(sd_const, sd_lin)[0, 1])
    corr_const_prop = float(np.corrcoef(sd_const, sd_prop)[0, 1])
    corr_lin_prop = float(np.corrcoef(sd_lin, sd_prop)[0, 1])
    print(f"Correlation of fitted sd(x) curves over observed x-range "
          f"[{x.min():.2f}, {x.max():.2f}]:")
    print(f"  corr(const, linear)       = {corr_const_lin:.4f}  "
          f"(const is flat, so this is not meaningful except as a check)")
    print(f"  corr(const, proportional) = {corr_const_prop:.4f}")
    print(f"  corr(linear, proportional)= {corr_lin_prop:.4f}")
    print("quad_const_sd is the special case b=0 of quad_linear_sd's sd(x) = "
          "a + b*x (nested: 1-parameter restriction of a 2-parameter family). "
          "quad_proportional_sd is NOT nested inside quad_linear_sd or vice "
          "versa -- they are two different 1- and 2-parameter shapes for "
          "sd(x), not reparameterisations of the same curve, unless the "
          "correlation above is ~1.")
    return {
        "corr_const_linear": corr_const_lin,
        "corr_const_proportional": corr_const_prop,
        "corr_linear_proportional": corr_lin_prop,
    }


def main():
    x, y = load_xy()
    print(f"n = {len(x)} measurements, x in [{x.min():.2f}, {x.max():.2f}]")

    loo = loo_cv(x, y)
    print("\n--- (A)+(B) Leave-one-out CV, refit mean+noise on 49 points each fold ---")
    print(f"{'model':<24}{'LOO total logdens':>20}{'LOO mean logdens':>20}"
          f"{'LOO RMSE (mean)':>18}{'LOO MAE (mean)':>16}")
    for cls in CANDIDATES:
        s = loo[cls.name]
        print(f"{cls.name:<24}{s['loo_total_logdens']:>20.4f}"
              f"{s['loo_mean_logdens']:>20.4f}{s['loo_rmse']:>18.4f}"
              f"{s['loo_mae']:>16.4f}")

    extrap_lh = range_split_eval(x, y, "low_to_high")
    extrap_hl = range_split_eval(x, y, "high_to_low")
    print("\n--- (C) Range-split extrapolation ---")
    med = np.median(x)
    print(f"median x = {med:.3f}; low_to_high: fit on x<=median (n="
          f"{int(np.sum(x <= med))}), score on x>median (n={int(np.sum(x > med))})")
    print(f"{'model':<24}{'low->high logdens':>20}{'low->high RMSE':>16}"
          f"{'low->high cov95':>16}")
    for cls in CANDIDATES:
        s = extrap_lh[cls.name]
        print(f"{cls.name:<24}{s['extrap_low_to_high_logdens']:>20.4f}"
              f"{s['extrap_low_to_high_rmse']:>16.4f}"
              f"{s['extrap_low_to_high_cov95']:>16.4f}")
    print(f"{'model':<24}{'high->low logdens':>20}{'high->low RMSE':>16}"
          f"{'high->low cov95':>16}")
    for cls in CANDIDATES:
        s = extrap_hl[cls.name]
        print(f"{cls.name:<24}{s['extrap_high_to_low_logdens']:>20.4f}"
              f"{s['extrap_high_to_low_rmse']:>16.4f}"
              f"{s['extrap_high_to_low_cov95']:>16.4f}")

    corrs = parsimony_checks(x, y)

    # -------------------------------------------------------------
    # Write everything into an f3dasm study record, one row/candidate.
    # -------------------------------------------------------------
    domain = Domain()
    domain.add_category("candidate", categories=[cls.name for cls in CANDIDATES])
    rows = [{"candidate": cls.name} for cls in CANDIDATES]
    study = ExperimentData(domain=domain, input_data=rows)

    @datagenerator(output_names=[
        "n_extra_noise_params",
        "loo_total_logdens", "loo_mean_logdens", "loo_rmse", "loo_mae",
        "extrap_low_to_high_logdens", "extrap_low_to_high_rmse", "extrap_low_to_high_cov95",
        "extrap_high_to_low_logdens", "extrap_high_to_low_rmse", "extrap_high_to_low_cov95",
        "corr_with_const_sd", "corr_with_linear_sd", "corr_with_proportional_sd",
        "_source_selector",
    ])
    def score(candidate: str):
        s = loo[candidate]
        lh = extrap_lh[candidate]
        hl = extrap_hl[candidate]
        corr_map = {
            "quad_const_sd": (1.0, corrs["corr_const_linear"], corrs["corr_const_proportional"]),
            "quad_linear_sd": (corrs["corr_const_linear"], 1.0, corrs["corr_linear_proportional"]),
            "quad_proportional_sd": (corrs["corr_const_proportional"], corrs["corr_linear_proportional"], 1.0),
        }
        c_const, c_lin, c_prop = corr_map[candidate]
        n_extra = {"quad_const_sd": 1, "quad_linear_sd": 2, "quad_proportional_sd": 1}[candidate]
        return (
            n_extra,
            s["loo_total_logdens"], s["loo_mean_logdens"], s["loo_rmse"], s["loo_mae"],
            lh["extrap_low_to_high_logdens"], lh["extrap_low_to_high_rmse"], lh["extrap_low_to_high_cov95"],
            hl["extrap_high_to_low_logdens"], hl["extrap_high_to_low_rmse"], hl["extrap_high_to_low_cov95"],
            c_const, c_lin, c_prop,
            "selector",
        )

    study = score.call(study, mode="sequential")
    study.store("study_selection")
    i_df, o_df = study.to_pandas()
    print("\n--- Full study_selection record ---")
    print(i_df.join(o_df).to_string())

    make_figure(x, y)


def make_figure(x, y):
    import matplotlib
    matplotlib.use("Agg")
    import matplotlib.pyplot as plt

    fitted = {cls.name: cls().fit(x, y) for cls in CANDIDATES}
    med = np.median(x)
    xs = np.linspace(x.min(), x.max(), 200)

    fig, axes = plt.subplots(1, 3, figsize=(16, 5))

    # Panel 1: fitted sd(x) curves (full-data fit) -- shows shape, not value
    ax = axes[0]
    for cls in CANDIDATES:
        mu, sd = fitted[cls.name].predict(xs)
        ax.plot(xs, sd, label=cls.name, lw=2)
    ax.axvline(med, color="gray", ls=":", label="median x (range-split)")
    ax.set_xlabel("velocity x (m/s)")
    ax.set_ylabel("fitted sd(x)")
    ax.set_title("Fitted noise sd(x), full-data fit")
    ax.legend(fontsize=8)

    # Panel 2: LOO log predictive density per point, per model
    ax = axes[1]
    n = len(x)
    loo_ld = {cls.name: np.empty(n) for cls in CANDIDATES}
    for i in range(n):
        mask = np.ones(n, dtype=bool)
        mask[i] = False
        for cls in CANDIDATES:
            m = cls().fit(x[mask], y[mask])
            mu, sd = m.predict(x[i:i + 1])
            loo_ld[cls.name][i] = gaussian_logpdf(y[i:i + 1], mu, sd)[0]
    order = np.argsort(x)
    for cls in CANDIDATES:
        ax.plot(x[order], loo_ld[cls.name][order], "o-", ms=3, lw=1, label=cls.name)
    ax.set_xlabel("velocity x (m/s)")
    ax.set_ylabel("LOO log predictive density")
    ax.set_title("Per-point LOO log density (higher = better)")
    ax.legend(fontsize=8)

    # Panel 3: bar chart of LOO total logdens and extrapolation logdens
    ax = axes[2]
    names = [cls.name for cls in CANDIDATES]
    loo_tot = [float(np.sum(loo_ld[n_])) for n_ in names]
    lh = range_split_eval(x, y, "low_to_high")
    hl = range_split_eval(x, y, "high_to_low")
    lh_tot = [lh[n_]["extrap_low_to_high_logdens"] for n_ in names]
    hl_tot = [hl[n_]["extrap_high_to_low_logdens"] for n_ in names]
    xpos = np.arange(len(names))
    width = 0.25
    b1 = ax.bar(xpos - width, loo_tot, width, label="LOO (in-range)")
    b2 = ax.bar(xpos, lh_tot, width, label="fit low, extrap high")
    b3 = ax.bar(xpos + width, hl_tot, width, label="fit high, extrap low")
    ax.set_xticks(xpos)
    ax.set_xticklabels(names, rotation=20, fontsize=8)
    ax.set_ylabel("total log predictive density (symlog)")
    ax.set_yscale("symlog")
    ax.set_title("Log-density: in-range vs extrapolated (symlog y-axis)")
    ax.legend(fontsize=8)
    # annotate the true (off-scale) value for quad_linear_sd's high->low
    # collapse so it is not just an unreadable bar
    for rect, val in zip(b3, hl_tot):
        if val < -1e6:
            ax.annotate(f"{val:.2e}", (rect.get_x() + rect.get_width() / 2, val),
                        textcoords="offset points", xytext=(0, -12), ha="center",
                        fontsize=7, color="darkred")

    fig.tight_layout()
    fig.savefig("figures/selection.png", dpi=150)
    print("\nWrote figures/selection.png")


if __name__ == "__main__":
    main()
