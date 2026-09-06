"""Choose (d_mean, d_noise) by cross-validation, not by taste.

Last step fixed the noise model but *I* picked d_mean = 2 and d_noise = 1 --
which is the same sin as the baseline, committed with a better model.  The
training log-likelihood cannot settle it either: it is the very quantity the
fit maximises, so it rises with every parameter added.

So the candidates are scored by repeated K-fold cross-validation **on `data/`
alone**.  Each fold fits on 48 points and is scored on the 12 it never saw.
The score is the mean log predictive density

    lpd = mean[ - log sd(x) - (y - mu(x))^2 / (2 sd(x)^2) - 0.5 log 2*pi ]

a *proper* scoring rule: it rewards a model for getting the spread right, not
just the centre.  RMSE is reported next to it precisely because it cannot --
RMSE only ever sees mu(x), so it is blind to the whole question at issue.

The grid is d_mean in 1..4 (the truth is 2) and d_noise in 0..2, where
d_noise = 0 is the baseline's constant band.  Nothing tells the search which
cell is right.

Writes the study as its own record `study_selection/`:

    cv_log_pred_density  mean held-out lpd over all folds   <-- the criterion
    cv_lpd_se            spread across repeats (optimistic, see note)
    cv_rmse              held-out RMSE of the mean          <-- the blind score
    n_params             d_mean + d_noise + 2
    n_fit_failures       folds where the optimiser did not converge
    _source_selection    provenance stamp

and the winner's refit-on-everything predictions into `data/`:

    y_pred_selected, sd_selected, _source_selected

Run:  python -m blocks.selection
"""

from __future__ import annotations

import matplotlib

matplotlib.use("Agg")

import matplotlib.pyplot as plt
import numpy as np
from f3dasm import ExperimentData, datagenerator
from f3dasm.design import Domain
from matplotlib.patches import Rectangle

from blocks.heteroscedastic import (fit_heteroscedastic, mean_log_density,
                                    mean_of, sd_of)

D_MEAN_GRID = (1, 2, 3, 4)
D_NOISE_GRID = (0, 1, 2)

N_SPLITS = 5
N_REPEATS = 5
CV_SEED = 2024


def cv_score(x, y, d_mean: int, d_noise: int, n_splits: int = N_SPLITS,
             n_repeats: int = N_REPEATS, seed: int = CV_SEED):
    """Repeated K-fold CV.  Returns (lpd, lpd_se, rmse, n_failures)."""
    rng = np.random.default_rng(seed)          # same folds for every candidate
    n = len(x)
    repeat_lpd, fold_lpd, sq_err, n_fail = [], [], [], 0

    for _ in range(n_repeats):
        perm = rng.permutation(n)
        lpds = []
        for fold in np.array_split(perm, n_splits):
            train = np.setdiff1d(perm, fold)
            a, b, _, res = fit_heteroscedastic(x[train], y[train],
                                               d_mean, d_noise)
            if not res.success:
                n_fail += 1
            lpd = mean_log_density(x[fold], y[fold], a, b, d_mean, d_noise)
            lpds.append(lpd)
            fold_lpd.append(lpd)
            sq_err.append((y[fold] - mean_of(x[fold], a, d_mean)) ** 2)
        repeat_lpd.append(np.mean(lpds))

    # Spread across the n_repeats full-CV scores.  Optimistic as a standard
    # error -- the repeats share all 60 points -- but it does show whether two
    # candidates are separated by more than fold-shuffling noise.
    se = float(np.std(repeat_lpd, ddof=1) / np.sqrt(n_repeats))
    return (float(np.mean(fold_lpd)), se,
            float(np.sqrt(np.mean(np.concatenate(sq_err)))), n_fail)


def selection_figure(table, d_mean_sel: int, d_noise_sel: int,
                     path: str = "figures/selection.png") -> None:
    """Two panels over the same 4x3 grid: the score that can see the noise
    model, and the score that cannot.  Both are oriented dark = better, one
    sequential hue each, and every cell carries its own number."""
    lpd = table.pivot(index="d_mean", columns="d_noise",
                      values="cv_log_pred_density")
    rmse = table.pivot(index="d_mean", columns="d_noise", values="cv_rmse")

    fig, axes = plt.subplots(1, 2, figsize=(11, 4.2))
    panels = [
        (axes[0], lpd, "Blues", "held-out log predictive density",
         "the proper score: sees mu(x) AND sd(x)", "{:.3f}"),
        (axes[1], rmse, "Blues_r", "held-out RMSE of the mean [m]",
         "the blind score: only ever sees mu(x)", "{:.1f}"),
    ]

    for ax, tab, cmap, cbar_label, subtitle, fmt in panels:
        values = tab.to_numpy(float)
        im = ax.imshow(values, cmap=cmap, aspect="auto")
        norm = plt.Normalize(values.min(), values.max())
        dark = (cmap == "Blues")
        for r in range(values.shape[0]):
            for c in range(values.shape[1]):
                frac = norm(values[r, c])
                is_dark = frac > 0.55 if dark else frac < 0.45
                ax.text(c, r, fmt.format(values[r, c]), ha="center",
                        va="center", fontsize=9,
                        color="white" if is_dark else "#1a1a1a")
        # the selected cell, outlined rather than recoloured
        r_sel = list(tab.index).index(d_mean_sel)
        c_sel = list(tab.columns).index(d_noise_sel)
        ax.add_patch(Rectangle((c_sel - 0.5, r_sel - 0.5), 1, 1, fill=False,
                               edgecolor="#d62728", lw=2.5, zorder=5))
        ax.set_xticks(range(len(tab.columns)), [str(c) for c in tab.columns])
        ax.set_yticks(range(len(tab.index)), [str(i) for i in tab.index])
        ax.set_xlabel("$d_{noise}$  (0 = the baseline's constant band)")
        ax.set_ylabel("$d_{mean}$")
        ax.set_title(subtitle, fontsize=10)
        for side in ("top", "right", "bottom", "left"):
            ax.spines[side].set_visible(False)
        ax.tick_params(length=0)
        cb = fig.colorbar(im, ax=ax, fraction=0.046, pad=0.04)
        cb.set_label(f"{cbar_label}\n(dark = better)", fontsize=8)
        cb.outline.set_visible(False)

    fig.suptitle(f"{N_SPLITS}-fold x {N_REPEATS} cross-validation on data/ "
                 f"— red box = selected ($d_{{mean}}$={d_mean_sel}, "
                 f"$d_{{noise}}$={d_noise_sel})", fontsize=11)
    fig.tight_layout(rect=(0, 0, 1, 0.94))
    fig.savefig(path, dpi=150)
    print(f"  wrote {path}")


def main() -> None:
    data = ExperimentData.from_file("data")
    input_df, output_df = data.to_pandas()
    x = input_df["x"].to_numpy(float)
    y = output_df["y"].to_numpy(float)

    print(f"cross-validating on data/ only: {len(x)} points, "
          f"{N_SPLITS}-fold x {N_REPEATS} repeats = "
          f"{N_SPLITS * N_REPEATS} fits per candidate, seed {CV_SEED}")
    print(f"  each fit sees {len(x) - len(x) // N_SPLITS} points, "
          f"each score uses the {len(x) // N_SPLITS} it never saw\n")

    # --- the candidate table, built by hand: 12 cells, nothing hidden --------
    domain = Domain()
    domain.add_int("d_mean", low=min(D_MEAN_GRID), high=max(D_MEAN_GRID))
    domain.add_int("d_noise", low=min(D_NOISE_GRID), high=max(D_NOISE_GRID))
    rows = [{"d_mean": a, "d_noise": b}
            for a in D_MEAN_GRID for b in D_NOISE_GRID]
    study = ExperimentData(domain=domain, input_data=rows)

    @datagenerator(output_names=["cv_log_pred_density", "cv_lpd_se", "cv_rmse",
                                 "n_params", "n_fit_failures",
                                 "_source_selection"])
    def score(d_mean: int, d_noise: int):
        lpd, se, rmse, n_fail = cv_score(x, y, int(d_mean), int(d_noise))
        return (float(lpd), float(se), float(rmse),
                int(d_mean) + int(d_noise) + 2, int(n_fail), "selection")

    study = score.call(study, mode="sequential")
    study.store("study_selection")

    si, so = study.to_pandas()
    table = si.join(so).sort_values("cv_log_pred_density", ascending=False)
    print("\nthe candidate table, best first "
          "(cv_log_pred_density: higher is better):")
    print(table.to_string(index=False))

    # --- the pick ------------------------------------------------------------
    # Taking the plain argmax would over-read the table: the top few cells sit
    # within fold-shuffling noise of each other, so "best" is not a fact.  The
    # one-standard-error rule is the usual discipline -- take every candidate
    # whose score is within 1 se of the best, then choose the SIMPLEST of them.
    best = table.iloc[0]
    runner = table.iloc[1]
    gap = float(best["cv_log_pred_density"] - runner["cv_log_pred_density"])
    pooled_se = float(np.hypot(best["cv_lpd_se"], runner["cv_lpd_se"]))

    print(f"\ntop score: d_mean = {int(best['d_mean'])}, "
          f"d_noise = {int(best['d_noise'])}   "
          f"cv lpd = {best['cv_log_pred_density']:.4f}")
    print(f"runner-up: d_mean = {int(runner['d_mean'])}, "
          f"d_noise = {int(runner['d_noise'])}   "
          f"cv lpd = {runner['cv_log_pred_density']:.4f}")
    print(f"  gap = {gap:.4f} nats/point, pooled se = {pooled_se:.4f} "
          f"-> {'separated' if gap > 2 * pooled_se else 'NOT separated'} "
          f"by more than fold noise")

    threshold = float(best["cv_log_pred_density"] - best["cv_lpd_se"])
    within = table[table["cv_log_pred_density"] >= threshold]
    print(f"\none-standard-error rule: keep cv lpd >= {threshold:.4f} "
          f"({len(within)} candidate(s)), then take the fewest parameters")
    print(within.to_string(index=False))
    pick = within.sort_values(["n_params", "cv_log_pred_density"],
                              ascending=[True, False]).iloc[0]
    d_mean, d_noise = int(pick["d_mean"]), int(pick["d_noise"])
    print(f"  -> selected d_mean = {d_mean}, d_noise = {d_noise} "
          f"({int(pick['n_params'])} parameters)")
    print(f"  (plain argmax would have said d_mean = {int(best['d_mean'])}, "
          f"d_noise = {int(best['d_noise'])}; the two rules "
          f"{'agree' if (d_mean, d_noise) == (int(best['d_mean']), int(best['d_noise'])) else 'DISAGREE'})")

    # --- why the scoring rule had to be the lpd and not RMSE ----------------
    by_rmse = table.sort_values("cv_rmse").iloc[0]
    print(f"\nscored by RMSE instead, the pick would be "
          f"d_mean = {int(by_rmse['d_mean'])}, "
          f"d_noise = {int(by_rmse['d_noise'])} "
          f"(cv_rmse = {by_rmse['cv_rmse']:.4f})")

    row = table[table["d_mean"] == d_mean].sort_values("d_noise")
    r_spread = float(row["cv_rmse"].max() - row["cv_rmse"].min())
    l_spread = float(row["cv_log_pred_density"].max()
                     - row["cv_log_pred_density"].min())
    print(f"  -- it lands on the same cell here, but by a hair and for a "
          f"borrowed reason.  Along the d_mean = {d_mean} row, RMSE moves "
          f"{r_spread:.4f} m ({100 * r_spread / row['cv_rmse'].min():.1f}%) "
          f"while the lpd moves {l_spread:.4f} nats.")
    print("     RMSE only ever touches mu(x), and d_noise changes mu(x) only "
          "indirectly, through the 1/sd^2 weights.")

    # --- refit the winner on everything and write it into the record --------
    a, b, nll, res = fit_heteroscedastic(x, y, d_mean, d_noise)
    print(f"\nrefit of the winner on all {len(x)} points "
          f"(converged = {res.success}):")
    print(f"  mean coefs      = {np.array2string(a, precision=4)}")
    print(f"  log-noise coefs = {np.array2string(b, precision=4)}")
    if d_noise == 1:
        print(f"  => sd(x) = {np.exp(b[0]):.4f} * x^{b[1]:.4f}    "
              f"(truth: 0.5 * x^1)")

    # The blunt version of the same point: keep mu(x) and multiply the band by
    # 10.  RMSE cannot tell the difference, because it never looks at sd.
    mu_sel = mean_of(x, a, d_mean)
    rmse_sel = float(np.sqrt(np.mean((y - mu_sel) ** 2)))
    b_inflated = b.copy()
    b_inflated[0] += np.log(10.0)
    print("\nthe same mean, with the uncertainty inflated 10x:")
    print(f"  RMSE:  selected {rmse_sel:.6f}   inflated {rmse_sel:.6f}   "
          f"(identical -- RMSE never evaluates sd)")
    print(f"  lpd :  selected "
          f"{mean_log_density(x, y, a, b, d_mean, d_noise):.4f}   "
          f"inflated "
          f"{mean_log_density(x, y, a, b_inflated, d_mean, d_noise):.4f}   "
          f"(the proper score collapses)")

    @datagenerator(output_names=["y_pred_selected", "sd_selected",
                                 "_source_selected"])
    def predict_selected(x: float):
        return (float(mean_of(np.array([x]), a, d_mean)[0]),
                float(sd_of(np.array([x]), b, d_noise)[0]),
                f"selected(d_mean={d_mean},d_noise={d_noise})")

    data = predict_selected.call(data.mark_all("open"), mode="sequential")
    data.store("data")
    print("\n  wrote y_pred_selected, sd_selected, _source_selected into data/")
    print("  wrote study_selection/ (the 12 candidates and their scores)")
    selection_figure(table, d_mean, d_noise)


if __name__ == "__main__":
    main()
