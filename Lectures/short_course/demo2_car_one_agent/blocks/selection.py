"""Choose (d_mean, d_noise, link) from the data, not by taste.

Last turn the degrees were hand-picked: the mean was kept at degree 2 because
the baseline used degree 2, and the noise link was picked by comparing two
candidates.  That is one person's taste with a number attached.  This script
puts every candidate in one table and lets a single rule decide.

The candidate space (20 rows, `study_selection/`):

    d_mean  in 1..4      polynomial degree of the mean
    d_noise in 0..2      polynomial degree of log sd   (0 = one constant sd,
                                                        i.e. the BASELINE's
                                                        noise model)
    link    in log, lin  whether that polynomial is in log x or in x
                         (at d_noise = 0 the link makes no difference, so
                          only 'log' is listed -- 4*(1 + 2 + 2) = 20 rows)

The criterion is K-fold cross-validated **log predictive density** on `data/`.
Not RMSE: every candidate here has a polynomial mean, so RMSE barely moves and
cannot tell a good uncertainty from a bad one.  Not the training log evidence
either -- it is stored alongside for comparison, but the noise weights c are
point-optimised inside it (type-II maximum likelihood), so it under-penalises
a flexible noise model.  Cross-validation pays the full price of every fitted
parameter.

Every candidate is scored on the SAME folds (seed 7), so candidates can be
compared pairwise, point by point.

The winner is then chosen by the one-standard-error rule: take the best
cross-validated score, and among every candidate within one standard error of
it, keep the one with the fewest parameters.  The rule exists because the CV
score is itself a noisy estimate; without it you chase noise and pick a model
more complicated than the data can support.

`data_test/` is NOT touched here.  Selection happens on the training record
alone; the held-out record stays clean so that `blocks/holdout_check.py` can
score the chosen model on data no part of this decision has seen.

Writes:
    study_selection/     the whole table, one row per candidate
    data/                y_pred_selected, sd_selected, _source_selected, and
                         d_mean_selected, d_noise_selected, link_selected --
                         the record remembers WHICH model was chosen
    figures/selection.png, figures/selected.png

Run:  python -m blocks.selection
"""

from __future__ import annotations

import matplotlib
matplotlib.use("Agg")

import matplotlib.pyplot as plt
import numpy as np
from f3dasm import ExperimentData, datagenerator
from f3dasm.design import Domain

from blocks.heteroscedastic import LINKS, fit, plot_model

D_MEAN_GRID = (1, 2, 3, 4)
D_NOISE_GRID = (0, 1, 2)
K_FOLDS = 6          # 60 training points -> 6 folds of 10
SEED_FOLDS = 7       # fixed, so the study is reproducible and paired

_PER_POINT: dict[tuple, np.ndarray] = {}   # per-point CV lpd, for paired tests


def folds_of(n: int, k: int = K_FOLDS, seed: int = SEED_FOLDS):
    """The same deterministic split for every candidate."""
    return np.array_split(np.random.default_rng(seed).permutation(n), k)


def cv_log_predictive_density(x, y, d_mean, d_noise, link):
    """K-fold CV: every point is predicted exactly once, by a fit that never saw it."""
    lpd, sq = np.empty(len(x)), np.empty(len(x))
    for fold in folds_of(len(x)):
        keep = np.ones(len(x), dtype=bool)
        keep[fold] = False
        m = fit(x[keep], y[keep], d_mean, d_noise, link)
        mu, sd = m["mu_of"](x[fold]), m["sd_of"](x[fold])
        lpd[fold] = -0.5 * np.log(2.0 * np.pi * sd ** 2) - 0.5 * ((y[fold] - mu) / sd) ** 2
        sq[fold] = (y[fold] - mu) ** 2
    return lpd, sq


def candidate_rows():
    """The 20 candidates.  At d_noise = 0 the link is a no-op, so it is dropped."""
    return [{"d_mean": dm, "d_noise": dn, "link": lk}
            for dm in D_MEAN_GRID for dn in D_NOISE_GRID
            for lk in (("log",) if dn == 0 else LINKS)]


def run_study(x, y) -> ExperimentData:
    """Build the candidate table and score every row into it."""
    domain = Domain()
    domain.add_int("d_mean", low=min(D_MEAN_GRID), high=max(D_MEAN_GRID))
    domain.add_int("d_noise", low=min(D_NOISE_GRID), high=max(D_NOISE_GRID))
    domain.add_category("link", categories=list(LINKS))
    study = ExperimentData(domain=domain, input_data=candidate_rows())

    @datagenerator(output_names=["cv_lpd", "cv_lpd_se", "cv_rmse",
                                 "log_evidence", "train_rmse", "n_params",
                                 "_source_selection"])
    def score(d_mean: int, d_noise: int, link: str):
        d_mean, d_noise = int(d_mean), int(d_noise)
        lpd, sq = cv_log_predictive_density(x, y, d_mean, d_noise, link)
        _PER_POINT[(d_mean, d_noise, link)] = lpd
        full = fit(x, y, d_mean, d_noise, link)          # fitted on all 60 rows
        train_rmse = float(np.sqrt(np.mean((y - full["mu_of"](x)) ** 2)))
        return (float(lpd.mean()),
                float(lpd.std(ddof=1) / np.sqrt(len(lpd))),
                float(np.sqrt(sq.mean())),
                float(full["log_evidence"]), train_rmse,
                float(full["n_params"]), "selection")

    return score.call(study, mode="sequential")


def choose(table) -> dict:
    """One-standard-error rule: the simplest candidate that is not measurably worse."""
    best = table.loc[table["cv_lpd"].idxmax()]
    threshold = float(best["cv_lpd"] - best["cv_lpd_se"])
    within = table[table["cv_lpd"] >= threshold]
    pick = within.sort_values(["n_params", "cv_lpd"],
                              ascending=[True, False]).iloc[0]
    return {"best": best, "threshold": threshold, "within": within, "pick": pick}


# ----------------------------------------------------------------------------
def main() -> None:
    data = ExperimentData.from_file("data")
    input_df, output_df = data.to_pandas()
    x = input_df["x"].to_numpy(float)
    y = output_df["y"].to_numpy(float)
    sd_b = float(output_df["sd_baseline"].iloc[0])
    print(f"selecting on data/ only: {len(data)} rows, {K_FOLDS}-fold CV "
          f"(seed {SEED_FOLDS}), {len(candidate_rows())} candidates")

    study = run_study(x, y)
    study.store("study_selection")
    si, so = study.to_pandas()
    table = si.join(so)
    print("\n  wrote study_selection/ -- the whole table, one row per candidate")

    # --- the record, printed -------------------------------------------------
    print("\nSTUDY_SELECTION  (sorted by cv_lpd, the criterion; higher is better)")
    shown = table.sort_values("cv_lpd", ascending=False)
    print(shown.to_string(float_format=lambda v: f"{v:9.4f}"))

    verdict = choose(table)
    best, pick = verdict["best"], verdict["pick"]
    print(f"\n  best cv_lpd        : d_mean={int(best['d_mean'])}, "
          f"d_noise={int(best['d_noise'])}, link='{best['link']}'  ->  "
          f"cv_lpd = {best['cv_lpd']:.4f} +/- {best['cv_lpd_se']:.4f}")
    print(f"  one-s.e. threshold : cv_lpd >= {verdict['threshold']:.4f}  "
          f"({len(verdict['within'])} of {len(table)} candidates qualify)")
    print(f"  simplest qualifier : d_mean={int(pick['d_mean'])}, "
          f"d_noise={int(pick['d_noise'])}, link='{pick['link']}'  ->  "
          f"{int(pick['n_params'])} parameters  <-- SELECTED")

    # Paired comparison: the same folds scored the same points, so compare
    # point by point instead of differencing two noisy averages.
    kb = (int(best["d_mean"]), int(best["d_noise"]), best["link"])
    kp = (int(pick["d_mean"]), int(pick["d_noise"]), pick["link"])
    if kb != kp:
        d = _PER_POINT[kb] - _PER_POINT[kp]
        print(f"  paired difference (best - selected) = {d.mean():+.4f} +/- "
              f"{d.std(ddof=1) / np.sqrt(len(d)):.4f} -- the extra parameters "
              f"do not pay for themselves")

    # The baseline's noise model is in the table: d_noise = 0.
    base_row = table[(table["d_mean"] == 2) & (table["d_noise"] == 0)].iloc[0]
    d = _PER_POINT[kp] - _PER_POINT[(2, 0, base_row["link"])]
    print(f"\n  the baseline's noise model is candidate d_mean=2, d_noise=0: "
          f"cv_lpd = {base_row['cv_lpd']:.4f}")
    print(f"  selected - that candidate = {d.mean():+.4f} +/- "
          f"{d.std(ddof=1) / np.sqrt(len(d)):.4f}  (paired, {len(d)} points)")

    # --- refit the winner on all of data/ and write it into the record -------
    dm, dn, lk = int(pick["d_mean"]), int(pick["d_noise"]), str(pick["link"])
    model = fit(x, y, dm, dn, lk, verbose=True)
    mu_of, sd_of = model["mu_of"], model["sd_of"]

    @datagenerator(output_names=["y_pred_selected", "sd_selected",
                                 "d_mean_selected", "d_noise_selected",
                                 "link_selected", "_source_selected"])
    def predict_selected(x: float):
        return (float(mu_of(x)[0]), float(sd_of(x)[0]),
                float(dm), float(dn), lk, "selected")

    data = predict_selected.call(data.mark_all("open"), mode="sequential")
    data.store("data")
    print("\n  wrote y_pred_selected, sd_selected, d_mean_selected, "
          "d_noise_selected,\n  link_selected, _source_selected into data/")

    # --- figures -------------------------------------------------------------
    fig = plot_selection(table, verdict)
    fig.savefig("figures/selection.png", dpi=150)
    print("  wrote figures/selection.png")

    fig, ax = plt.subplots(figsize=(7, 4.5))
    plot_model(ax, x, y, model, sd_b, label="selected")
    ax.set_title(f"Selected by {K_FOLDS}-fold CV + 1-s.e. rule: "
                 f"d_mean={dm}, d_noise={dn}, link='{lk}'")
    fig.tight_layout()
    fig.savefig("figures/selected.png", dpi=150)
    print("  wrote figures/selected.png")


def plot_selection(table, verdict):
    """The study, as a picture: the criterion on the left, the evidence right."""
    pick = verdict["pick"]
    fig, axes = plt.subplots(1, 2, figsize=(11, 4.5), sharex=True)
    combos = sorted({(int(r.d_noise), r.link) for r in table.itertuples()})
    colors = plt.cm.viridis(np.linspace(0, 0.85, len(combos)))
    for ax, col, name in ((axes[0], "cv_lpd", f"{K_FOLDS}-fold CV log predictive density"),
                          (axes[1], "log_evidence", "training log evidence")):
        for (dn_i, lk_i), color in zip(combos, colors):
            sub = table[(table["d_noise"] == dn_i) & (table["link"] == lk_i)]
            sub = sub.sort_values("d_mean")
            label = (f"d_noise={dn_i} (constant sd)" if dn_i == 0
                     else f"d_noise={dn_i}, link='{lk_i}'")
            ax.plot(sub["d_mean"], sub[col], "o-", color=color, label=label)
        ax.set_xlabel("d_mean (degree of the mean)")
        ax.set_ylabel(name)
        ax.set_xticks(list(D_MEAN_GRID))
    axes[0].axhline(verdict["threshold"], color="#d62728", ls="--", lw=1.2,
                    label="best - 1 s.e.")
    axes[0].plot(pick["d_mean"], pick["cv_lpd"], "*", ms=20, color="#d62728",
                 zorder=5, label="selected")
    lo = float(table["cv_lpd"].max()) - 3.0
    if float(table["cv_lpd"].min()) < lo:
        axes[0].set_ylim(bottom=lo)
        axes[0].set_ylabel(f"{name}\n(clipped at best - 3; worse candidates run off)")
    axes[0].set_title("the criterion: cross-validated, so it pays for parameters")
    axes[1].set_title("for comparison: training evidence, which does not")
    axes[0].legend(fontsize=7, loc="lower right")
    fig.tight_layout()
    return fig


if __name__ == "__main__":
    main()
