"""
selector_loo_cv5.py

Fresh, standalone evaluation of ALL FIVE noise-model candidates now living in
car_noise_models.py (constant, linear, powerlaw, proportional,
additive_proportional). This supersedes selector_loo_cv.py / study_loo_cv /
study_loo_folds / study_pairwise, which only ever compared three candidates
(constant, powerlaw, proportional) and predate `linear` and
`additive_proportional`. Those old records are left on disk untouched but are
explicitly NOT used for any number in this report.

Does not edit car_noise_models.py, fit_noise_models.py, or any other model
code. Reads data/ read-only for x, y; writes only its own study_* records and
figures/selection.png.

The question this script exists to answer
------------------------------------------
The modeler fixed a constant-noise assumption and provisionally wrote
`additive_proportional` into the record, unjudged. Two questions:

  (1) Does *any* heteroscedastic fix (linear / powerlaw / proportional /
      additive_proportional) actually beat the constant-noise baseline on
      stops the fitting process never saw -- not just in-sample?
  (2) Is `additive_proportional` actually the best of the four fixes, or is
      it statistically indistinguishable from one or more of the others
      given only 50 points?

The test
--------
There is no held-out record and 50 measurements is the entire supply. The
only honest out-of-sample test buildable from that is refit-from-scratch
leave-one-out cross-validation (LOOCV): for each of the 50 drivers, refit
*everything* (the quadratic OLS mean AND the noise model's own parameters)
on the other 49, then score how well that refit model predicted the held-out
driver, as the Gaussian log predictive density of the true y under the
candidate's own (mean, sd) at that x. This is the same design as the earlier
3-candidate study, extended to 5 candidates, and it has the same weaknesses
(50 folds, mostly-shared training data, one dataset) -- see "what this does
not show" in the final report.

What gets scored
-----------------
1. Per-fold Gaussian log predictive density, summed/averaged over 50 folds
   -> headline LOO score per candidate (5 rows).
2. Per-fold standardized residual z=(y-mu)/sd -> calibration check: does the
   claimed sd(x) actually match the spread of held-out errors, not just the
   mean.
3. All C(5,2)=10 paired comparisons on the same 50 folds (paired t-test +
   Wilcoxon signed-rank), with a Bonferroni-adjusted alpha reported alongside
   the raw p-values, because testing 10 pairs out of one 50-point dataset
   inflates the chance of a "significant" pair by chance alone.
4. A bootstrap 95% CI on the mean LOO score per candidate (resampling folds),
   to show how much the ranking could move under resampling of the same 50
   points.
5. A single-fold leverage check: how much of any gap rides on one point.

Everything is written to three fresh f3dasm records (study_loo_folds5,
study_loo_cv5, study_pairwise5), all stamped _source_selector, plus
figures/selection.png (overwritten to reflect the current, 5-candidate
comparison; the study_pairwise / study_loo_cv / study_loo_folds three-
candidate records from the prior round are left as-is on disk).

Run:
    python selector_loo_cv5.py
"""
import itertools

import numpy as np
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
from scipy import stats as sstats

from f3dasm import ExperimentData, datagenerator
from f3dasm.design import Domain

from car_noise_models import CANDIDATES

RNG_SEED = 0  # bootstrap CI only; LOO folds themselves are deterministic


def loo_cv(x: np.ndarray, y: np.ndarray, name: str, cls) -> dict:
    """Leave-one-out CV for one candidate class. Refits mean+noise on the
    other n-1 points for every fold. Returns per-fold arrays."""
    n = len(x)
    mu = np.empty(n)
    sd = np.empty(n)
    logpdf = np.empty(n)
    z = np.empty(n)
    n_failed = 0
    for i in range(n):
        mask = np.ones(n, dtype=bool)
        mask[i] = False
        try:
            model = cls()
            model.fit(x[mask], y[mask])
            m_i, s_i = model.predict(np.array([x[i]]))
            mu[i], sd[i] = float(m_i[0]), float(s_i[0])
        except Exception as e:  # noqa: BLE001 - report, do not silently patch a candidate's bug
            n_failed += 1
            mu[i], sd[i] = np.nan, np.nan
            print(f"  [{name}] fold {i} (held-out x={x[i]:.2f}) raised: {e!r}")
            continue
        logpdf[i] = sstats.norm.logpdf(y[i], loc=mu[i], scale=sd[i])
        z[i] = (y[i] - mu[i]) / sd[i]
    return dict(mu=mu, sd=sd, logpdf=logpdf, z=z, n_failed=n_failed)


def main():
    data = ExperimentData.from_file("data")
    input_df, output_df = data.to_pandas()
    x = input_df["x"].to_numpy(float)
    y = output_df["y"].to_numpy(float)
    n = len(x)
    print(f"Loaded record: n = {n} measurements, x in [{x.min():.2f}, {x.max():.2f}]")
    print()

    names = list(CANDIDATES.keys())  # constant, linear, powerlaw, proportional, additive_proportional
    print(f"Candidates under test ({len(names)}): {names}")
    print()

    results = {}
    for name in names:
        print(f"=== LOO-CV refit: {name} ===")
        res = loo_cv(x, y, name, CANDIDATES[name])
        results[name] = res
        ok = res["logpdf"][~np.isnan(res["logpdf"])]
        print(f"  folds ok: {len(ok)}/{n}, folds failed: {res['n_failed']}")
        print(f"  mean LOO log predictive density per point: {ok.mean():.6f}")
        print(f"  total LOO log predictive density (sum over folds): {ok.sum():.6f}")
        print(f"  SEM of per-point LOO log-density (sample std / sqrt(n)): "
              f"{ok.std(ddof=1) / np.sqrt(len(ok)):.6f}")
        zok = res["z"][~np.isnan(res["z"])]
        print(f"  standardized LOO residual z=(y-mu)/sd: mean={zok.mean():.4f}, std={zok.std(ddof=1):.4f}")
        cov90 = np.mean(np.abs(zok) <= 1.645)
        cov50 = np.mean(np.abs(zok) <= 0.674)
        print(f"  empirical coverage of nominal 90% predictive interval: {cov90:.3f} (target 0.90)")
        print(f"  empirical coverage of nominal 50% predictive interval: {cov50:.3f} (target 0.50)")
        print()

    # ---------------- paired comparisons on the same 50 folds, ALL 10 pairs ----------------
    print("=== Paired comparisons (same folds, same mean fit each fold): all C(5,2)=10 pairs ===")
    pairs = list(itertools.combinations(names, 2))
    n_pairs = len(pairs)
    bonf_alpha = 0.05 / n_pairs
    print(f"  {n_pairs} pairs tested; Bonferroni-adjusted alpha for 0.05 family-wise = {bonf_alpha:.5f}")
    pairwise_rows = []
    for a, b in pairs:
        da = results[a]["logpdf"]
        db = results[b]["logpdf"]
        ok = ~np.isnan(da) & ~np.isnan(db)
        diff = db[ok] - da[ok]  # positive => b beats a on that fold
        t_stat, t_p = sstats.ttest_rel(db[ok], da[ok])
        try:
            w_stat, w_p = sstats.wilcoxon(diff)
        except ValueError as e:
            w_stat, w_p = np.nan, np.nan
            print(f"  wilcoxon({b} vs {a}) failed: {e!r}")
        mean_diff = float(diff.mean())
        sem_diff = float(diff.std(ddof=1) / np.sqrt(len(diff)))
        sig_bonf = bool(t_p < bonf_alpha)
        print(f"  {b} vs {a}: mean per-fold log-density diff = {mean_diff:+.5f} "
              f"(SEM {sem_diff:.5f}), paired t p={t_p:.4f}, Wilcoxon p={w_p:.4f}, "
              f"survives Bonferroni={sig_bonf}")
        pairwise_rows.append(dict(pair=f"{b}_minus_{a}", mean_diff=mean_diff, sem_diff=sem_diff,
                                   t_stat=float(t_stat), t_pvalue=float(t_p),
                                   wilcoxon_stat=float(w_stat) if not np.isnan(w_stat) else float("nan"),
                                   wilcoxon_pvalue=float(w_p),
                                   bonferroni_alpha=float(bonf_alpha),
                                   significant_bonferroni=sig_bonf))
    print()

    # ---------------- leverage check ----------------
    worst_fold = int(np.nanargmax(np.abs(results["constant"]["z"])))
    print(f"=== Leverage check: fold {worst_fold} (x={x[worst_fold]:.2f}, y={y[worst_fold]:.2f}) "
          f"has the largest |z| under 'constant' ===")
    for name in names:
        d = results[name]["logpdf"]
        keep = np.ones(n, dtype=bool)
        keep[worst_fold] = False
        ok = keep & ~np.isnan(d)
        print(f"  {name}: mean LOO log-density over all 50 folds = {d[~np.isnan(d)].mean():.5f}; "
              f"over the other 49 (excl. fold {worst_fold}) = {d[ok].mean():.5f}")
    print()

    # ---------------- bootstrap CI on the mean LOO score (resampling folds) ----------------
    print("=== Bootstrap 95% CI on mean LOO log-density (resampling the 50 folds, B=10000) ===")
    rng = np.random.default_rng(RNG_SEED)
    B = 10000
    boot_ci = {}
    boot_means_all = {}
    for name in names:
        d = results[name]["logpdf"]
        ok = d[~np.isnan(d)]
        idx = rng.integers(0, len(ok), size=(B, len(ok)))
        boot_means = ok[idx].mean(axis=1)
        boot_means_all[name] = boot_means
        lo, hi = np.percentile(boot_means, [2.5, 97.5])
        boot_ci[name] = (float(lo), float(hi))
        print(f"  {name}: mean={ok.mean():.5f}, 95% CI=[{lo:.5f}, {hi:.5f}]")
    print()

    # How often, under the SAME bootstrap resample of folds, does each fix beat
    # constant, and how often is additive_proportional the single best of the five?
    print("=== Bootstrap: P(fix beats constant) and P(candidate is best-of-5), same B=10000 resamples ===")
    # rebuild paired bootstrap using the SAME resample indices across candidates (paired resampling)
    ok_mask = np.ones(n, dtype=bool)
    for name in names:
        ok_mask &= ~np.isnan(results[name]["logpdf"])
    d_matrix = np.stack([results[name]["logpdf"][ok_mask] for name in names], axis=1)  # (n_ok, 5)
    n_ok = d_matrix.shape[0]
    rng2 = np.random.default_rng(RNG_SEED)
    idx2 = rng2.integers(0, n_ok, size=(B, n_ok))
    boot_matrix = d_matrix[idx2].mean(axis=1)  # (B, 5) mean per-fold logdensity per candidate per resample
    const_col = names.index("constant")
    prob_beats_constant = {}
    for j, name in enumerate(names):
        if name == "constant":
            continue
        p = float(np.mean(boot_matrix[:, j] > boot_matrix[:, const_col]))
        prob_beats_constant[name] = p
        print(f"  P({name} > constant) over resamples = {p:.4f}")
    best_idx = np.argmax(boot_matrix, axis=1)
    prob_best = {name: float(np.mean(best_idx == j)) for j, name in enumerate(names)}
    for name in names:
        print(f"  P({name} is best-of-5 on a resample) = {prob_best[name]:.4f}")
    print()

    # ================= write records =================
    fold_domain = Domain()
    fold_domain.add_int("fold", low=0, high=n - 1)
    fold_domain.add_category("candidate", categories=names)
    fold_rows = [{"fold": i, "candidate": name} for name in names for i in range(n)]
    fold_study = ExperimentData(domain=fold_domain, input_data=fold_rows)

    x_by_fold = x
    y_by_fold = y

    @datagenerator(output_names=["x_held_out", "y_held_out", "mu_pred", "sd_pred",
                                  "loglik", "z_resid", "_source_selector"])
    def score_fold(fold: int, candidate: str):
        r = results[candidate]
        return (float(x_by_fold[fold]), float(y_by_fold[fold]),
                float(r["mu"][fold]), float(r["sd"][fold]),
                float(r["logpdf"][fold]), float(r["z"][fold]), "selector")

    fold_study = score_fold.call(fold_study, mode="sequential")
    fold_study.store("study_loo_folds5")
    print("Wrote study_loo_folds5/ (50 folds x 5 candidates, per-fold LOO predictions & scores).")

    summary_domain = Domain()
    summary_domain.add_category("candidate", categories=names)
    summary_rows = [{"candidate": name} for name in names]
    summary_study = ExperimentData(domain=summary_domain, input_data=summary_rows)

    @datagenerator(output_names=["n_folds_ok", "n_folds_failed", "mean_loglik", "total_loglik",
                                  "sem_loglik", "ci95_lo", "ci95_hi", "z_mean", "z_std",
                                  "coverage90", "coverage50", "mean_loglik_excl_worst_fold",
                                  "prob_beats_constant", "prob_best_of_5",
                                  "_source_selector"])
    def summarize(candidate: str):
        r = results[candidate]
        ok = ~np.isnan(r["logpdf"])
        d = r["logpdf"][ok]
        zok = r["z"][ok]
        lo, hi = boot_ci[candidate]
        cov90 = float(np.mean(np.abs(zok) <= 1.645))
        cov50 = float(np.mean(np.abs(zok) <= 0.674))
        keep = np.ones(n, dtype=bool)
        keep[worst_fold] = False
        d_excl = r["logpdf"][keep & ok]
        pbc = prob_beats_constant.get(candidate, float("nan"))
        pb5 = prob_best[candidate]
        return (int(ok.sum()), int((~ok).sum()), float(d.mean()), float(d.sum()),
                float(d.std(ddof=1) / np.sqrt(len(d))), lo, hi,
                float(zok.mean()), float(zok.std(ddof=1)), cov90, cov50,
                float(d_excl.mean()), pbc, pb5, "selector")

    summary_study = summarize.call(summary_study, mode="sequential")
    summary_study.store("study_loo_cv5")
    print("Wrote study_loo_cv5/ (one row per candidate: the comparison table).")

    pair_domain = Domain()
    pair_names = [row["pair"] for row in pairwise_rows]
    pair_domain.add_category("pair", categories=pair_names)
    pair_study = ExperimentData(domain=pair_domain, input_data=[{"pair": p} for p in pair_names])

    @datagenerator(output_names=["mean_diff", "sem_diff", "t_stat", "t_pvalue",
                                  "wilcoxon_stat", "wilcoxon_pvalue", "bonferroni_alpha",
                                  "significant_bonferroni", "_source_selector"])
    def pairwise(pair: str):
        row = next(r for r in pairwise_rows if r["pair"] == pair)
        return (row["mean_diff"], row["sem_diff"], row["t_stat"], row["t_pvalue"],
                row["wilcoxon_stat"], row["wilcoxon_pvalue"], row["bonferroni_alpha"],
                row["significant_bonferroni"], "selector")

    pair_study = pairwise.call(pair_study, mode="sequential")
    pair_study.store("study_pairwise5")
    print("Wrote study_pairwise5/ (all 10 paired-test comparisons on shared folds).")
    print()

    # ================= print the full comparison table =================
    i_df, o_df = summary_study.to_pandas()
    full = i_df.join(o_df)
    print("=== Full comparison table (study_loo_cv5), ALL FIVE CANDIDATES ===")
    print(full.to_string(index=False))
    print()
    ip_df, op_df = pair_study.to_pandas()
    print("=== Pairwise table (study_pairwise5), all 10 pairs ===")
    print(ip_df.join(op_df).to_string(index=False))
    print()

    best = full.loc[full["mean_loglik"].idxmax(), "candidate"]
    print(f"Highest mean LOO log predictive density: '{best}'")

    # ================= figure =================
    colors = {"constant": "#888888", "linear": "#55a868", "powerlaw": "#4c72b0",
              "proportional": "#dd8452", "additive_proportional": "#c44e52"}
    order_for_plot = names

    fig, axes = plt.subplots(1, 3, figsize=(17, 5))

    ax = axes[0]
    means = [full.loc[full["candidate"] == nm, "mean_loglik"].iloc[0] for nm in order_for_plot]
    los = [boot_ci[nm][0] for nm in order_for_plot]
    his = [boot_ci[nm][1] for nm in order_for_plot]
    yerr = np.array([[m - lo, hi - m] for m, lo, hi in zip(means, los, his)]).T
    ax.bar(order_for_plot, means, yerr=yerr, capsize=5,
           color=[colors[nm] for nm in order_for_plot])
    ax.set_ylabel("mean LOO log predictive density")
    ax.set_title("LOO-CV score (bootstrap 95% CI over folds)")
    ax.tick_params(axis="x", rotation=25)

    ax = axes[1]
    z_stds = [full.loc[full["candidate"] == nm, "z_std"].iloc[0] for nm in order_for_plot]
    ax.bar(order_for_plot, z_stds, color=[colors[nm] for nm in order_for_plot])
    ax.axhline(1.0, color="k", linestyle="--", linewidth=1, label="ideal (std=1)")
    ax.set_ylabel("std of LOO standardized residual z")
    ax.set_title("Calibration: sd(x) vs actual spread")
    ax.legend()
    ax.tick_params(axis="x", rotation=25)

    ax = axes[2]
    for name in order_for_plot:
        r = results[name]
        ok = ~np.isnan(r["z"])
        ax.scatter(x[ok], r["z"][ok], s=18, alpha=0.6, label=name, color=colors[name])
    ax.axhline(0, color="k", linewidth=0.8)
    ax.axhline(1.645, color="k", linestyle=":", linewidth=0.8)
    ax.axhline(-1.645, color="k", linestyle=":", linewidth=0.8)
    ax.set_xlabel("x (velocity, held-out)")
    ax.set_ylabel("LOO standardized residual z")
    ax.set_title("z vs velocity (no trend if sd(x) is right)")
    ax.legend(fontsize=7)

    fig.suptitle("Selector: LOO-CV comparison of all 5 noise-model candidates (_source_selector)")
    fig.tight_layout()
    fig.savefig("figures/selection.png", dpi=150)
    print("Wrote figures/selection.png (overwritten with the fresh 5-candidate comparison).")


if __name__ == "__main__":
    main()
