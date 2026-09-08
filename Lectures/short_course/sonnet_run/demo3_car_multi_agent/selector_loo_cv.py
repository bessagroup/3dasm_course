"""
selector_loo_cv.py

Judges the three noise-model candidates in car_noise_models.py
(constant, powerlaw, proportional) without touching that file or
fit_noise_models.py, and without editing anything the modeler wrote.

The test
---------
There is no held-out set and the 50 measurements in data/ are the entire
supply.  The fair thing that can be built out of 50 points is leave-one-out
cross-validation (LOOCV): for each of the 50 drivers, refit *everything* --
the quadratic mean (OLS) AND the noise model's own parameters (sigma / (c,p)
/ k) -- on the other 49, then ask how well the refit model predicted the one
driver that was held back, scored as the Gaussian log predictive density of
the true y under that candidate's own (mean, sd) at that x.

This is refit-from-scratch LOOCV, not "fit once on all 50 and peek held-in".
It is the only thing that can be called an honest out-of-sample test here,
and even it has just 50 folds, most sharing >=49/50 of the data with the
full fit, so it is a weak, high-variance test -- see "What this does not
show" in the report for exactly how weak.

What gets scored
-----------------
1. Per-fold Gaussian log predictive density log N(y_i | mu_i, sd_i), summed
   and averaged over the 50 folds -> the headline LOO score per candidate.
2. Per-fold standardized residual z_i = (y_i - mu_i) / sd_i, used to check
   whether each candidate's uncertainty is *calibrated* (z should look like
   a standard normal if sd(x) is right, not just if mu(x) is right).
3. Paired comparisons between candidates on the per-fold log-density
   differences (paired t-test and Wilcoxon signed-rank), because the three
   candidates share the same 50 folds and the same mean fit per fold, so a
   paired test is the right test, not three independent ones.

Everything is written to two f3dasm records (study_loo_folds, study_loo_cv)
and one pairwise-comparison record (study_pairwise), all stamped
_source_selector, plus figures/selection.png.

Run:
    python selector_loo_cv.py
"""
import numpy as np
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
from scipy import stats as sstats

from f3dasm import ExperimentData, datagenerator
from f3dasm.design import Domain

from car_noise_models import CANDIDATES

RNG_SEED = 0  # only used for the bootstrap CI on the mean LOO score, not for the LOO folds themselves


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

    names = list(CANDIDATES.keys())
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

    # ---------------- paired comparisons on the same 50 folds ----------------
    print("=== Paired comparisons (same folds, same mean fit each fold) ===")
    pairs = [("constant", "powerlaw"), ("constant", "proportional"), ("powerlaw", "proportional")]
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
        print(f"  {b} vs {a}: mean per-fold log-density diff = {mean_diff:+.5f} "
              f"(SEM {sem_diff:.5f}), paired t p={t_p:.4f}, Wilcoxon p={w_p:.4f}")
        pairwise_rows.append(dict(pair=f"{b}_minus_{a}", mean_diff=mean_diff, sem_diff=sem_diff,
                                   t_stat=float(t_stat), t_pvalue=float(t_p),
                                   wilcoxon_stat=float(w_stat) if not np.isnan(w_stat) else float("nan"),
                                   wilcoxon_pvalue=float(w_p)))
    print()

    # ---------------- leverage check: how much does the single worst fold matter? ----------------
    # The fold with the largest |standardized residual| under the constant model is the
    # highest-leverage point for telling the noise models apart (it is where a fixed sd
    # is most badly wrong). Report every candidate's mean LOO score with that one fold
    # removed, so the headline number above is not mistaken for something that isn't
    # resting on 50 independent, equally-informative folds.
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
    for name in names:
        d = results[name]["logpdf"]
        ok = d[~np.isnan(d)]
        idx = rng.integers(0, len(ok), size=(B, len(ok)))
        boot_means = ok[idx].mean(axis=1)
        lo, hi = np.percentile(boot_means, [2.5, 97.5])
        boot_ci[name] = (float(lo), float(hi))
        print(f"  {name}: mean={ok.mean():.5f}, 95% CI=[{lo:.5f}, {hi:.5f}]")
    print()

    # ================= write records =================
    # (1) per-fold record: one row per (candidate, fold)
    fold_domain = Domain()
    fold_domain.add_int("fold", low=0, high=n - 1)
    fold_domain.add_category("candidate", categories=names)
    fold_rows = [{"fold": i, "candidate": name} for name in names for i in range(n)]
    fold_study = ExperimentData(domain=fold_domain, input_data=fold_rows)

    # lookup tables closed over by the datagenerator
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
    fold_study.store("study_loo_folds")
    print("Wrote study_loo_folds/ (50 folds x 3 candidates, per-fold LOO predictions & scores).")

    # (2) summary record: one row per candidate
    summary_domain = Domain()
    summary_domain.add_category("candidate", categories=names)
    summary_rows = [{"candidate": name} for name in names]
    summary_study = ExperimentData(domain=summary_domain, input_data=summary_rows)

    @datagenerator(output_names=["n_folds_ok", "n_folds_failed", "mean_loglik", "total_loglik",
                                  "sem_loglik", "ci95_lo", "ci95_hi", "z_mean", "z_std",
                                  "coverage90", "coverage50", "mean_loglik_excl_worst_fold",
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
        return (int(ok.sum()), int((~ok).sum()), float(d.mean()), float(d.sum()),
                float(d.std(ddof=1) / np.sqrt(len(d))), lo, hi,
                float(zok.mean()), float(zok.std(ddof=1)), cov90, cov50,
                float(d_excl.mean()), "selector")

    summary_study = summarize.call(summary_study, mode="sequential")
    summary_study.store("study_loo_cv")
    print("Wrote study_loo_cv/ (one row per candidate: the comparison table).")

    # (3) pairwise-comparison record
    pair_domain = Domain()
    pair_names = [row["pair"] for row in pairwise_rows]
    pair_domain.add_category("pair", categories=pair_names)
    pair_study = ExperimentData(domain=pair_domain, input_data=[{"pair": p} for p in pair_names])

    @datagenerator(output_names=["mean_diff", "sem_diff", "t_stat", "t_pvalue",
                                  "wilcoxon_stat", "wilcoxon_pvalue", "_source_selector"])
    def pairwise(pair: str):
        row = next(r for r in pairwise_rows if r["pair"] == pair)
        return (row["mean_diff"], row["sem_diff"], row["t_stat"], row["t_pvalue"],
                row["wilcoxon_stat"], row["wilcoxon_pvalue"], "selector")

    pair_study = pairwise.call(pair_study, mode="sequential")
    pair_study.store("study_pairwise")
    print("Wrote study_pairwise/ (paired-test table between candidates on shared folds).")
    print()

    # ================= print the full comparison table =================
    i_df, o_df = summary_study.to_pandas()
    full = i_df.join(o_df)
    print("=== Full comparison table (study_loo_cv) ===")
    print(full.to_string(index=False))
    print()
    ip_df, op_df = pair_study.to_pandas()
    print("=== Pairwise table (study_pairwise) ===")
    print(ip_df.join(op_df).to_string(index=False))
    print()

    best = full.loc[full["mean_loglik"].idxmax(), "candidate"]
    print(f"Highest mean LOO log predictive density: '{best}'")

    # ================= figure =================
    fig, axes = plt.subplots(1, 3, figsize=(15, 4.5))

    ax = axes[0]
    means = [full.loc[full["candidate"] == nm, "mean_loglik"].iloc[0] for nm in names]
    los = [boot_ci[nm][0] for nm in names]
    his = [boot_ci[nm][1] for nm in names]
    yerr = np.array([[m - lo, hi - m] for m, lo, hi in zip(means, los, his)]).T
    ax.bar(names, means, yerr=yerr, capsize=5, color=["#888", "#4c72b0", "#dd8452"])
    ax.set_ylabel("mean LOO log predictive density")
    ax.set_title("LOO-CV score (bootstrap 95% CI over folds)")

    ax = axes[1]
    z_stds = [full.loc[full["candidate"] == nm, "z_std"].iloc[0] for nm in names]
    z_means = [full.loc[full["candidate"] == nm, "z_mean"].iloc[0] for nm in names]
    ax.bar(names, z_stds, color=["#888", "#4c72b0", "#dd8452"])
    ax.axhline(1.0, color="k", linestyle="--", linewidth=1, label="ideal (std=1)")
    ax.set_ylabel("std of LOO standardized residual z")
    ax.set_title("Calibration: sd(x) vs actual spread")
    ax.legend()

    ax = axes[2]
    colors = {"constant": "#888", "powerlaw": "#4c72b0", "proportional": "#dd8452"}
    for name in names:
        r = results[name]
        ok = ~np.isnan(r["z"])
        ax.scatter(x[ok], r["z"][ok], s=18, alpha=0.6, label=name, color=colors[name])
    ax.axhline(0, color="k", linewidth=0.8)
    ax.axhline(1.645, color="k", linestyle=":", linewidth=0.8)
    ax.axhline(-1.645, color="k", linestyle=":", linewidth=0.8)
    ax.set_xlabel("x (velocity, held-out)")
    ax.set_ylabel("LOO standardized residual z")
    ax.set_title("z vs velocity (should show no trend if sd(x) is right)")
    ax.legend(fontsize=8)

    fig.suptitle("Selector: leave-one-out comparison of noise-model candidates (_source_selector)")
    fig.tight_layout()
    fig.savefig("figures/selection.png", dpi=150)
    print("Wrote figures/selection.png")


if __name__ == "__main__":
    main()
