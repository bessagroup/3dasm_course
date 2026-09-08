"""Heteroscedastic noise fix for the quadratic stopping-distance model.

quadratic_model.py fits y = a*x^2 + b*x + c by OLS and assumes ONE constant
noise sd for every velocity. Residuals from that fit show scatter growing with
x (small at low velocity, large at high velocity), so a constant band is too
wide where the car is slow and too narrow where it is fast.

This script keeps the same quadratic mean (that part was never in question)
and replaces the constant noise sd with sigma(x) = exp(p0 + p1*x), fit by
maximum likelihood on the OLS residuals. sigma(x) is positive by construction
and grows/shrinks smoothly with velocity instead of being one fixed number.

To show the fix is actually better -- not just a different assumption -- this
script runs leave-one-out cross-validation: for each of the 50 stops, both
models (constant-sd and heteroscedastic-sd) are refit on the OTHER 49 stops
and scored on the held-out one by Gaussian negative log-likelihood (NLL,
lower is better). No stop ever contributes to its own score.

Writes to the f3dasm record in `data/`:
  - y_pred_quad, sd_quad, _source_quad        (full-data constant-sd model, as before)
  - y_pred_hetero, sd_hetero, _source_hetero  (full-data heteroscedastic model)
  - sd_const_loo, sd_hetero_loo,
    nll_const_loo, nll_hetero_loo, _source_loocv   (per-stop held-out scores)

Run: python heteroscedastic_model.py
"""
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np
from scipy.optimize import minimize
from scipy.stats import wilcoxon
from f3dasm import ExperimentData, datagenerator

SOURCE_QUAD = "quadratic_ls"
SOURCE_HETERO = "heteroscedastic_mle"
SOURCE_LOOCV = "loocv_const_vs_hetero"
N_MEAN_PARAMS = 3  # a, b, c

COLOR_DATA = "#1f77b4"
COLOR_CONST = "#ff7f0e"
COLOR_HETERO = "#9467bd"


def fit_quadratic_mean(x, y):
    coeffs = np.polyfit(x, y, 2)
    mu = np.polyval(coeffs, x)
    return coeffs, mu


def fit_constant_sigma(resid, n_params):
    n = len(resid)
    return float(np.sqrt(np.sum(resid ** 2) / (n - n_params)))


def hetero_nll(params, x, resid):
    p0, p1 = params
    log_sigma = p0 + p1 * x
    sigma = np.exp(log_sigma)
    return float(np.sum(log_sigma + 0.5 * np.log(2 * np.pi) +
                         resid ** 2 / (2 * sigma ** 2)))


def fit_hetero_sigma(x, resid):
    p0_init = float(np.log(max(np.std(resid), 1e-6)))
    res = minimize(hetero_nll, x0=[p0_init, 0.0], args=(x, resid), method="BFGS")
    return res.x  # p0, p1


def gaussian_nll(y_obs, mu, sigma):
    return 0.5 * np.log(2 * np.pi * sigma ** 2) + (y_obs - mu) ** 2 / (2 * sigma ** 2)


def main():
    data = ExperimentData.from_file('data')
    input_df, output_df = data.to_pandas()
    x = input_df['x'].to_numpy(float)
    y = output_df['y'].to_numpy(float)
    n = len(x)

    # ---- full-data fits (for the record and the plot) -------------------
    coeffs, mu_full = fit_quadratic_mean(x, y)
    resid_full = y - mu_full
    sigma_const_full = fit_constant_sigma(resid_full, N_MEAN_PARAMS)
    p0_full, p1_full = fit_hetero_sigma(x, resid_full)
    sigma_hetero_full = np.exp(p0_full + p1_full * x)

    a, b, c = coeffs
    print(f"Quadratic mean (unchanged): y = {a:.6g}*x^2 + {b:.6g}*x + {c:.6g}")
    print(f"Constant noise sd (full data, {n - N_MEAN_PARAMS} dof): {sigma_const_full:.6g}")
    print(f"Heteroscedastic noise sd(x) = exp({p0_full:.6g} + {p1_full:.6g}*x) (full data)")
    print(f"  -> sd at x={x.min():.1f}: {np.exp(p0_full + p1_full * x.min()):.4g}, "
          f"sd at x={x.max():.1f}: {np.exp(p0_full + p1_full * x.max()):.4g}")

    # ---- leave-one-out cross-validation ---------------------------------
    sd_const_loo = np.empty(n)
    sd_hetero_loo = np.empty(n)
    nll_const_loo = np.empty(n)
    nll_hetero_loo = np.empty(n)

    for i in range(n):
        train = np.arange(n) != i
        x_tr, y_tr = x[train], y[train]
        x_ho, y_ho = x[i], y[i]

        coeffs_tr, mu_tr = fit_quadratic_mean(x_tr, y_tr)
        resid_tr = y_tr - mu_tr
        mu_ho = float(np.polyval(coeffs_tr, x_ho))

        sigma_const_tr = fit_constant_sigma(resid_tr, N_MEAN_PARAMS)
        p0_tr, p1_tr = fit_hetero_sigma(x_tr, resid_tr)
        sigma_hetero_ho = float(np.exp(p0_tr + p1_tr * x_ho))

        sd_const_loo[i] = sigma_const_tr
        sd_hetero_loo[i] = sigma_hetero_ho
        nll_const_loo[i] = gaussian_nll(y_ho, mu_ho, sigma_const_tr)
        nll_hetero_loo[i] = gaussian_nll(y_ho, mu_ho, sigma_hetero_ho)

    mean_nll_const = float(np.mean(nll_const_loo))
    mean_nll_hetero = float(np.mean(nll_hetero_loo))
    n_hetero_wins = int(np.sum(nll_hetero_loo < nll_const_loo))

    median_x = float(np.median(x))
    low = x < median_x
    high = ~low
    mean_nll_const_low = float(np.mean(nll_const_loo[low]))
    mean_nll_hetero_low = float(np.mean(nll_hetero_loo[low]))
    mean_nll_const_high = float(np.mean(nll_const_loo[high]))
    mean_nll_hetero_high = float(np.mean(nll_hetero_loo[high]))

    stat, p_value = wilcoxon(nll_const_loo, nll_hetero_loo, alternative='greater')

    print("\nLeave-one-out cross-validation (Gaussian NLL on held-out stop, lower = better):")
    print(f"  mean NLL, constant-sd model:       {mean_nll_const:.4f}")
    print(f"  mean NLL, heteroscedastic model:   {mean_nll_hetero:.4f}")
    print(f"  heteroscedastic model wins on {n_hetero_wins}/{n} held-out stops")
    print(f"  low-velocity half  (x < {median_x:.1f}): const {mean_nll_const_low:.4f} vs hetero {mean_nll_hetero_low:.4f}")
    print(f"  high-velocity half (x >= {median_x:.1f}): const {mean_nll_const_high:.4f} vs hetero {mean_nll_hetero_high:.4f}")
    print(f"  Wilcoxon signed-rank (H1: const NLL > hetero NLL): stat={stat:.4f}, p={p_value:.3g}")

    # ---- write everything into the record --------------------------------
    hetero_lookup = {
        float(xi): (float(mu_full[j]), float(sigma_hetero_full[j]))
        for j, xi in enumerate(x)
    }
    loo_lookup = {
        float(xi): (float(sd_const_loo[j]), float(sd_hetero_loo[j]),
                    float(nll_const_loo[j]), float(nll_hetero_loo[j]))
        for j, xi in enumerate(x)
    }

    @datagenerator(output_names=['y_pred_quad', 'sd_quad', '_source_quad'])
    def predict_quad(x: float):
        mu_i = float(np.polyval(coeffs, x))
        return mu_i, sigma_const_full, SOURCE_QUAD

    @datagenerator(output_names=['y_pred_hetero', 'sd_hetero', '_source_hetero'])
    def predict_hetero(x: float):
        mu_i, sigma_i = hetero_lookup[float(x)]
        return mu_i, sigma_i, SOURCE_HETERO

    @datagenerator(output_names=['sd_const_loo', 'sd_hetero_loo',
                                  'nll_const_loo', 'nll_hetero_loo', '_source_loocv'])
    def attach_loocv(x: float):
        sd_c, sd_h, nll_c, nll_h = loo_lookup[float(x)]
        return sd_c, sd_h, nll_c, nll_h, SOURCE_LOOCV

    data = data.mark_all('open')
    data = predict_quad.call(data, mode='sequential')
    data = data.mark_all('open')
    data = predict_hetero.call(data, mode='sequential')
    data = data.mark_all('open')
    data = attach_loocv.call(data, mode='sequential')
    data.store('data')

    # ---- reload from the record so the plots are provably reproducible ---
    data = ExperimentData.from_file('data')
    input_df, output_df = data.to_pandas()
    x = input_df['x'].to_numpy(float)
    y = output_df['y'].to_numpy(float)
    sigma_const_full = float(output_df['sd_quad'].to_numpy(float)[0])
    sd_hetero_full = output_df['sd_hetero'].to_numpy(float)
    y_pred_quad = output_df['y_pred_quad'].to_numpy(float)
    nll_const_loo = output_df['nll_const_loo'].to_numpy(float)
    nll_hetero_loo = output_df['nll_hetero_loo'].to_numpy(float)

    order = np.argsort(x)
    x_sorted = x[order]
    mu_sorted = y_pred_quad[order]
    sd_hetero_sorted = sd_hetero_full[order]

    # Fig 1: constant band vs heteroscedastic band over the data
    fig, ax = plt.subplots(figsize=(7.5, 5))
    ax.scatter(x, y, color=COLOR_DATA, s=25, label='measurements', zorder=4)
    ax.plot(x_sorted, mu_sorted, color='0.3', lw=2, label='quadratic mean fit', zorder=3)
    ax.fill_between(x_sorted, mu_sorted - sigma_const_full, mu_sorted + sigma_const_full,
                     color=COLOR_CONST, alpha=0.25, label='mean ± 1 sd (constant)', zorder=1)
    ax.fill_between(x_sorted, mu_sorted - sd_hetero_sorted, mu_sorted + sd_hetero_sorted,
                     color=COLOR_HETERO, alpha=0.35, label='mean ± 1 sd (heteroscedastic)', zorder=2)
    ax.set_xlabel('velocity x (m/s)')
    ax.set_ylabel('stopping distance y (m)')
    ax.set_title('Constant-sd vs heteroscedastic-sd noise bands')
    ax.legend()
    fig.tight_layout()
    fig.savefig('heteroscedastic_fit.png', dpi=150)
    print("\nSaved plot to heteroscedastic_fit.png")

    # Fig 2: per-stop held-out NLL, constant vs heteroscedastic
    fig, ax = plt.subplots(figsize=(7.5, 5))
    ax.scatter(x, nll_const_loo, color=COLOR_CONST, s=28, label='constant-sd model', zorder=3)
    ax.scatter(x, nll_hetero_loo, color=COLOR_HETERO, s=28, label='heteroscedastic model', zorder=3,
               marker='^')
    for xi, nc, nh in zip(x, nll_const_loo, nll_hetero_loo):
        ax.plot([xi, xi], [nc, nh], color='0.7', lw=0.8, zorder=1)
    ax.axhline(0, color='0.85', lw=1, zorder=0)
    ax.set_xlabel('velocity x (m/s)')
    ax.set_ylabel('held-out negative log-likelihood (lower = better)')
    ax.set_title('Leave-one-out score per stop: does the fix generalize?')
    ax.legend()
    fig.tight_layout()
    fig.savefig('loocv_comparison.png', dpi=150)
    print("Saved plot to loocv_comparison.png")


if __name__ == '__main__':
    main()
