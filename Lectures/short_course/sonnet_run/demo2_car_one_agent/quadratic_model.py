"""Quadratic least-squares mean model with a single constant noise sd.

Fits y = a*x^2 + b*x + c to the stopping-distance record by ordinary least
squares, estimates one constant noise standard deviation from the residuals,
writes the predictions into the f3dasm record, and plots the fit.

Run: python quadratic_model.py
"""
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np
from f3dasm import ExperimentData, datagenerator

SOURCE = "quadratic_ls"
N_PARAMS = 3  # a, b, c


def main():
    data = ExperimentData.from_file('data')
    input_df, output_df = data.to_pandas()
    x = input_df['x'].to_numpy(float)
    y = output_df['y'].to_numpy(float)
    n = len(x)

    coeffs = np.polyfit(x, y, 2)  # [a, b, c] for a*x^2 + b*x + c
    mu = np.polyval(coeffs, x)
    resid = y - mu
    sigma = float(np.sqrt(np.sum(resid ** 2) / (n - N_PARAMS)))

    a, b, c = coeffs
    print(f"Fitted quadratic mean: y = {a:.6g}*x^2 + {b:.6g}*x + {c:.6g}")
    print(f"Constant noise std (sigma, {n - N_PARAMS} dof): {sigma:.6g}")

    @datagenerator(output_names=['y_pred_quad', 'sd_quad', '_source_quad'])
    def predict(x: float):
        mu_i = float(np.polyval(coeffs, x))
        return mu_i, sigma, SOURCE

    data = data.mark_all('open')
    data = predict.call(data, mode='sequential')
    data.store('data')

    # Reload from the record so the plot is provably reproducible from disk.
    data = ExperimentData.from_file('data')
    input_df, output_df = data.to_pandas()
    x = input_df['x'].to_numpy(float)
    y = output_df['y'].to_numpy(float)
    sigma = float(output_df['sd_quad'].to_numpy(float)[0])

    xs_line = np.linspace(x.min(), x.max(), 200)
    mu_line = np.polyval(coeffs, xs_line)

    fig, ax = plt.subplots(figsize=(7, 5))
    ax.scatter(x, y, color='tab:blue', s=25, label='measurements', zorder=3)
    ax.plot(xs_line, mu_line, color='tab:red', label='quadratic mean fit', zorder=2)
    ax.fill_between(xs_line, mu_line - sigma, mu_line + sigma,
                     color='tab:red', alpha=0.2, label='mean ± 1 sd (constant)')
    ax.fill_between(xs_line, mu_line - 2 * sigma, mu_line + 2 * sigma,
                     color='tab:red', alpha=0.1, label='mean ± 2 sd (constant)')
    ax.set_xlabel('velocity x (m/s)')
    ax.set_ylabel('stopping distance y (m)')
    ax.set_title('Quadratic mean, constant-sd noise model')
    ax.legend()
    fig.tight_layout()
    fig.savefig('quadratic_fit.png', dpi=150)
    print("Saved plot to quadratic_fit.png")


if __name__ == '__main__':
    main()
