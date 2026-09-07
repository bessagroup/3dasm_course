"""Fit the quadratic-mean / constant-noise model, store it, plot it.

    python run_quadratic.py

Reads the record in ``data/``, fits ``car_ml.QuadraticConstantNoise`` on it,
writes ``y_pred_quad``, ``sd_quad`` and ``_source_quad`` back into the record,
and draws ``figures/quadratic_constant_noise.png``.

Safe to run twice: re-using a column name overwrites that column.
"""

import matplotlib
matplotlib.use('Agg')

import os

import matplotlib.pyplot as plt
import numpy as np

from f3dasm import ExperimentData

from car_ml import NAME, FitQuadraticConstantNoise

# reference-palette slots 1 and 2 (all-pairs validated, light + dark)
BLUE = '#2a78d6'     # the 50 measurements
ORANGE = '#eb6834'   # the model
INK = '#0b0b0b'
INK_SOFT = '#52514e'
GRID = '#d8d7d2'

FIGDIR = 'figures'
FIGPATH = os.path.join(FIGDIR, 'quadratic_constant_noise.png')


def main():
    # ---- fit and write -----------------------------------------------------
    data = ExperimentData.from_file('data')

    block = FitQuadraticConstantNoise(name=NAME)
    block.arm(data)                 # fits; prints the coefficients and sd
    data = block.call(data)         # writes the prediction columns
    data.store('data')

    model = block.model

    # ---- read everything back out of the record ----------------------------
    # the figure is drawn from the stored record, not from local variables,
    # so it can only show what was actually written
    stored = ExperimentData.from_file('data')
    input_df, output_df = stored.to_pandas()
    x = input_df['x'].to_numpy(float)
    y = output_df['y'].to_numpy(float)
    mu = output_df[f'y_pred_{NAME}'].to_numpy(float)
    sd = output_df[f'sd_{NAME}'].to_numpy(float)

    print(f'\nwrote columns: y_pred_{NAME}, sd_{NAME}, _source_{NAME}')
    print(f'_source_{NAME} = '
          f'{sorted(set(output_df[f"_source_{NAME}"]))}')
    print(f'columns now in the record: {list(output_df.columns)}')

    resid = y - mu
    sd_const = float(sd[0])
    print(f'\nsd column is constant: {bool(np.allclose(sd, sd_const))} '
          f'(value {sd_const:.6g})')
    print(f'residuals: mean {resid.mean():+.6g}, '
          f'min {resid.min():+.6g}, max {resid.max():+.6g}')
    inside = float(np.mean(np.abs(resid) <= 2 * sd_const))
    print(f'fraction of points inside mu +/- 2sd: {inside:.2f} '
          f'(Gaussian would give 0.95)')

    # ---- plot --------------------------------------------------------------
    grid = np.linspace(x.min(), x.max(), 400)
    mu_grid = model.mean(grid)
    sd_grid = model.std(grid)

    fig, (ax, axr) = plt.subplots(
        2, 1, figsize=(7.2, 6.6), sharex=True,
        gridspec_kw={'height_ratios': [2.6, 1.0], 'hspace': 0.12})
    fig.patch.set_facecolor('white')

    for a in (ax, axr):
        a.set_facecolor('white')
        a.grid(True, color=GRID, linewidth=0.6, alpha=0.9)
        a.set_axisbelow(True)
        for side in ('top', 'right'):
            a.spines[side].set_visible(False)
        for side in ('left', 'bottom'):
            a.spines[side].set_color(GRID)
        a.tick_params(colors=INK_SOFT, labelsize=9, length=3)

    # main panel: measurements, fitted mean, constant-width noise band
    ax.fill_between(grid, mu_grid - 2 * sd_grid, mu_grid + 2 * sd_grid,
                    color=ORANGE, alpha=0.16, linewidth=0,
                    label=r'model  $\mu(x)\pm 2\,\mathrm{sd}$')
    ax.plot(grid, mu_grid, color=ORANGE, linewidth=2,
            label=r'model mean  $\mu(x)$, quadratic')
    ax.scatter(x, y, s=42, color=BLUE, edgecolor='white', linewidth=1.0,
               zorder=3, label='50 measurements')

    ax.set_ylabel('stopping distance  y  [m]', color=INK, fontsize=10)
    ax.set_title('Car stopping distance: quadratic mean, one constant noise sd',
                 color=INK, fontsize=12, loc='left', pad=10)

    b0, b1, b2 = model.beta
    ax.text(0.03, 0.95,
            f'$\\mu(x) = {b0:+.3g} {b1:+.3g}\\,x {b2:+.3g}\\,x^2$\n'
            f'$\\mathrm{{sd}} = {model.sd:.3g}$ m  (constant)\n'
            f'$R^2 = {model.r2:.4f}$,  $n = {model.n}$',
            transform=ax.transAxes, va='top', ha='left',
            color=INK_SOFT, fontsize=9,
            bbox=dict(facecolor='white', edgecolor=GRID, boxstyle='round,pad=0.5'))
    ax.legend(loc='lower right', frameon=False, fontsize=9,
              labelcolor=INK_SOFT)

    # residual panel: the constant-sd assumption, held up to the light
    axr.axhline(0, color=ORANGE, linewidth=2)
    axr.axhline(2 * sd_const, color=ORANGE, linewidth=1.2, linestyle='--')
    axr.axhline(-2 * sd_const, color=ORANGE, linewidth=1.2, linestyle='--')
    axr.scatter(x, resid, s=30, color=BLUE, edgecolor='white', linewidth=0.9,
                zorder=3)
    axr.text(x.max(), 2 * sd_const, '  $+2\\,\\mathrm{sd}$', color=ORANGE,
             fontsize=9, va='center', ha='left', clip_on=False)
    axr.text(x.max(), -2 * sd_const, '  $-2\\,\\mathrm{sd}$', color=ORANGE,
             fontsize=9, va='center', ha='left', clip_on=False)
    axr.set_xlabel('velocity when the obstacle is seen  x  [m/s]',
                   color=INK, fontsize=10)
    axr.set_ylabel('residual\n$y-\\mu(x)$  [m]', color=INK, fontsize=10)
    axr.set_title('residuals: a constant band assumes the scatter does not '
                  'grow with x', color=INK_SOFT, fontsize=9, loc='left', pad=6)

    fig.subplots_adjust(right=0.90)
    os.makedirs(FIGDIR, exist_ok=True)
    fig.savefig(FIGPATH, dpi=200, bbox_inches='tight', facecolor='white')
    plt.close(fig)
    print(f'\nwrote {FIGPATH}')


if __name__ == '__main__':
    main()
