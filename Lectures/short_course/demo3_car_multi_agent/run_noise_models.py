"""Fit every velocity-dependent-noise candidate, store one, plot it.

    python run_noise_models.py

Reads the record in ``data/``, fits all the candidates in ``car_noise``
(including the unchanged ``car_ml`` baseline) on the record's own 50
measurements, prints what each fit found, writes the predictions of the
*provisionally chosen* candidate back into the record as

    y_pred_phys, sd_phys, _source_phys

(leaving ``y_pred_quad``, ``sd_quad``, ``_source_quad`` untouched), and draws
``figures/model.png`` from the record read back off disk.

Nothing here is a verdict.  In-sample log-likelihood, AIC and BIC are printed
as descriptions of the fits; ranking the candidates on data they have not seen
belongs to the selector agent, not to this script.

Safe to run twice: re-using a column name overwrites that column.
"""

import matplotlib
matplotlib.use('Agg')

import os

import matplotlib.pyplot as plt
import numpy as np

from f3dasm import ExperimentData, datagenerator

import car_noise

# the candidate whose predictions go into the record (provisional -- see report)
CHOSEN = 'phys'
WRITER = 'car_noise.PhysTwoTermNoise'
BASE = 'quad'                 # the baseline's existing column suffix

BLUE = '#2a78d6'
ORANGE = '#eb6834'
GREEN = '#3f9c5a'
INK = '#0b0b0b'
INK_SOFT = '#52514e'
GRID = '#d8d7d2'

FIGDIR = 'figures'
FIGPATH = os.path.join(FIGDIR, 'model.png')

G = 9.81                      # m/s^2, only used to translate b2 into friction


def rule(title=''):
    print('\n' + '-' * 78)
    if title:
        print(title)
        print('-' * 78)


def main():
    # ---- the data ----------------------------------------------------------
    data = ExperimentData.from_file('data')
    input_df, output_df = data.to_pandas()
    x = input_df['x'].to_numpy(float)
    y = output_df['y'].to_numpy(float)
    print(f'record: n = {len(x)} measurements, '
          f'x in [{x.min():.4g}, {x.max():.4g}] m/s, '
          f'y in [{y.min():.4g}, {y.max():.4g}] m')
    print(f'columns already in the record: {list(output_df.columns)}')

    # ---- what the residuals of the baseline actually show ------------------
    rule('why a velocity-dependent sd: the baseline residuals')
    base = car_noise.make('quad_ols').fit(x, y)
    r = y - base.mean(x)
    print(f'baseline (car_ml) sd = {base.params["s"]:.6g} m, one number for all x')
    edges = [0, 20, 40, 60, 100]
    for lo, hi in zip(edges[:-1], edges[1:]):
        m = (x >= lo) & (x < hi)
        print(f'  x in [{lo:3d},{hi:3d}): n = {m.sum():2d}   '
              f'sd(residual) = {np.std(r[m], ddof=1):8.4f} m   '
              f'max|residual| = {np.max(np.abs(r[m])):8.4f} m')
    print(f'corr( |residual| , x )      = {np.corrcoef(np.abs(r), x)[0, 1]:+.6f}')
    print(f'corr( log|residual| , log x ) = '
          f'{np.corrcoef(np.log(np.abs(r)), np.log(x))[0, 1]:+.6f}')
    print(f'baseline band at x = {x.min():.4g}: '
          f'mu - 2sd = {base.mean(x.min()) - 2 * base.std(x.min()):+.4f} m '
          f'(negative stopping distance)')

    # ---- fit every candidate ----------------------------------------------
    rule('joint-MLE fits of every candidate (all 50 measurements, in-sample)')
    models = {}
    rows = []
    for name in car_noise.CANDIDATES:
        m = car_noise.make(name).fit(x, y)
        models[name] = m
        ic = car_noise.information_criteria(m, x, y)
        z = m.standardized_residuals(x, y)
        rows.append((name, ic, z, m))
        conv = getattr(m, 'converged', None)
        spread = getattr(m, 'start_spread', None)
        print(f'\n{name:9s} sd(x): {m.description}')
        print('          ' + ', '.join(f'{k} = {m.params[k]:+.6g}'
                                       for k in m.param_names))
        print(f'          loglik = {ic["loglik"]:+.4f}   k = {ic["n_params"]}   '
              f'AIC = {ic["aic"]:.4f}   BIC = {ic["bic"]:.4f}')
        if conv is not None:
            print(f'          optimiser: converged = {conv}, '
                  f'{m.n_starts} starts, best-worst NLL spread = {spread:.3e}')
        else:
            print('          optimiser: none (closed-form OLS, car_ml untouched)')
        print(f'          standardized residuals z = (y-mu)/sd:  '
              f'mean = {z.mean():+.4f}, sd = {z.std(ddof=0):.4f}, '
              f'max|z| = {np.max(np.abs(z)):.4f}')
        print(f'          corr(|z|, x) = {np.corrcoef(np.abs(z), x)[0, 1]:+.4f}   '
              f'frac |z| <= 2: {np.mean(np.abs(z) <= 2):.2f}   '
              f'mu-2sd at x={x.min():.4g}: '
              f'{m.mean(x.min()) - 2 * m.std(x.min()):+.4f} m')

    rule('one table, sorted by the name only (no ranking is implied here)')
    print(f'{"name":10s} {"k":>2s} {"loglik":>12s} {"AIC":>10s} {"BIC":>10s} '
          f'{"corr(|z|,x)":>12s}')
    for name, ic, z, m in rows:
        print(f'{name:10s} {ic["n_params"]:2d} {ic["loglik"]:12.4f} '
              f'{ic["aic"]:10.4f} {ic["bic"]:10.4f} '
              f'{np.corrcoef(np.abs(z), x)[0, 1]:12.4f}')

    # ---- the provisional choice, in detail --------------------------------
    chosen = models[CHOSEN]
    rule(f'provisional choice written into the record: {CHOSEN!r} '
         f'({WRITER})')
    print(f'mu(x)  = {chosen.params["b1"]:+.6g}*x {chosen.params["b2"]:+.6g}*x^2')
    print(f'sd(x)  = sqrt( ({chosen.params["a"]:.6g}*x)^2 '
          f'+ ({chosen.params["c"]:.6g}*x^2)^2 )')
    se = chosen.standard_errors(x, y)
    if se:
        for k in chosen.param_names:
            print(f'   {k:>3s} = {chosen.params[k]:+.6g}  '
                  f'+/- {se[k]:.6g}  (asymptotic MLE se)')
    else:
        print('   standard errors unavailable: numerical Hessian not '
              'positive definite')
    print(f'   NLL at optimum = {chosen.nll:.6f}   '
          f'loglik = {chosen.loglik(x, y):+.6f}')
    print('physical reading of the four parameters:')
    print(f'   reaction time   t_r  = b1        = {chosen.params["b1"]:.4f} s')
    print(f'   spread of t_r        = a         = {chosen.params["a"]:.4f} s')
    print(f'   braking constant     = b2        = {chosen.params["b2"]:.6f} s^2/m'
          f'  ->  friction mu = 1/(2*g*b2) = '
          f'{1.0 / (2 * G * chosen.params["b2"]):.4f}')
    print(f'   spread of that       = c         = {chosen.params["c"]:.6f} s^2/m'
          f'  ->  relative spread c/b2 = '
          f'{chosen.params["c"] / chosen.params["b2"]:.4f}')
    print(f'   sd at x = {x.min():.4g}: {chosen.std(x.min()):.4f} m ;  '
          f'sd at x = {x.max():.4g}: {chosen.std(x.max()):.4f} m  '
          f'(baseline: {base.params["s"]:.4f} m at both)')

    # ---- write it into the record -----------------------------------------
    name = CHOSEN
    model = chosen

    @datagenerator(output_names=[f'y_pred_{name}', f'sd_{name}',
                                 f'_source_{name}'])
    def predict(x: float):
        return float(model.mean(float(x))), float(model.std(float(x))), WRITER

    data = data.mark_all('open')
    data = predict.call(data, mode='sequential')
    data.store('data')

    # ---- read the record back and check what landed -----------------------
    rule('the record, read back from disk')
    stored = ExperimentData.from_file('data')
    in_df, out_df = stored.to_pandas()
    xs = in_df['x'].to_numpy(float)
    ys = out_df['y'].to_numpy(float)
    mu_new = out_df[f'y_pred_{name}'].to_numpy(float)
    sd_new = out_df[f'sd_{name}'].to_numpy(float)
    mu_base = out_df[f'y_pred_{BASE}'].to_numpy(float)
    sd_base = out_df[f'sd_{BASE}'].to_numpy(float)

    print(f'columns now: {list(out_df.columns)}')
    print(f'wrote: y_pred_{name}, sd_{name}, _source_{name}')
    print(f'_source_{name} = {sorted(set(out_df[f"_source_{name}"]))}')
    print(f'_source_{BASE} = {sorted(set(out_df[f"_source_{BASE}"]))} '
          f'(baseline kept, not overwritten)')
    print(f'sd_{name}: min = {sd_new.min():.4f}, max = {sd_new.max():.4f} '
          f'(varies with x); sd_{BASE}: min = {sd_base.min():.4f}, '
          f'max = {sd_base.max():.4f}')
    print(f'stored columns match the fitted model: '
          f'{bool(np.allclose(mu_new, model.mean(xs)) and np.allclose(sd_new, model.std(xs)))}')
    z_new = (ys - mu_new) / sd_new
    z_base = (ys - mu_base) / sd_base
    print(f'from the record: frac |z| <= 2 is {np.mean(np.abs(z_new) <= 2):.2f} '
          f'for {name} and {np.mean(np.abs(z_base) <= 2):.2f} for {BASE} '
          f'(a Gaussian would give 0.95)')

    # ---- figure, drawn from the record ------------------------------------
    grid = np.linspace(x.min(), x.max(), 400)
    mu_g = model.mean(grid)
    sd_g = model.std(grid)
    sd_gb = base.std(grid)
    mu_gb = base.mean(grid)

    fig, (ax, axs, axz) = plt.subplots(
        3, 1, figsize=(7.4, 9.4), sharex=True,
        gridspec_kw={'height_ratios': [2.4, 1.15, 1.15], 'hspace': 0.13})
    fig.patch.set_facecolor('white')
    for a in (ax, axs, axz):
        a.set_facecolor('white')
        a.grid(True, color=GRID, linewidth=0.6, alpha=0.9)
        a.set_axisbelow(True)
        for side in ('top', 'right'):
            a.spines[side].set_visible(False)
        for side in ('left', 'bottom'):
            a.spines[side].set_color(GRID)
        a.tick_params(colors=INK_SOFT, labelsize=9, length=3)

    ax.fill_between(grid, mu_g - 2 * sd_g, mu_g + 2 * sd_g, color=ORANGE,
                    alpha=0.18, linewidth=0,
                    label=r'chosen  $\mu(x)\pm2\,\mathrm{sd}(x)$')
    ax.plot(grid, mu_gb - 2 * sd_gb, color=GREEN, linewidth=1.1, linestyle='--',
            label=r'baseline  $\mu(x)\pm2\,\mathrm{sd}$ (constant)')
    ax.plot(grid, mu_gb + 2 * sd_gb, color=GREEN, linewidth=1.1, linestyle='--')
    ax.plot(grid, mu_g, color=ORANGE, linewidth=2, label=r'chosen mean $\mu(x)$')
    ax.axhline(0, color=INK_SOFT, linewidth=0.8, alpha=0.5)
    ax.scatter(xs, ys, s=42, color=BLUE, edgecolor='white', linewidth=1.0,
               zorder=3, label='50 measurements (from the record)')
    ax.set_ylabel('stopping distance  y  [m]', color=INK, fontsize=10)
    ax.set_title('Car stopping distance: noise that grows with velocity',
                 color=INK, fontsize=12, loc='left', pad=10)
    ax.text(0.03, 0.95,
            f'$\\mu(x) = {model.params["b1"]:+.4g}\\,x '
            f'{model.params["b2"]:+.4g}\\,x^2$\n'
            f'$\\mathrm{{sd}}(x) = \\sqrt{{({model.params["a"]:.3g}x)^2'
            f'+({model.params["c"]:.3g}x^2)^2}}$\n'
            f'$\\log L = {model.loglik(x, y):.2f}$,  '
            f'$k = {model.n_params}$,  $n = {len(xs)}$\n'
            f'(provisional: not yet judged by the selector)',
            transform=ax.transAxes, va='top', ha='left',
            color=INK_SOFT, fontsize=9,
            bbox=dict(facecolor='white', edgecolor=GRID,
                      boxstyle='round,pad=0.5'))
    ax.legend(loc='lower right', frameon=False, fontsize=9, labelcolor=INK_SOFT)

    # panel 2: the sd columns themselves, straight out of the record
    axs.scatter(xs, np.abs(ys - mu_new), s=26, color=BLUE, edgecolor='white',
                linewidth=0.8, zorder=3, label=r'$|y-\mu(x)|$ (record)')
    axs.plot(xs, sd_new, color=ORANGE, linewidth=2, marker='o', markersize=3,
             label=f'sd_{name} column (record)')
    axs.plot(xs, sd_base, color=GREEN, linewidth=1.4, linestyle='--',
             label=f'sd_{BASE} column (record)')
    axs.set_ylabel('spread  [m]', color=INK, fontsize=10)
    axs.set_title('the noise column: constant vs growing with x',
                  color=INK_SOFT, fontsize=9, loc='left', pad=6)
    axs.legend(loc='upper left', frameon=False, fontsize=9, labelcolor=INK_SOFT)

    # panel 3: standardized residuals of both models, from the record
    axz.axhline(0, color=INK_SOFT, linewidth=0.9)
    for lvl in (-2, 2):
        axz.axhline(lvl, color=INK_SOFT, linewidth=0.8, linestyle=':')
    axz.scatter(xs, z_base, s=28, color=GREEN, marker='s', edgecolor='white',
                linewidth=0.7, zorder=3, label=f'{BASE} (constant sd)')
    axz.scatter(xs, z_new, s=34, color=ORANGE, edgecolor='white',
                linewidth=0.8, zorder=4, label=f'{name} (sd(x))')
    axz.set_xlabel('velocity when the obstacle is seen  x  [m/s]',
                   color=INK, fontsize=10)
    axz.set_ylabel('standardized\n$(y-\\mu)/\\mathrm{sd}$', color=INK,
                   fontsize=10)
    axz.set_title('standardized residuals: a flat spread is what the noise '
                  'model is for', color=INK_SOFT, fontsize=9, loc='left', pad=6)
    axz.legend(loc='upper left', frameon=False, fontsize=9, labelcolor=INK_SOFT)

    os.makedirs(FIGDIR, exist_ok=True)
    fig.savefig(FIGPATH, dpi=200, bbox_inches='tight', facecolor='white')
    plt.close(fig)
    print(f'\nwrote {FIGPATH}')


if __name__ == '__main__':
    main()
