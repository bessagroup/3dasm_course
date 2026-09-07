"""Replace the constant-noise assumption, and test the replacement on stops
the model never saw.

    python run_noise_study.py

What it does, in order:

1. Scores four noise forms -- constant, proportional, affine, power -- each
   with the same quadratic mean and the same maximum-likelihood fitter, so the
   only thing that differs between them is sd(x). The scoring is *out of
   sample*: leave-one-out (train on 49 stops, predict the 50th) and repeated
   5-fold (train on 40, predict 10) . The score is the held-out log predictive
   density, which is what a density model is actually for; held-out RMSE and
   the coverage of the +/-2sd interval are reported beside it.
   The candidate table is an f3dasm record of its own, ``study_noise/``.
2. Writes into ``data/``: the winning model's fit, and -- for the winner and
   for the constant-noise baseline -- the leave-one-out prediction of every
   stop, so the comparison can be recomputed from the record alone.
3. Draws two figures from the stored columns.

Safe to run twice: every column it writes is overwritten in place.
"""

import matplotlib
matplotlib.use('Agg')

import os

import matplotlib.pyplot as plt
import numpy as np
from scipy.stats import norm, wilcoxon

from f3dasm import ExperimentData, datagenerator
from f3dasm.design import Domain

from car_noise import (DEGREE, NOISE_FORMS, FitNoiseModel, GaussianPolyModel,
                       WriteLeaveOneOut, kfold_log_density, leave_one_out)

# categorical slots 1-3 of the reference palette (all-pairs validated, both
# modes); slot 1 stays on the measurements, as in the first figure
BLUE = '#2a78d6'      # the 50 measurements
ORANGE = '#eb6834'    # the old model: constant sd
AQUA = '#1baf7a'      # the new model: sd grows with x
INK = '#0b0b0b'
INK_SOFT = '#52514e'
GRID = '#d8d7d2'

BASELINE = 'const'
STUDY_DIR = 'study_noise'
FIGDIR = 'figures'
KFOLD_K, KFOLD_REPEATS, KFOLD_SEED = 5, 20, 0
STUDY_WRITER = 'run_noise_study.score'


# --------------------------------------------------------------------------
# 1. the candidate study
# --------------------------------------------------------------------------

def run_study(x, y):
    """Score every noise form out of sample; store the table as a record."""
    domain = Domain()
    domain.add_category('noise', sorted(NOISE_FORMS))
    domain.add_constant('mean_degree', DEGREE)

    rows = [{'noise': name, 'mean_degree': DEGREE} for name in NOISE_FORMS]
    study = ExperimentData(domain=domain, input_data=rows)

    @datagenerator(output_names=[
        'loo_lpd', 'loo_rmse', 'loo_coverage_2sd', 'loo_z_sd',
        'kfold_lpd', 'kfold_lpd_sd', 'n_params', 'nll', 'aic', '_source_study'])
    def score(noise: str, mean_degree: int):
        loo = leave_one_out(x, y, noise, mean_degree)
        z = (y - loo['mu']) / loo['sd']
        kf = kfold_log_density(x, y, noise, mean_degree,
                               k=KFOLD_K, repeats=KFOLD_REPEATS, seed=KFOLD_SEED)
        fitted = GaussianPolyModel(noise=noise, degree=mean_degree).fit(x, y)
        return (float(loo['log_density'].mean()),
                float(np.sqrt(np.mean((y - loo['mu']) ** 2))),
                float(np.mean(np.abs(z) <= 2.0)),
                float(z.std(ddof=1)),
                kf['mean'], kf['sd'],
                float(fitted.n_params), fitted.nll, fitted.aic,
                STUDY_WRITER)

    study = score.call(study, mode='sequential')
    study.store(STUDY_DIR)

    study_in, study_out = ExperimentData.from_file(STUDY_DIR).to_pandas()
    table = study_in.join(study_out)
    return table


# --------------------------------------------------------------------------
# 2. the head-to-head, from the record's own held-out columns
# --------------------------------------------------------------------------

def head_to_head(x, y, winner, out_df):
    """Compare winner against baseline stop by stop, on the held-out columns."""
    lpd_w = out_df[f'logpd_loo_{winner}'].to_numpy(float)
    lpd_b = out_df[f'logpd_loo_{BASELINE}'].to_numpy(float)
    gain = lpd_w - lpd_b

    print(f'\n--- {winner} vs {BASELINE}, stop by stop, none of it in sample ---')
    print(f'mean held-out log density : {lpd_w.mean():+.4f}  vs {lpd_b.mean():+.4f}'
          f'   (gain {gain.mean():+.4f} nats per stop)')
    print(f'total over the 50 stops   : {lpd_w.sum():+.3f} vs {lpd_b.sum():+.3f}'
          f'   (gain {gain.sum():+.3f} nats)')
    print(f'stops where {winner} scores higher: {int(np.sum(gain > 0))}/{gain.size}')
    stat, pvalue = wilcoxon(gain)
    print(f'Wilcoxon signed-rank on the 50 paired gains: '
          f'statistic {stat:.1f}, p = {pvalue:.3g}')
    print(f'mean gain / standard error: '
          f'{gain.mean() / (gain.std(ddof=1) / np.sqrt(gain.size)):.3f}')
    print('(both treat the 50 gains as independent, which they are not quite:'
          '\n the 50 leave-one-out fits share 48 stops each, so read these as'
          '\n descriptive. The repeated 5-fold column above is the cleaner test'
          '\n -- there the two forms are separated by many times the spread'
          '\n across its 20 shuffles.)')

    # calibration: a standardised held-out residual should have spread 1
    # everywhere. This is the assumption being tested, so it is reported bin by
    # bin rather than pooled -- pooling is exactly what hid the problem.
    order = np.argsort(x)
    edges = [order[k * 10:(k + 1) * 10] for k in range(5)]
    print('\nspread of the standardised held-out residual  z = (y - mu)/sd,')
    print('by velocity bin -- it should be 1.0 in every bin:')
    header = ''.join(f'{x[b].min():5.0f}-{x[b].max():<5.0f}' for b in edges)
    print(f'{"x [m/s]":>16s}  {header}')
    rows = {}
    for name in (BASELINE, winner):
        z = ((y - out_df[f'y_pred_loo_{name}'].to_numpy(float))
             / out_df[f'sd_loo_{name}'].to_numpy(float))
        rows[name] = np.array([z[b].std(ddof=1) for b in edges])
        print(f'{"sd(z), " + name:>16s}  '
              + ''.join(f'{v:^10.2f}' for v in rows[name]))
    print(f'{"":>16s}  worst-bin departure from 1.0:  '
          + ',  '.join(f'{k}: {np.abs(v - 1.0).max():.2f}' for k, v in rows.items()))
    return gain, rows, edges


# --------------------------------------------------------------------------
# 3. figures, drawn from the stored record
# --------------------------------------------------------------------------

def _style(ax):
    ax.set_facecolor('white')
    ax.grid(True, color=GRID, linewidth=0.6, alpha=0.9)
    ax.set_axisbelow(True)
    for side in ('top', 'right'):
        ax.spines[side].set_visible(False)
    for side in ('left', 'bottom'):
        ax.spines[side].set_color(GRID)
    ax.tick_params(colors=INK_SOFT, labelsize=9, length=3)


def figure_bands(x, y, model_new, model_old, winner, path):
    """The fix itself: one band that fans out, against one that cannot."""
    grid = np.linspace(x.min(), x.max(), 400)
    mu_new, sd_new = model_new.mean(grid), model_new.std(grid)
    mu_old, sd_old = model_old.mean(grid), model_old.std(grid)

    fig, ax = plt.subplots(figsize=(7.6, 5.2))
    fig.patch.set_facecolor('white')
    _style(ax)

    ax.fill_between(grid, mu_new - 2 * sd_new, mu_new + 2 * sd_new,
                    color=AQUA, alpha=0.18, linewidth=0,
                    label=r'new: $\mu(x)\pm2\,\mathrm{sd}(x)$, sd grows with $x$')
    ax.plot(grid, mu_old - 2 * sd_old, color=ORANGE, linewidth=1.6,
            linestyle='--',
            label=r'old: $\mu(x)\pm2\,\mathrm{sd}$, one constant sd')
    ax.plot(grid, mu_old + 2 * sd_old, color=ORANGE, linewidth=1.6,
            linestyle='--')
    ax.plot(grid, mu_new, color=AQUA, linewidth=2)
    ax.scatter(x, y, s=40, color=BLUE, edgecolor='white', linewidth=1.0,
               zorder=3, label='50 measurements')
    ax.axhline(0, color=INK_SOFT, linewidth=0.8, alpha=0.5)

    low, high = 8.0, 78.0
    ax.annotate('the old band is far too wide here --\n'
                'it predicts negative stopping distances',
                xy=(low, float(model_old.mean(np.array([low]))[0]
                               - 2 * model_old.std(np.array([low]))[0])),
                xytext=(19, -150), color=ORANGE, fontsize=9,
                arrowprops=dict(arrowstyle='->', color=ORANGE, linewidth=1.0))
    ax.annotate('and too narrow here',
                xy=(high, float(model_old.mean(np.array([high]))[0]
                                + 2 * model_old.std(np.array([high]))[0])),
                xytext=(48, 840), color=ORANGE, fontsize=9,
                arrowprops=dict(arrowstyle='->', color=ORANGE, linewidth=1.0))
    ax.set_ylim(-210, 910)

    ax.set_xlabel('velocity when the obstacle is seen  x  [m/s]', color=INK,
                  fontsize=10)
    ax.set_ylabel('stopping distance  y  [m]', color=INK, fontsize=10)
    ax.set_title('The fix: a noise that grows with velocity',
                 color=INK, fontsize=12, loc='left', pad=10)
    ax.text(0.03, 0.96,
            f'new  sd(x) = {np.exp(model_new.theta[0]):.3g}'
            + (r'$\,x$' if winner == 'prop' else ' (fitted form)')
            + f'\nold  sd = {model_old.std(np.array([1.0]))[0]:.3g} m, everywhere',
            transform=ax.transAxes, va='top', ha='left', color=INK_SOFT,
            fontsize=9,
            bbox=dict(facecolor='white', edgecolor=GRID,
                      boxstyle='round,pad=0.5'))
    ax.legend(loc='upper left', bbox_to_anchor=(0.03, 0.80), frameon=False,
              fontsize=9, labelcolor=INK_SOFT)

    fig.savefig(path, dpi=200, bbox_inches='tight', facecolor='white')
    plt.close(fig)
    print(f'wrote {path}')


def figure_heldout(x, y, out_df, winner, gain, bin_sd, bins, path):
    """The evidence: every point here was predicted by a fit that lacked it."""
    fig, axes = plt.subplots(3, 1, figsize=(7.6, 8.6), sharex=True,
                             gridspec_kw={'height_ratios': [1, 1, 1.15],
                                          'hspace': 0.30})
    fig.patch.set_facecolor('white')
    for ax in axes:
        _style(ax)

    # panels 1-2: standardised held-out residuals, one model each, same scale.
    # The claim is not that fewer points fall outside a band -- it is that the
    # spread of z is the same at every velocity, so each panel also carries the
    # spread measured inside each velocity bin, which should sit on +/-1.
    for ax, name, color, title in (
            (axes[0], BASELINE,  ORANGE,
             'old model, one constant sd: $z$ is squashed at low $x$ and '
             'bursts at high $x$'),
            (axes[1], winner, AQUA,
             'new model, sd grows with $x$: $z$ has the same spread at every '
             'velocity')):
        z = ((y - out_df[f'y_pred_loo_{name}'].to_numpy(float))
             / out_df[f'sd_loo_{name}'].to_numpy(float))
        ax.axhline(0, color=INK_SOFT, linewidth=0.9, alpha=0.6)
        for sign in (1, -1):
            ax.axhline(sign, color=INK_SOFT, linewidth=0.9, linestyle=':',
                       alpha=0.8)
        for b, s in zip(bins, bin_sd[name]):
            lo, hi = x[b].min(), x[b].max()
            for sign in (1, -1):
                ax.hlines(sign * s, lo, hi, color=color, linewidth=2.4,
                          zorder=2)
        ax.scatter(x, z, s=34, color=BLUE, edgecolor='white', linewidth=0.9,
                   zorder=3)
        ax.set_ylim(-4.2, 4.2)
        ax.set_ylabel('held-out\n$z=(y-\\mu)/\\mathrm{sd}$', color=INK,
                      fontsize=10)
        ax.set_title(title, color=INK_SOFT, fontsize=9, loc='left', pad=6)
        ax.text(0.985, 0.05,
                f'spread of $z$ within a bin: '
                f'{bin_sd[name].min():.2f} to {bin_sd[name].max():.2f}'
                f'   (should be 1.00)',
                transform=ax.transAxes, ha='right', va='bottom',
                color=INK_SOFT, fontsize=9,
                bbox=dict(facecolor='white', edgecolor=GRID,
                          boxstyle='round,pad=0.35'))
    axes[0].text(x.min(), 1.0, ' $\\pm1$ ', color=INK_SOFT, fontsize=8,
                 va='bottom', ha='left')

    # panel 3: the score itself, stop by stop
    ax = axes[2]
    ax.axhline(0, color=INK_SOFT, linewidth=1.0)
    for mask, color in ((gain > 0, AQUA), (gain <= 0, ORANGE)):
        ax.vlines(x[mask], 0, gain[mask], color=color, linewidth=2.2)
        ax.scatter(x[mask], gain[mask], s=26, color=color, edgecolor='white',
                   linewidth=0.8, zorder=3)
    ax.set_ylim(min(gain.min() * 1.4, -1.0), gain.max() * 1.35)
    ax.set_ylabel('gain in held-out\n$\\log p(y\\,|\\,x)$  [nats]', color=INK,
                  fontsize=10)
    ax.set_xlabel('velocity when the obstacle is seen  x  [m/s]', color=INK,
                  fontsize=10)
    ax.set_title(f'per-stop score: new model minus old '
                 f'({int(np.sum(gain > 0))} of {gain.size} stops in favour of '
                 f'the new one, mean {gain.mean():+.2f} nats)',
                 color=INK_SOFT, fontsize=9, loc='left', pad=6)
    handles = [plt.Line2D([], [], color=AQUA, linewidth=2.6),
               plt.Line2D([], [], color=ORANGE, linewidth=2.6)]
    ax.legend(handles, ['new model better', 'old model better'],
              loc='upper center', ncol=2, frameon=False, fontsize=9,
              labelcolor=INK_SOFT)

    fig.suptitle('Leave-one-out: every point below was predicted by a model '
                 'fitted without it', color=INK, fontsize=12, x=0.005,
                 ha='left', y=0.955)

    fig.savefig(path, dpi=200, bbox_inches='tight', facecolor='white')
    plt.close(fig)
    print(f'wrote {path}')


# --------------------------------------------------------------------------

def main():
    data = ExperimentData.from_file('data')
    input_df, output_df = data.to_pandas()
    x = input_df['x'].to_numpy(float)
    y = output_df['y'].to_numpy(float)

    # the constant-noise model, refitted here by the new code, must reproduce
    # the old one -- otherwise the comparison is between two fitters
    old = GaussianPolyModel(noise=BASELINE).fit(x, y)
    print('--- the old assumption, refitted by the new code ---')
    print(old.summary())
    print('(car_ml reported sd_mle = 29.0102 for the same 50 stops)')

    print('\n--- scoring four noise forms out of sample ---')
    table = run_study(x, y)
    print(table.to_string(index=False))

    winner = str(table.loc[table['loo_lpd'].idxmax(), 'noise'])
    print(f'\nbest held-out log density: {winner!r}  '
          f'({NOISE_FORMS[winner][2]})')
    if winner == BASELINE:
        raise SystemExit('the constant-noise model won; nothing to replace')

    # ---- write the winner and the held-out columns into the record ---------
    new = GaussianPolyModel(noise=winner).fit(x, y)
    print(f'\n--- fitting {winner} on all 50 stops and storing it ---')
    data = FitNoiseModel(noise=winner).call(data)
    for name in (BASELINE, winner):
        data = WriteLeaveOneOut(noise=name).call(data)
    data.store('data')

    stored = ExperimentData.from_file('data')
    input_df, output_df = stored.to_pandas()
    x = input_df['x'].to_numpy(float)
    y = output_df['y'].to_numpy(float)
    print(f'\ncolumns now in the record: {list(output_df.columns)}')
    missing = [c for c in output_df.columns if output_df[c].isna().any()]
    print(f'columns with missing entries: {missing if missing else "none"}')
    for c in output_df.columns:
        if c.startswith('_source_'):
            print(f'  {c} = {sorted(set(output_df[c]))}')

    gain, bin_sd, bins = head_to_head(x, y, winner, output_df)

    # both ends of the comparison, to make the two numbers concrete: the stop
    # the new model gains most on, and the stop it loses most on
    for label, i in (('the stop the new model gains most on', int(np.argmax(gain))),
                     ('the stop it loses most on', int(np.argmin(gain)))):
        print(f'\n{label}: x = {x[i]:.2f} m/s, y = {y[i]:.1f} m, '
              f'held out of both fits')
        for name in (BASELINE, winner):
            mu = output_df[f'y_pred_loo_{name}'].to_numpy(float)[i]
            sd = output_df[f'sd_loo_{name}'].to_numpy(float)[i]
            print(f'  {name:6s}: mu = {mu:7.1f} m, sd = {sd:6.1f} m  ->  '
                  f'y is {abs(y[i] - mu) / sd:.2f} sd away, '
                  f'log density {norm.logpdf(y[i], mu, sd):+.3f}')
        print(f'  gain {gain[i]:+.3f} nats')

    # ---- what the fitted numbers would mean, if the textbook form held ------
    # y = t*x + x^2/(2a) is the usual reaction-plus-braking decomposition. The
    # record cannot confirm it -- nobody said where these numbers came from --
    # but the fitted coefficients can be read in its units and checked for
    # plausibility, which is a test the model could fail.
    if winner == 'prop':
        b0, b1, b2 = new.beta
        s1 = float(np.exp(new.theta[0]))
        print('\n--- reading the fitted numbers as reaction + braking ---')
        print(f'x^1 coefficient   {b1:.4f} s        -> mean reaction time')
        print(f'x^2 coefficient   {b2:.5f} s^2/m   -> deceleration '
              f'a = 1/(2*b2) = {1.0 / (2.0 * b2):.3f} m/s^2 '
              f'({1.0 / (2.0 * b2) / 9.81:.3f} g)')
        print(f'intercept         {b0:+.4f} m       -> should be 0; it is '
              f'{abs(b0) / y.mean() * 100:.2f}% of the mean stopping distance')
        print(f'sd(x) = {s1:.4f}*x, i.e. {s1:.4f} s of travel  -> spread in '
              f'reaction time,\n{"":18s}{s1 / b1 * 100:.1f}% of the mean '
              f'reaction time. A noise proportional to x is what a spread in '
              f'reaction\n{"":18s}time alone would produce, since only the '
              f'reaction term carries it.')

    # ---- figures ------------------------------------------------------------
    print()
    os.makedirs(FIGDIR, exist_ok=True)
    figure_bands(x, y, new, old, winner,
                 os.path.join(FIGDIR, 'noise_grows_with_velocity.png'))
    figure_heldout(x, y, output_df, winner, gain, bin_sd, bins,
                   os.path.join(FIGDIR, 'held_out_comparison.png'))


if __name__ == '__main__':
    main()
