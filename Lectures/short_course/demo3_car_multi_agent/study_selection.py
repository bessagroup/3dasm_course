"""Out-of-sample selection among the eight conditional-density candidates.

Run with the interpreter that actually has f3dasm 2.4.0:

    /home/mbessa/miniforge3/envs/aescape_course/bin/python study_selection.py

What this script does, and nothing else
---------------------------------------
It is the *judge*.  It does not fit, patch or improve any model: it imports
``car_noise`` and calls ``car_noise.make(name).fit(x_train, y_train)`` inside
every training split, then evaluates the resulting conditional density
``p(y | x) = Normal(mu(x), sd(x))`` only on the rows that fit never saw.

Three split schemes over the 50 measurements (there are no more):

  ``loo``     leave-one-out.  50 folds, 49 training rows each, every
              measurement held out exactly once.  Fully deterministic -- no
              seed, nothing to reproduce by luck.  Chosen as the primary
              scheme because at n = 50 with 4-5 parameter models the training
              set should stay as large as possible, and because it happens to
              hold out the two extreme velocities (x = 3 and x = 81.75) as
              genuine extrapolations.

  ``skf5x20`` repeated stratified 5-fold.  20 repeats, 1000 held-out
              predictions in total, 40 training rows per fit.  Stratified on
              x: the 50 rows are sorted by velocity, cut into blocks of 5,
              and one row of each block goes to each fold, so no fold is a
              pure extrapolation and every training set still spans the whole
              velocity range.  Randomness is only in the within-block
              permutation: ``np.random.default_rng(SEED + repeat)`` with
              ``SEED = 20260907``, so the numbers reproduce exactly.
              Chosen as a second opinion because LOO training sets overlap in
              48 of 49 rows and LOO is a high-variance risk estimator; a 20%
              hold-out with 20 repeats stresses the fits harder.

  ``edges``   deterministic extrapolation stress test.  Train on the middle
              40 velocities, hold out the 5 slowest and the 5 fastest.  This
              is where a fitted sd(x) can go to zero or explode, so it is the
              scheme that exposes a candidate that cannot be trusted off the
              training range.  One fold, 10 held-out rows -- a small, noisy
              number, reported as a warning flag rather than as a ranking.

Metrics on held-out rows only
-----------------------------
  ``mean_lpd``   mean log predictive density, nats per held-out stop.  This
                 is the metric that can see a noise fix at all.
  ``rmse``       root mean squared error of the mean -- reported precisely
                 because it should be nearly identical across candidates,
                 which is the point: it cannot judge a noise model.
  coverage of the central 50% and 90% predictive intervals, the spread of the
  held-out z-scores overall and separately below/above the median held-out
  velocity (the calibration-vs-x check), a Kolmogorov-Smirnov statistic on
  the PIT values, and blow-up watch columns: smallest held-out sd, largest
  |z|, worst single log density, and a count of non-finite scores.

Outputs
-------
  ``study_selection/``  f3dasm ExperimentData: 8 candidates x 3 schemes,
                        one row each, with all of the above as output columns
                        plus ``_source_selector``.
  ``data/``             the main record gains per-measurement LOO columns for
                        the baseline and for the out-of-sample winner,
                        stamped ``_source_selector``.
  ``figures/selection.png``
"""

from __future__ import annotations

import os

import numpy as np
import pandas as pd
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
from scipy.stats import norm, kstest

from f3dasm import ExperimentData, datagenerator
from f3dasm.design import Domain

import car_noise

SELECTOR = 'study_selection.py (selector)'
SEED = 20260907
N_REPEATS = 20
K = 5
BASELINE = 'quad_ols'
NAMES = list(car_noise.CANDIDATES)          # 8 candidates, registry order
LOG2PI = float(np.log(2.0 * np.pi))


# --------------------------------------------------------------------------
# the data: 50 rows, and that is the entire supply
# --------------------------------------------------------------------------

# Columns this script owns in data/.  f3dasm keeps existing output columns and
# overwrites same-named ones, so re-running is idempotent for the names below;
# the transitional names from an earlier run of this same script are pruned so
# that a clean copy of the folder and a dirty one end up with the same record.
OWNED = ([f'{p}_loo_{c}' for c in (BASELINE, 'affine', 'phys')
          for p in ('y', 'sd', 'lpd')]
         + ['y_loo_base', 'sd_loo_base', 'lpd_loo_base',
            'y_loo_best', 'sd_loo_best', 'lpd_loo_best', 'loo_best_name'])


def prune_owned():
    """Drop stale selector columns from data/ before writing the current set."""
    path = os.path.join('data', 'experiment_data', 'output.csv')
    df = pd.read_csv(path, index_col=0)
    drop = [c for c in df.columns if c in OWNED]
    if drop:
        print(f'pruning stale selector columns from data/: {drop}')
        df.drop(columns=drop).to_csv(path)


def load():
    prune_owned()
    data = ExperimentData.from_file('data')
    input_df, output_df = data.to_pandas()
    x = input_df['x'].to_numpy(float)
    y = output_df['y'].to_numpy(float)
    return data, x, y


# --------------------------------------------------------------------------
# the split schemes.  Each returns a list of (train_idx, test_idx).
# --------------------------------------------------------------------------

def folds_loo(n):
    allidx = np.arange(n)
    return [(np.delete(allidx, i), np.array([i])) for i in range(n)]


def folds_stratified_kfold(x, k=K, n_repeats=N_REPEATS, seed=SEED):
    n = len(x)
    order = np.argsort(x, kind='stable')
    out = []
    for r in range(n_repeats):
        rng = np.random.default_rng(seed + r)
        label = np.empty(n, dtype=int)
        for b0 in range(0, n, k):
            block = order[b0:b0 + k]
            label[block] = rng.permutation(len(block))
        for f in range(k):
            test = np.flatnonzero(label == f)
            train = np.flatnonzero(label != f)
            out.append((train, test))
    return out


def folds_edges(x, m=5):
    n = len(x)
    order = np.argsort(x, kind='stable')
    test = np.concatenate([order[:m], order[n - m:]])
    train = np.setdiff1d(np.arange(n), test)
    return [(train, test)]


# --------------------------------------------------------------------------
# one pass: refit every candidate from scratch inside every training split
# --------------------------------------------------------------------------

def run_scheme(name, folds, x, y):
    """Return {candidate: dict of pooled held-out arrays}."""
    n_test_total = sum(len(t) for _, t in folds)
    store = {c: {'x': np.empty(n_test_total), 'y': np.empty(n_test_total),
                 'mu': np.empty(n_test_total), 'sd': np.empty(n_test_total),
                 'fold': np.empty(n_test_total, dtype=int),
                 'idx': np.empty(n_test_total, dtype=int),
                 'params': [], 'fit_errors': 0}
             for c in NAMES}
    pos = 0
    for fi, (tr, te) in enumerate(folds):
        for c in NAMES:
            try:
                model = car_noise.make(c).fit(x[tr], y[tr])
                store[c]['params'].append(dict(model.params))
                mu = np.atleast_1d(np.asarray(model.mean(x[te]), float))
                sd = np.atleast_1d(np.asarray(model.std(x[te]), float))
            except Exception as exc:            # reported, never repaired
                print(f'  FIT FAILURE {name} fold {fi} candidate {c}: {exc!r}')
                store[c]['fit_errors'] += 1
                mu = np.full(len(te), np.nan)
                sd = np.full(len(te), np.nan)
            s = store[c]
            sl = slice(pos, pos + len(te))
            s['x'][sl] = x[te]; s['y'][sl] = y[te]
            s['mu'][sl] = mu; s['sd'][sl] = sd
            s['fold'][sl] = fi; s['idx'][sl] = te
        pos += len(te)
    if (fi + 1) % 1 == 0:
        pass
    print(f'  {name}: {len(folds)} folds x {len(NAMES)} candidates '
          f'= {len(folds) * len(NAMES)} independent fits, '
          f'{n_test_total} held-out predictions each')
    return store


def metrics(s, n_folds):
    """Held-out metrics from one candidate's pooled arrays."""
    xo, yo, mu, sd = s['x'], s['y'], s['mu'], s['sd']
    with np.errstate(divide='ignore', invalid='ignore'):
        z = (yo - mu) / sd
        lpd = -0.5 * (LOG2PI + 2.0 * np.log(sd) + z * z)
    finite = np.isfinite(lpd)
    n_bad = int(np.sum(~finite))
    err = yo - mu
    xmed = float(np.median(xo))
    lo, hi = xo <= xmed, xo > xmed
    pit = norm.cdf(z[finite])
    ks = float(kstest(pit, 'uniform').statistic) if finite.any() else np.nan
    # per-fold mean lpd, to expose fold-to-fold spread
    fold_means = []
    for f in np.unique(s['fold']):
        m = (s['fold'] == f) & finite
        if m.any():
            fold_means.append(float(np.mean(lpd[m])))
    return {
        'n_out': int(len(yo)),
        'n_folds': int(n_folds),
        'n_nonfinite_lpd': n_bad,
        'n_fit_errors': int(s['fit_errors']),
        'mean_lpd': float(np.mean(lpd[finite])) if finite.any() else np.nan,
        'total_lpd': float(np.sum(lpd[finite])) if finite.any() else np.nan,
        'lpd_se': float(np.std(lpd[finite], ddof=1) / np.sqrt(finite.sum()))
                  if finite.sum() > 1 else np.nan,
        'fold_lpd_sd': float(np.std(fold_means, ddof=1))
                       if len(fold_means) > 1 else np.nan,
        'rmse': float(np.sqrt(np.mean(err ** 2))),
        'mae': float(np.mean(np.abs(err))),
        'cov50': float(np.mean(np.abs(z[finite]) <= norm.ppf(0.75)))
                 if finite.any() else np.nan,
        'cov90': float(np.mean(np.abs(z[finite]) <= norm.ppf(0.95)))
                 if finite.any() else np.nan,
        'width90_med': float(np.median(2 * norm.ppf(0.95) * sd[np.isfinite(sd)])),
        'z_mean': float(np.mean(z[finite])) if finite.any() else np.nan,
        'z_sd': float(np.std(z[finite], ddof=1)) if finite.sum() > 1 else np.nan,
        'z_sd_lowx': float(np.std(z[lo & finite], ddof=1)),
        'z_sd_highx': float(np.std(z[hi & finite], ddof=1)),
        'pit_ks': ks,
        'min_sd_out': float(np.nanmin(sd)),
        'max_absz': float(np.nanmax(np.abs(z))),
        'worst_lpd': float(np.min(lpd[finite])) if finite.any() else np.nan,
    }


# --------------------------------------------------------------------------
def main():
    data, x, y = load()
    n = len(x)
    print(f'record: n = {n}, x in [{x.min():g}, {x.max():g}] m/s, '
          f'y in [{y.min():.3f}, {y.max():.3f}] m')
    print(f'candidates ({len(NAMES)}): {NAMES}')
    print(f'baseline to beat: {BASELINE}')
    print(f'seed = {SEED}, repeats = {N_REPEATS}, K = {K}\n')

    schemes = {
        'loo':     folds_loo(n),
        'skf5x20': folds_stratified_kfold(x),
        'edges':   folds_edges(x),
    }
    raw, res = {}, {}
    for sname, folds in schemes.items():
        raw[sname] = run_scheme(sname, folds, x, y)
        res[sname] = {c: metrics(raw[sname][c], len(folds)) for c in NAMES}
    print()

    # ---- paired differences vs the baseline, on identical held-out rows ----
    for sname in schemes:
        base = raw[sname][BASELINE]
        with np.errstate(divide='ignore', invalid='ignore'):
            zb = (base['y'] - base['mu']) / base['sd']
            lpd_b = -0.5 * (LOG2PI + 2.0 * np.log(base['sd']) + zb * zb)
        for c in NAMES:
            s = raw[sname][c]
            with np.errstate(divide='ignore', invalid='ignore'):
                zc = (s['y'] - s['mu']) / s['sd']
                lpd_c = -0.5 * (LOG2PI + 2.0 * np.log(s['sd']) + zc * zc)
            d = lpd_c - lpd_b
            ok = np.isfinite(d)
            res[sname][c]['d_lpd_mean'] = float(np.mean(d[ok]))
            res[sname][c]['d_lpd_se'] = (
                float(np.std(d[ok], ddof=1) / np.sqrt(ok.sum()))
                if ok.sum() > 1 else np.nan)
            res[sname][c]['dens_ratio'] = float(np.exp(np.mean(d[ok])))

    # ---- the study table as an f3dasm ExperimentData -----------------------
    domain = Domain()
    domain.add_category('candidate', NAMES)
    domain.add_category('scheme', list(schemes))
    rows = [{'candidate': c, 'scheme': s} for s in schemes for c in NAMES]

    out_names = ['n_out', 'n_folds', 'n_params', 'mean_lpd', 'lpd_se',
                 'd_lpd_mean', 'd_lpd_se', 'dens_ratio', 'total_lpd',
                 'fold_lpd_sd', 'rmse', 'mae', 'cov50', 'cov90',
                 'width90_med', 'z_mean', 'z_sd', 'z_sd_lowx', 'z_sd_highx',
                 'pit_ks', 'min_sd_out', 'max_absz', 'worst_lpd',
                 'n_nonfinite_lpd', 'n_fit_errors', '_source_selector']
    nparams = {c: car_noise.make(c).n_params for c in NAMES}

    @datagenerator(output_names=out_names)
    def score(candidate: str, scheme: str):
        m = dict(res[scheme][candidate])
        m['n_params'] = nparams[candidate]
        vals = [m[k] for k in out_names[:-1]]
        return tuple(float(v) for v in vals) + (SELECTOR,)

    study = ExperimentData(domain=domain, input_data=rows)
    study = score.call(study, mode='sequential')
    study.store('study_selection')
    si, so = study.to_pandas()
    tbl = si.join(so)

    pd_opts = dict(float_format=lambda v: f'{v:.4f}')
    print('=' * 118)
    print('HELD-OUT SCORING TABLE  (every fit refitted from scratch inside '
          'its own training split)')
    print('=' * 118)
    for sname in schemes:
        sub = tbl[tbl['scheme'] == sname].copy()
        sub = sub.sort_values('mean_lpd', ascending=False)
        print(f'\n--- scheme = {sname} '
              f'({int(sub["n_folds"].iloc[0])} folds, '
              f'{int(sub["n_out"].iloc[0])} held-out predictions) ---')
        cols = ['candidate', 'n_params', 'mean_lpd', 'lpd_se', 'd_lpd_mean',
                'd_lpd_se', 'dens_ratio', 'rmse', 'cov50', 'cov90',
                'z_sd_lowx', 'z_sd_highx', 'pit_ks', 'min_sd_out',
                'max_absz', 'worst_lpd', 'n_nonfinite_lpd', 'n_fit_errors']
        print(sub[cols].to_string(index=False, **pd_opts))
    print('\n--- full stored table, all columns ---')
    print(tbl.to_string(**pd_opts))

    # ---- the verdict, in interpretable units ------------------------------
    loo = tbl[tbl['scheme'] == 'loo'].set_index('candidate')
    skf = tbl[tbl['scheme'] == 'skf5x20'].set_index('candidate')
    winner = loo['mean_lpd'].idxmax()
    winner_skf = skf['mean_lpd'].idxmax()
    print('\n' + '=' * 118)
    print(f'LOO winner              : {winner}')
    print(f'skf5x20 winner          : {winner_skf}')
    print(f'baseline                : {BASELINE}')
    for nm, t in (('loo', loo), ('skf5x20', skf)):
        d = float(t.loc[winner, 'mean_lpd'] - t.loc[BASELINE, 'mean_lpd'])
        print(f'[{nm}] {winner} - {BASELINE}: '
              f'd(mean lpd) = {d:+.4f} nats/stop  '
              f'(paired SE {float(t.loc[winner, "d_lpd_se"]):.4f}), '
              f'density ratio = {np.exp(d):.4g}x per held-out stop, '
              f'total over {int(t.loc[winner, "n_out"])} points = '
              f'{d * int(t.loc[winner, "n_out"]):+.2f} nats')
        print(f'[{nm}] rmse: {winner} = {float(t.loc[winner, "rmse"]):.4f} m, '
              f'{BASELINE} = {float(t.loc[BASELINE, "rmse"]):.4f} m, '
              f'ratio = {float(t.loc[winner, "rmse"]) / float(t.loc[BASELINE, "rmse"]):.4f}'
              f'   (spread of rmse across all 8 candidates: '
              f'{float(t["rmse"].min()):.4f} to {float(t["rmse"].max()):.4f} m, '
              f'{100 * (float(t["rmse"].max()) / float(t["rmse"].min()) - 1):.2f}%)')
        print(f'[{nm}] spread of mean_lpd across all 8 candidates: '
              f'{float(t["mean_lpd"].min()):.4f} to '
              f'{float(t["mean_lpd"].max()):.4f} nats/stop')
    # ---- head-to-head: is the top of the table actually separated? --------
    def lpd_vec(sname, c):
        s = raw[sname][c]
        with np.errstate(divide='ignore', invalid='ignore'):
            z = (s['y'] - s['mu']) / s['sd']
            return -0.5 * (LOG2PI + 2.0 * np.log(s['sd']) + z * z)

    print('\nHEAD-TO-HEAD, paired on identical held-out rows')
    print('(d = mean lpd of A minus mean lpd of B; wins = rows where A wins;')
    print(' the LOO folds share 48 of 49 training rows, so the LOO paired SE')
    print(' is an approximation, not an independent-sample standard error)')
    for sname in ('loo', 'skf5x20', 'edges'):
        t = tbl[tbl['scheme'] == sname].set_index('candidate')
        top = t['mean_lpd'].sort_values(ascending=False).index.tolist()[:3]
        print(f'\n  scheme = {sname}   (top three: {top})')
        pairs = [(top[0], top[1]), (top[0], top[2]), (top[1], top[2]),
                 (top[0], BASELINE), ('phys', BASELINE), ('phys', 'affine')]
        seen = set()
        for A, B in pairs:
            if A == B or (A, B) in seen:
                continue
            seen.add((A, B))
            d = lpd_vec(sname, A) - lpd_vec(sname, B)
            ok = np.isfinite(d)
            m = float(np.mean(d[ok]))
            se = (float(np.std(d[ok], ddof=1) / np.sqrt(ok.sum()))
                  if ok.sum() > 1 else float('nan'))
            wins = int(np.sum(d[ok] > 0))
            print(f'    {A:9s} vs {B:9s}  d = {m:+.4f} nats/stop  '
                  f'SE = {se:.4f}  |d|/SE = {abs(m) / se if se else float("nan"):5.2f}  '
                  f'wins {wins}/{int(ok.sum())}')

    # ---- fit stability across LOO folds -----------------------------------
    print('\nFIT STABILITY ACROSS THE 50 LOO TRAINING SPLITS')
    print('(median, and min/max, of each fitted parameter over the 50 refits)')
    for c in NAMES:
        P = raw['loo'][c]['params']
        keys = list(P[0])
        bits = []
        for k in keys:
            v = np.array([p[k] for p in P], float)
            bits.append(f'{k}: med {np.median(v):+.5g} '
                        f'[{v.min():+.5g}, {v.max():+.5g}]')
        print(f'  {c:9s} ' + ' | '.join(bits))
    print('  boundary hits: how many of the 50 refits drove a positive '
          'noise parameter to ~0 (< 1e-6)')
    for c in NAMES:
        P = raw['loo'][c]['params']
        scale = [k for k in P[0] if not k.startswith('b')]
        hits = {k: int(np.sum(np.abs([p[k] for p in P]) < 1e-6))
                for k in scale}
        if any(hits.values()):
            print(f'    {c:9s} {hits}   <-- degenerate: the shape has fewer '
                  'effective parameters than it claims')
        else:
            print(f'    {c:9s} {hits}')

    # ---- an interpretable readout: predictive sd at the extreme stops -----
    print('\nPREDICTIVE sd AT A STOP THE FIT NEVER SAW (leave-one-out), '
          'slowest and fastest driver')
    xs_lo, xs_hi = int(np.argmin(x)), int(np.argmax(x))
    print(f'  {"candidate":9s} {"sd @ x=" + f"{x[xs_lo]:g}":>18s} '
          f'{"sd @ x=" + f"{x[xs_hi]:g}":>18s}   '
          f'(actual y there: {y[xs_lo]:.2f} m and {y[xs_hi]:.2f} m)')
    for c in NAMES:
        s = raw['loo'][c]
        m = {int(i): float(v) for i, v in zip(s['idx'], s['sd'])}
        print(f'  {c:9s} {m[xs_lo]:18.3f} {m[xs_hi]:18.3f}')

    hetero = [c for c in NAMES if c not in (BASELINE, 'const')]
    print(f'\nvelocity-dependent-noise candidates {hetero}')
    print('all beat the constant-noise baseline on LOO mean_lpd: '
          f'{bool(np.all([loo.loc[c, "mean_lpd"] > loo.loc[BASELINE, "mean_lpd"] for c in hetero]))}')
    print('all beat it on skf5x20 mean_lpd:                      '
          f'{bool(np.all([skf.loc[c, "mean_lpd"] > skf.loc[BASELINE, "mean_lpd"] for c in hetero]))}')
    print(f"modeler's provisional pick 'phys' is the LOO winner:  "
          f'{winner == "phys"}')

    print('\nDECISION')
    print(f'  primary scheme (loo) ranks: '
          f'{loo["mean_lpd"].sort_values(ascending=False).index.tolist()}')
    print(f'  secondary (skf5x20) ranks : '
          f'{skf["mean_lpd"].sort_values(ascending=False).index.tolist()}')
    edg = tbl[tbl['scheme'] == 'edges'].set_index('candidate')
    print(f'  extrapolation (edges)     : '
          f'{edg["mean_lpd"].sort_values(ascending=False).index.tolist()}')
    d_ap_loo = lpd_vec('loo', 'affine') - lpd_vec('loo', 'phys')
    se_ap = float(np.std(d_ap_loo, ddof=1) / np.sqrt(len(d_ap_loo)))
    print(f'  affine vs phys on loo: d = {float(np.mean(d_ap_loo)):+.4f} '
          f'nats/stop, paired SE = {se_ap:.4f}  ->  '
          f'separated at |d| > 2 SE: '
          f'{bool(abs(float(np.mean(d_ap_loo))) > 2 * se_ap)}')
    print('  the constant-vs-velocity-dependent question IS settled '
          '(see the gap to quad_ols, many SE wide);')
    print('  which velocity-dependent shape is best is NOT settled by these '
          '50 rows on interpolation,')
    print('  and only the edges scheme separates them -- on 10 points, so '
          'treat it as a warning flag.')
    print('=' * 118)

    # ---- write per-measurement LOO results into the main record -----------
    # Honest naming: the baseline plus the two candidates that tie at the top,
    # not one column called "best".
    loo_raw = raw['loo']
    WRITE = [BASELINE, 'affine', 'phys']

    def by_x(c):
        s = loo_raw[c]
        o = np.argsort(s['idx'])
        with np.errstate(divide='ignore', invalid='ignore'):
            z = (s['y'] - s['mu']) / s['sd']
            lp = -0.5 * (LOG2PI + 2.0 * np.log(s['sd']) + z * z)
        return {round(float(v), 9): (float(a), float(b), float(c2))
                for v, a, b, c2 in zip(s['x'][o], s['mu'][o], s['sd'][o], lp[o])}
    maps = {c: by_x(c) for c in WRITE}
    written = [f'{p}_loo_{c}' for c in WRITE for p in ('y', 'sd', 'lpd')]

    @datagenerator(output_names=written + ['_source_selector'])
    def write_loo(x: float):
        k = round(float(x), 9)
        vals = []
        for c in WRITE:
            vals.extend(maps[c][k])
        return tuple(vals) + (SELECTOR,)

    data = data.mark_all('open')
    data = write_loo.call(data, mode='sequential')
    data.store('data')
    di, do = data.to_pandas()
    print('\nrecord data/ now carries columns:')
    print(list(di.columns) + list(do.columns))
    print('\nfirst six rows of the leave-one-out columns this script wrote')
    print('(each y_loo_* / sd_loo_* is a prediction from a fit that never saw '
          'that row):')
    print(do[['y'] + written].head(6).to_string(**pd_opts))

    # ---- figure ------------------------------------------------------------
    make_figure(loo_raw, x, y, tbl, winner)
    print('\nwrote figures/selection.png, study_selection/, and the '
          '_source_selector columns in data/')


def make_figure(loo_raw, x, y, tbl, winner):
    def arrays(c):
        s = loo_raw[c]
        o = np.argsort(s['x'])
        with np.errstate(divide='ignore', invalid='ignore'):
            z = (s['y'] - s['mu']) / s['sd']
            lp = -0.5 * (LOG2PI + 2.0 * np.log(s['sd']) + z * z)
        return s['x'][o], s['y'][o], s['mu'][o], s['sd'][o], z[o], lp[o]

    loo = tbl[tbl['scheme'] == 'loo'].set_index('candidate')
    skf = tbl[tbl['scheme'] == 'skf5x20'].set_index('candidate')
    edg = tbl[tbl['scheme'] == 'edges'].set_index('candidate')
    order = loo['mean_lpd'].sort_values(ascending=False).index.tolist()
    THREE = ((BASELINE, 'crimson', 'o'), ('affine', '#35b779', '^'),
             ('phys', '#31688e', 's'))

    fig, ax = plt.subplots(2, 3, figsize=(18, 9.5))

    a = ax[0, 0]
    xs = np.arange(len(order))
    a.bar(xs - 0.2, [loo.loc[c, 'mean_lpd'] for c in order], 0.4,
          yerr=[loo.loc[c, 'lpd_se'] for c in order],
          label='LOO (50 held-out)', color='#31688e', capsize=2)
    a.bar(xs + 0.2, [skf.loc[c, 'mean_lpd'] for c in order], 0.4,
          yerr=[skf.loc[c, 'lpd_se'] for c in order],
          label='stratified 5-fold x20 (1000)', color='#35b779', capsize=2)
    a.axhline(loo.loc[BASELINE, 'mean_lpd'], color='crimson', ls='--', lw=1.2,
              label=f'{BASELINE} baseline (LOO)')
    a.set_xticks(xs); a.set_xticklabels(order, rotation=30, ha='right')
    a.set_ylabel('mean held-out log predictive density  [nats / stop]')
    a.set_title('(a) the metric that can see a noise fix\n'
                'higher is better; refitted inside every split')
    a.legend(fontsize=8); a.grid(alpha=.3, axis='y')

    a = ax[0, 1]
    a.bar(xs, [loo.loc[c, 'rmse'] for c in order], 0.6, color='#999999')
    a.set_xticks(xs); a.set_xticklabels(order, rotation=30, ha='right')
    lo_, hi_ = float(loo['rmse'].min()), float(loo['rmse'].max())
    a.set_ylim(0.95 * lo_, 1.05 * hi_)
    a.set_ylabel('LOO RMSE of the mean  [m]')
    a.set_title('(b) the metric that cannot: LOO RMSE\n'
                f'all 8 candidates within {100*(hi_/lo_-1):.1f}% '
                '(note the zoomed axis)')
    a.grid(alpha=.3, axis='y')

    a = ax[0, 2]
    ev = [edg.loc[c, 'mean_lpd'] for c in order]
    a.bar(xs, ev, 0.6, color=['#31688e' if v > edg.loc[BASELINE, 'mean_lpd']
                              else '#c44e52' for v in ev])
    a.axhline(edg.loc[BASELINE, 'mean_lpd'], color='crimson', ls='--', lw=1.2,
              label=f'{BASELINE} baseline')
    for i, v in enumerate(ev):
        a.text(i, max(v, -14.5) + 0.4, f'{v:.1f}', ha='center', fontsize=7)
    a.set_ylim(-15, 0)
    a.set_xticks(xs); a.set_xticklabels(order, rotation=30, ha='right')
    a.set_ylabel('mean held-out log predictive density  [nats / stop]')
    a.set_title('(c) extrapolation stress test "edges": train on the\n'
                'middle 40, hold out 5 slowest + 5 fastest '
                '(bars clipped at -15)')
    a.legend(fontsize=8); a.grid(alpha=.3, axis='y')

    a = ax[1, 0]
    for c, col, mk in THREE:
        xo, yo, mu, sd, z, lp = arrays(c)
        a.plot(xo, z, mk, ms=5, color=col, alpha=.8,
               label=f'{c}  (LOO z-sd low-x {loo.loc[c, "z_sd_lowx"]:.2f} / '
                     f'high-x {loo.loc[c, "z_sd_highx"]:.2f})')
    for v in (-1.96, 1.96):
        a.axhline(v, color='k', ls=':', lw=1)
    a.axhline(0, color='k', lw=.8)
    a.set_xlabel('velocity x  [m/s]')
    a.set_ylabel('held-out z = (y - mu)/sd')
    a.set_title('(d) calibration vs velocity, leave-one-out\n'
                'constant sd is over-wide at low x and too narrow at high x')
    a.legend(fontsize=8, loc='lower left'); a.grid(alpha=.3)

    a = ax[1, 1]
    for c, col, mk in THREE:
        xo, yo, mu, sd, z, lp = arrays(c)
        pit = np.sort(norm.cdf(z))
        a.plot(np.linspace(0, 1, len(pit) + 2)[1:-1], pit, '-', marker=mk,
               ms=4, color=col, alpha=.85,
               label=f'{c}  (PIT KS = {loo.loc[c, "pit_ks"]:.3f})')
    a.plot([0, 1], [0, 1], 'k--', lw=1, label='exactly calibrated')
    a.set_xlabel('uniform quantile'); a.set_ylabel('sorted held-out PIT')
    a.set_title('(e) PIT QQ plot of the 50 leave-one-out predictions')
    a.legend(fontsize=8); a.grid(alpha=.3)

    a = ax[1, 2]
    for c, col, mk in THREE:
        xo, yo, mu, sd, z, lp = arrays(c)
        a.plot(xo, sd, '-', marker=mk, ms=4, color=col, alpha=.85,
               label=f'{c}: held-out sd(x)')
    xo, yo, mu, sd, z, lp = arrays('phys')
    a.plot(xo, np.abs(yo - mu), 'kx', ms=6, alpha=.6,
           label='|held-out residual| (phys)')
    a.set_xlabel('velocity x  [m/s]')
    a.set_ylabel('predictive sd at a stop the fit never saw  [m]')
    a.set_title('(f) what the fix actually changes: sd(x)\n'
                'one flat band vs a band that grows with velocity')
    a.legend(fontsize=8, loc='upper left'); a.grid(alpha=.3)

    fig.suptitle('Out-of-sample selection, car stopping distance, n = 50   '
                 f'(LOO leader: {loo["mean_lpd"].idxmax()};  '
                 f'repeated-5-fold leader: {skf["mean_lpd"].idxmax()};  '
                 f'extrapolation leader: {edg["mean_lpd"].idxmax()};  '
                 f'baseline: {BASELINE})', fontsize=12)
    fig.tight_layout(rect=[0, 0, 1, 0.97])
    fig.savefig('figures/selection.png', dpi=150)
    plt.close(fig)


if __name__ == '__main__':
    main()
