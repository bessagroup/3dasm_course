"""Reproduce everything: both models, every stored column, all three figures.

    python run_all.py

Runs, in order:

  1. ``run_quadratic.py``  -- the quadratic mean with one constant noise sd
                             (columns ``*_quad``, figure 1)
  2. ``run_noise_study.py`` -- the noise forms scored on stops they did not
                             see, the winner stored (columns ``*_prop``,
                             ``*_loo_*``, ``study_noise/``, figures 2 and 3)

Both steps overwrite the columns they own, so this is safe to run on a record
that already carries them, and safe to run twice.
"""

import run_noise_study
import run_quadratic

STEPS = (('the constant-noise model', run_quadratic),
         ('the noise study, scored out of sample', run_noise_study))


def main():
    for n, (label, module) in enumerate(STEPS, start=1):
        rule = '=' * 74
        print(f'\n{rule}\n== step {n} of {len(STEPS)}: {label}\n{rule}')
        module.main()


if __name__ == '__main__':
    main()
