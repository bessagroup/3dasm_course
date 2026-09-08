#!/usr/bin/env bash
# Reproduce the whole car stopping-distance study, in the order it was built.
#
#     bash reproduce.sh
#
# Three stages, each idempotent, each owning its own columns of the record:
#
#   1. run_quadratic.py     baseline: quadratic mean, ONE constant sd
#                           -> y_pred_quad, sd_quad, _source_quad
#                           -> figures/quadratic_constant_noise.png
#   2. run_noise_models.py  the fix (modeler): eight Gaussian candidates whose
#                           sd is allowed to vary with x, joint-MLE fitted
#                           -> y_pred_phys, sd_phys, _source_phys
#                           -> figures/model.png
#   3. study_selection.py   the judgement (selector): every candidate refitted
#                           from scratch inside every training split, scored
#                           only on stops that fit never saw
#                           -> study_selection/ (the scoring record)
#                           -> y_loo_*, sd_loo_*, lpd_loo_*, _source_selector
#                           -> figures/selection.png
#
# NOTE on the interpreter: `python` on PATH here has no f3dasm at all, so the
# interpreter is named explicitly. Override it if yours lives elsewhere:
#
#     PYTHON=/path/to/python bash reproduce.sh

set -euo pipefail
cd "$(dirname "$0")"

PYTHON="${PYTHON:-/home/mbessa/miniforge3/envs/aescape_course/bin/python}"

"$PYTHON" -c "import importlib.metadata as m; print('python:', __import__('sys').executable); print('f3dasm', m.version('f3dasm'))"

for stage in run_quadratic.py run_noise_models.py study_selection.py; do
    echo
    echo "=============================================================="
    echo "== $stage"
    echo "=============================================================="
    "$PYTHON" "$stage"
done

echo
echo "done. figures/quadratic_constant_noise.png, figures/model.png, figures/selection.png"
