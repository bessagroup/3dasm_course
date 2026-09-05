#!/usr/bin/env bash
# Put the folder back to the pre-run state, so the demo can be run again.
#
#   ./reset.sh
#
# Kept:     make_data.py, baseline.py, CLAUDE.md, README.md, .claude/,
#           templates/, the transcripts, and the training record data/
#           (deleted and rebuilt from its fixed seeds, so it comes back
#           byte-for-byte identical).
# Removed:  everything an agent wrote -- data_test/, blocks/*, the study
#           records, model.png and selection.png, stray scripts -- and
#           pipeline.ipynb is restored from templates/.
set -euo pipefail
cd "$(dirname "$0")"

PYTHON="${PYTHON:-python}"
export MPLBACKEND=Agg

echo "reset: removing agent-written artefacts"
rm -rf data data_test study_selection blocks/__pycache__ __pycache__ .ipynb_checkpoints
find blocks -mindepth 1 ! -name '.gitkeep' -delete 2>/dev/null || true
find figures -mindepth 1 ! -name '.gitkeep' -delete 2>/dev/null || true
# top-level scripts an agent may have written (the four originals are kept)
for f in *.py; do
  case "$f" in
    make_data.py|baseline.py) ;;
    *) echo "  rm $f"; rm -f "$f" ;;
  esac
done
rm -rf data_*/ 2>/dev/null || true

echo "reset: restoring pipeline.ipynb from templates/"
cp templates/pipeline.ipynb pipeline.ipynb

echo "reset: rebuilding the training record and the baseline"
"$PYTHON" make_data.py
"$PYTHON" baseline.py

echo
echo "reset: done. The folder is back to the pre-run state:"
echo "  data/ (60 training points + baseline columns), figures/baseline.png,"
echo "  blocks/ empty, no held-out record, pipeline.ipynb is the template."
