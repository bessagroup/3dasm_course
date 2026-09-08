Fixed and verified. Summary:

**Diagnosis** (from the actual residuals, not just the report's claim): scatter grows with velocity — under 15 m at x<20 m/s, up to 110 m at x>55 m/s. A constant sd over-covers slow stops and under-covers fast ones.

**Fix** — `heteroscedastic_model.py`: keeps the same quadratic mean, replaces the single noise sd with `sigma(x) = exp(p0 + p1·x)` fit by MLE on the residuals (positive by construction). Result: sd grows from 5.4 m at x=3 to 69.4 m at x=82, vs. a flat 29.9 m before.

**Proof it's better on unseen stops** — leave-one-out CV (refit on the other 49, score the held-out one by Gaussian NLL):
- mean NLL: 4.976 (constant) vs **4.488 (heteroscedastic)**
- wins on 33/50 held-out stops, in both the low- and high-velocity halves
- Wilcoxon signed-rank p = 0.0037 (significant)
- Biggest wins are exactly the fast, high-scatter stops the constant model choked on (NLL 16.5→7.9 and 9.0→6.0)

Everything (both full-data fit and per-stop LOO scores) is written into `data/` as new columns with `_source_hetero`/`_source_loocv` provenance; plots are `heteroscedastic_fit.png` and `loocv_comparison.png`. Confirmed the script is deterministic and idempotent (byte-identical `output.csv` on rerun). One command reproduces it all: `python heteroscedastic_model.py`.
