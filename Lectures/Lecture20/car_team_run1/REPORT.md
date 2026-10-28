Both models fitted and independently checked; overall check PASS.

Results (workspace/results/, 37 train / 13 test, split train_test_split(random_state=1987)):
- M1 (y = a*x + b*x², sd = s): a=1.556793, b=0.100360, s=27.983031, coverage=0.846 (11/13), nll=4.976150
- M2 (y/x = a + b*x, mean a*x + b*x², sd = c*x): a=1.380403, b=0.103235, c=0.553231, coverage=0.846 (11/13), nll=4.503771

Deliverables: fit_M1.py, M1.csv, fit_M2.py, M2.csv, check.md. The checker (D003) recomputed everything from scratch, with no import of the fit scripts, and matched all stored values at relative tolerance 1e-6, plus split sizes and column schema.

Caveats: M2 was read as a = intercept and b = slope of the y/x regression, so the mean is a*x + b*x², as the problem statement specifies. The stored M2 numbers match this reading. The 13-row test set gives coverage resolution of about 0.077, so the 0.846 coverage is consistent with nominal 0.95 but not a strong test. No ground-truth evaluations were used (0). The setup notice about a canonical evaluator was not applicable, since this is an analytic fit.