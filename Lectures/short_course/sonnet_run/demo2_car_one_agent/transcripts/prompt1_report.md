The fit looks reasonable — the constant-sd band is clearly too narrow at low x and too wide at high x, since scatter visibly grows with velocity, but that's an honest depiction of what this model assumes, not a bug.

Done. `quadratic_model.py` fits `y = a*x² + b*x + c` by OLS, estimates one constant noise sd from the residuals (47 dof), and writes `y_pred_quad`, `sd_quad`, `_source_quad` into `data/`. Fitted result: `y = 0.100388*x^2 + 1.44641*x - 0.656207`, sigma = 29.9218. Plot saved to `quadratic_fit.png`. Reproduce everything with:

```
python quadratic_model.py
```
