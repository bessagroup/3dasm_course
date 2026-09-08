Both subagents are done. Here's what happened, in delegation order.

## Modeler

**Built:** extended `car_noise_models.py`'s existing shared interface (`fit`, `predict(x)->(mean,sd)`, `.name`, params) from three noise candidates to five, all sharing the same quadratic mean `mu(x) = 0.100388x² + 1.446409x − 0.656207`:
- `constant` — the flagged-weak homoscedastic baseline
- `powerlaw`, `proportional` — the two already in the file
- `linear` (new) — sd(x) = |c0 + c1·x|
- `additive_proportional` (new) — sd(x) = √(σ0² + (k·μ(x))²), a floor term plus a multiplicative term

**Why:** it noticed `powerlaw` and `proportional` both force sd→0 as x→0, which its own tercile diagnostic contradicts — the low-speed third of the data still has real residual scatter (std ≈ 7 m), not near-zero. It added the two new candidates specifically to give the noise model a floor at low speed.

**Numbers printed:** in-sample loglik — constant −239.33, linear −216.45, powerlaw −216.75, proportional −226.49, additive_proportional −218.57. Low/high-speed sd: additive_proportional 5.72→65.93 (matches the diagnostic's low-speed floor); powerlaw 1.34→49.18 and proportional 0.61→104.69 (both understate low-speed noise).

**Wrote to record:** picked `additive_proportional` as its provisional, unjudged pick — explicitly because its low-speed floor matches the diagnostic, not because of any fit-quality ranking (which isn't its call).

## Selector

**Did:** treated this as a fresh evaluation of all 5 candidates (didn't reuse the old 3-candidate study). Ran refit-from-scratch LOO-CV, scored held-out log predictive density, checked calibration, ran all 10 pairwise comparisons with Bonferroni correction, and a paired bootstrap for confidence.

**Numbers:** mean LOO log-density — constant **−4.987** (worst), linear −4.480, powerlaw −4.481, additive_proportional −4.511, proportional −4.604. All four fixes beat constant with bootstrap `P(fix>constant)` = 0.94–1.00. But **significance is fragile**: none of the four survive Bonferroni correction (α=0.005); linear/powerlaw are closest (raw p≈0.007–0.010), additive_proportional is borderline (p=0.049), proportional isn't significant even uncorrected (p=0.17). A single high-leverage fold (x=76.75, the fastest driver) accounts for much of constant's disadvantage.

**Direct answers it gave:** Yes, fixing constant noise helps directionally and robustly under resampling. But `additive_proportional` (the modeler's pick) is **not** shown to be the best fix — it's statistically indistinguishable from `linear` and `powerlaw` (paired p=0.71–0.72), a near three-way tie in the bootstrap (37%/29%/28% chance of being best), with `proportional` behind.

**What it says the data doesn't show:** 50 points can't cleanly separate the four heteroscedastic fixes from each other; `proportional` is best-calibrated (z-std 1.038) despite weakest LOO evidence, an unresolved tension; the mean function and Gaussian-residual assumption were never tested here; and the "constant loses" result leans partly on one high-speed observation.

## Bottom line

The fix works directionally: replacing constant noise with any heteroscedastic model is favored on stops the model never fit, at a level the bootstrap takes seriously (≥94% probability of beating constant per fix) even though no single comparison clears a strict multiple-comparisons bar with n=50. But the modeler's specific pick, `additive_proportional`, is only one of three statistically tied options (with `linear`/`powerlaw`) — selector explicitly declines to call it the winner. Neither subagent overstepped: modeler never scored against held-out data, selector never touched model code.

**To reproduce everything:**
```
python fit_noise_models.py && python draw_model_figure.py && python selector_loo_cv5.py
```
