Both models were fit as specified in PROBLEM_STATEMENT.md, and an independent recomputation by the checker agreed with both.

| model | a | b | sd_parameter | coverage | nll | n_train | n_test |
|---|---|---|---|---|---|---|---|
| M1 | 1.5567927444618306 | 0.10035968486568629 | s = 27.983030806957824 | 0.8461538461538461 | 4.976149641475809 | 37 | 13 |
| M2 | 1.3804030477145794 | 0.10323542514078463 | c = 0.5532310661868639 | 0.8461538461538461 | 4.503770939937339 | 37 | 13 |

Checks:
- M1: PASS. The checker's independent recomputation matched every CSV field at relative tolerance 1e-6 (difference 0).
- M2: PASS. Same result.
- Both CSVs have the exact header model,a,b,sd_parameter,coverage,nll,n_train,n_test, with one data row each.

Deliverables are in workspace/results/: fit_M1.py, M1.csv, fit_M2.py, M2.csv, check.md. The checker overwrote a check.md that already existed there.

Caveats:
- The builder reported that fit_M1.py, M1.csv, fit_M2.py and M2.csv already existed before its run. It re-ran both scripts, which reproduced identical values. Its reports also referred to an "earlier reply" that is not in this session's record, so I relied on the re-runs and the checker's recomputation rather than on that earlier output.
- No evaluator was used (0 evals). This is an analytic study with no ground-truth source registered, so the setup notice did not apply.
- Nothing about the decisions was changed. Coverage and NLL come from only 13 test rows, so they are coarse; coverage is 11/13.