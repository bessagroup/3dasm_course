# Find the highest y, with a strategy written as code

An experiment returns one number y for an input x = (x1, x2, x3), each in [0, 1]. Its formula is unknown, and each
experiment is expensive. The goal: the input with the highest y.

- 15 experiments were already run (a Latin hypercube): `record.csv`.
- **Your method** (assigned): a linear-regression surrogate optimized with L-BFGS: fit a linear model on basis functions of your choice (e.g. polynomials) to all the experiments, find its maximum with scipy.optimize.minimize (method='L-BFGS-B', bounds [0, 1]), probe that point, and repeat. You choose the basis, its degree and the regularization. Stay with this method: you may change only its settings and the structure of its model.
- **The result is the best experiment actually run.** A model's predicted maximum counts only after you probe it, and that probe is one of the 15.
- Write it as a Python function in `strategy.py`, next to this file:

      def strategy(probe, X0, y0, budget, seed):
          # probe(x) runs one experiment at x (a numpy array of 3 numbers in [0, 1]) and returns its y.
          # X0 (15 x 3) and y0 (15,) are the experiments already run; budget = 15 probes; seed: for any randomness.
          # Return nothing: every call to probe is recorded.

  It must use only numpy, scipy and scikit-learn, call probe at most `budget` times, and finish in under 2 minutes.
- Test it as much as you like on functions of your own. Then call the tool `RunStrategy` **once**: it runs your
  strategy on the real experiment with seed 0 and adds the 15 probes to `record.csv`. It cannot be run twice.
- Do not search the web yourself. End with a short report: your strategy, its settings, and its best experiment.
