A car at speed x (m/s) stops after y = z*x + 0.1*x**2 meters; the reaction time z ~ N(1.5, 0.5^2) differs per driver.
With f3dasm: sample 30 speeds in [3, 83] with the Sobol sampler (seed 0) and compute y. The 30 experiments are 30
different drivers, each with its own z, drawn with numpy's default_rng(0). Store x and y (z is hidden, not a column)
as an ExperimentData in the folder data/. Fit a polynomial of degree 2 to y with scikit-learn, and write predict.py,
next to this file, with predict(x) returning the predicted y. End with a report of at most three short bullet points.
