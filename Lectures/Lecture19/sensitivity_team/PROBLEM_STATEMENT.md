Which of five inputs matter, and how much?
Inputs x1 to x5, continuous and uniform in [-1, 1]. Each experiment returns one output y; its formula is unknown, and
the only way to run experiments is the tool Experiment. For each input we want two numbers: the fraction of the
variance of y that it explains alone, and the fraction in which it takes part, alone or through interactions with
other inputs. Budget: 64 experiments.
1. Method: before choosing one, ask the reviewer which methods give these two numbers with 64 experiments, and which
   it recommends, with its settings. Write method.md: the methods it found (one sentence each), its recommendation,
   and the papers it quoted (authors, year, title).
2. Experiments: 64 points from the Sobol sampler of f3dasm (seed 0), as an ExperimentData stored in the folder
   training/ next to this file. Then call Experiment("training"): it runs the open experiments and stores their
   outputs y in the same ExperimentData.
3. Analysis: follow the recommendation to estimate the two numbers for each input from these 64 experiments only, in
   the script sensitivity.py, which writes results.json with the keys "method" (text), "first_order" and "total"
   (lists of 5 numbers, x1 to x5) and "n_experiments" (the experiments it used).
End with a report of at most three short bullet points.
