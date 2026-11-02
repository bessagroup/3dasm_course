# Find the highest y

An experiment returns one number y for an input x = (x1, x2, x3), each in [0, 1]. Its formula is unknown, and each
experiment is expensive. The goal: the input with the **highest y** that you can find.

- 15 experiments were already run: a Latin hypercube, in `record.csv` (agent "LHS", round 0), and the same
  experiments as an f3dasm ExperimentData in `experiments/`.
- **Budget: 15 more experiments**, run only through the probe tools. The team has 3 agents (agent1, agent2, agent3), and each one has its own probe tool with 5 probes. The probes go in rounds: in each round every agent makes one probe, and no agent can probe again until all the others have probed in that round (5 rounds in total).
- Every probe is added to `record.csv` with who made it, in which round, its y, and the reason given for it. Every
  agent reads `record.csv`: it is how the team shares what it has learned.
- Choose your strategy freely, and change it whenever the record suggests it. Any agent may ask the reviewer (a
  literature-review agent) which methods suit the problem. Do not search the web yourself; you may write and run
  code in your own folder.
- **The result is the best experiment actually run.** A model's predicted maximum counts only after it is probed.
- When the budget is spent, write `best.json` next to this file: {"x1": ..., "x2": ..., "x3": ..., "y": ...}, the
  best experiment in `record.csv` (it must be one that was run).
