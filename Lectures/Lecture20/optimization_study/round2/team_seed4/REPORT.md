Ran all 5 rounds (15 probes, 3 agents × 5). Best experiment actually run: row 30 (agent3, round 5), x = (0.62, 0.66, 0.86), y = 1.85816. Runner-up: row 29 (agent2, round 5), (0.62, 0.64, 0.86), y = 1.84632; row 28 (agent1, round 5), (0.66, 0.63, 0.84), y = 1.83274. LHS best before the probes was 1.677.

What each agent tried:
- agent1: round 1 (0.73, 0.65, 0.74) → 1.746; round 2 x2 up to 0.85 → 1.68; round 3 x2 down to 0.50 → 1.50; round 4 x3 up to 0.97 → 1.74; round 5 (0.66, 0.63, 0.84) → 1.833.
- agent2: round 1 (0.72, 0.65, 0.75) → 1.76 (duplicated agent3); round 2 x3 down to 0.55 → 1.56; round 3 x3 up to 0.88 → 1.806; round 4 x2 to 0.70 → 1.797; round 5 (0.62, 0.64, 0.86) → 1.846.
- agent3: round 1 (0.72, 0.65, 0.75) → 1.76 (duplicated agent2); round 2 x1 down to 0.55 → 1.7585; round 3 diagonal (0.60, 0.55, 0.85) → 1.603; round 4 x1 up to 0.90 → 1.368; round 5 (0.62, 0.66, 0.86) → 1.858.

Findings: x3 high (about 0.84–0.90) helps, with the peak below 0.97. x2 peaks near 0.63–0.67. y rose as x1 went down from 0.72 to 0.62, and x1=0.90 is strongly bad. The x1 trend was not checked below 0.62.

Caveats: the team used round 1 poorly (two agents probed the same point). The best x1 improvement came in the last round, so the optimum may lie below x1=0.62. Only one run was made in each round for each agent.

Deliverable: best.json was written with the Write tool, but that tool only writes to the delegation folder. It is at runs/20261102T021236/debug/delegations/best.json, not next to PROBLEM_STATEMENT.md. The content is {"x1": 0.62, "x2": 0.66, "x3": 0.86, "y": 1.85816}. Copy it to the study folder if needed.