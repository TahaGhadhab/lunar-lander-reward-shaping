# Experiment log

Newest entry at the bottom. One entry per training run (or per batch of seeds).
Write the hypothesis **before** looking at the results.

## Environment
- OS: Windows
- Python: 3.12.0 (venv created with `py -3.12`; system default is 3.14)
- gymnasium: 1.3.0            stable-baselines3: 2.9.0            torch: 2.14.1+cpu
- numpy 2.5.4, moviepy 2.1.2, matplotlib 3.11.2, pandas 3.0.6
- CPU: Intel Core i7-10750H @ 2.60GHz (6 cores, 12 threads), no GPU used
- 1M steps wall-clock time:

---

## Template
### YYYY-MM-DD | <reward name> | seeds <list>
- **Hypothesis:** what I expect and why.
- **Reward change:** formula and constants (thresholds, fees).
- **Command:** `python -m lander.train ...`
- **Hyperparams changed vs default:** none, or which and why
- **Results (plain env, 50 episodes per seed):**

  | Seed | Mean +/- std | Success % | Mean length | Truncation % |
  |------|--------------|-----------|-------------|--------------|

- **Clip:** media/...
- **What I saw in the clip:**
- **Verdict:** confirmed, refuted or inconclusive (and why)
- **Next:**
