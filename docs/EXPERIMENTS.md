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

---

### 2026-10-15 | throughput dry runs (baseline, 50k steps, seed 0)
- **Purpose:** pick the vectorised env type and PyTorch thread count (amendment D). Not a reward experiment.
- **Results:** `SubprocVecEnv` 3,144 steps/s, `DummyVecEnv` 3,135 steps/s, both ending at `ep_rew_mean` -111
  (identical: seeding is reproducible across env types). `DummyVecEnv` + `torch_threads=1`: 3,892 steps/s;
  `torch_threads=4`: 3,532 steps/s. The PPO update (tiny MLP, batch 64) dominates; thread overhead hurts.
- **Decision:** all runs use `--vec dummy --torch-threads 1`. Each run then uses one core, so the 3 seeds of a
  reward run in parallel (6 physical cores). No PPO hyperparameter changes. Expected: about 5 min per 1M-step run.

### 2026-10-15 | baseline | seeds 0, 1, 2
- **Hypothesis:** with the course hyperparameters and 1M steps, PPO lands: plain-env deterministic mean around
  200 to 260 per seed, truncation near 0%, mean episode length 200 to 400 frames. One seed may fall short of 200.
- **Reward change:** none (built-in reward only).
- **Command:** `python -m lander.train --reward baseline --seed <s> --steps 1000000 --vec dummy --torch-threads 1`
- **Hyperparams changed vs default:** none
