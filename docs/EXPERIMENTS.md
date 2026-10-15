# Experiment log

Newest entry at the bottom. One entry per training run (or per batch of seeds).
Write the hypothesis **before** looking at the results.

## Environment
- OS: Windows
- Python: 3.12.10 (venv created with `py -3.12`; system default is 3.14). 3.12.0 breaks `PPO.load`, see baseline entry.
- gymnasium: 1.3.0            stable-baselines3: 2.9.0            torch: 2.14.1+cpu
- numpy 2.5.4, moviepy 2.1.2, matplotlib 3.11.2, pandas 3.0.6
- CPU: Intel Core i7-10750H @ 2.60GHz (6 cores, 12 threads), no GPU used
- 1M steps wall-clock time: 6.0 min (DummyVecEnv, 1 torch thread, 3 seeds in parallel)

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
- **Results (plain env, 50 deterministic episodes per seed, eval seeds 10,000 to 10,049):**

  | Seed | Mean +/- std | Success % | Mean length | Truncation % | Landed % |
  |------|--------------|-----------|-------------|--------------|----------|
  | 0 | 252.6 +/- 33.6 | 96 | 397 | 4 | 96 |
  | 1 | 259.7 +/- 23.5 | 98 | 346 | 2 | 98 |
  | 2 | 259.6 +/- 19.3 | 100 | 354 | 0 | 100 |
  | **all** | **257.3** (worst 252.6) | **98** | **366** | **2** | **98** |

- **Wall clock:** 6.0 min per seed, 3 seeds in parallel (about 2,820 steps/s each).
- **Clip:** `media/baseline_landing.mp4` (median seed 2, eval seed 10,000: return 247.5, 354 frames, landed).
- **What I saw in the clip (key frames):** steady upright descent, small main-engine pulses, settles between the flags.
- **Verdict:** confirmed. Solved on every seed. Training-time `ep_rew_mean` was only 151 to 165: it is the
  stochastic policy; the deterministic policy is about 100 points better. The few truncations (seed 0: 2 of 50)
  are the reference level for later comparisons.
- **Incident:** `PPO.load` failed ("PytorchStreamReader failed reading file"). The saved files were fine
  (loading from bytes worked); reading a zip member through `zipfile.ZipExtFile` failed under **Python 3.12.0**.
  Upgraded the base interpreter to 3.12.10 (`winget upgrade Python.Python.3.12`); loading works. Models trained
  under 3.12.0 load fine under 3.12.10.
- **Next:** time_penalty, 3 seeds.
