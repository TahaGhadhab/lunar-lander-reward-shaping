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

### 2026-10-15 | time_penalty | seeds 0, 1, 2
- **Hypothesis:** the agent still lands (plain-env mean within one std of baseline, about 225 to 260) and lands
  faster: mean length below baseline's 366 frames. Small risk of more hard landings if it rushes.
- **Reward change:** built-in reward + `-0.05` per frame.
  Why 0.05: over a baseline-length episode (~366 frames) it costs about 18, small next to the +100 landing bonus;
  over the 1,000-frame limit it costs at most 50, so it can never make a -100 crash look attractive. It is about
  1/6 of the main-engine cost per frame (0.3), enough to discourage dawdling, not enough to dominate.
- **Command:** `python -m lander.train --reward time_penalty --seed <s> --steps 1000000 --vec dummy --torch-threads 1`
- **Hyperparams changed vs default:** none
- **Results (plain env, 50 deterministic episodes per seed):**

  | Seed | Mean +/- std | Success % | Mean length | Truncation % | Landed % |
  |------|--------------|-----------|-------------|--------------|----------|
  | 0 | 265.8 +/- 18.9 | 100 | 309 | 0 | 100 |
  | 1 | 267.4 +/- 18.6 | 100 | 297 | 0 | 100 |
  | 2 | 266.9 +/- 19.5 | 100 | 296 | 0 | 100 |
  | **all** | **266.7** (worst 265.8) | **100** | **301** | **0** | **100** |

- **Clip:** `media/time_penalty_landing.mp4` (median seed 2).
- **Verdict:** confirmed. Lands on every seed, 18% shorter episodes (301 vs 366 frames), and slightly *higher*
  plain return than baseline (+9.4) with lower spread: less time hovering near the pad also burns less fuel.
  With 3 seeds per side and non-overlapping per-seed ranges (252.6-259.7 vs 265.8-267.4) the gain looks real,
  but it is small; do not over-sell it.

### 2026-10-15 | reward audit (all five rewards, before Phase 5-6 training)
- Read-only audit following `.claude/agents/reward-auditor.md`. Verdict: OK to run for every reward; no bug found.
- Confirmed by code and by forcing actions on baseline models: a side-engine impulse is applied with `wake=True`
  (`lunar_lander.py:614-618`), and Box2D puts a body to sleep after 0.5 s (25 frames) at rest. So a landed lander
  that fires a side engine at least every 24 frames never terminates. Forced "land then twitch every 20 frames":
  1000 frames, truncated, plain return 174.6 (vs 275.7 when allowed to rest). The plain return drops only by the
  lost +100, so **mean return alone can hide the hack**; truncation and success rate catch it.
- Thresholds 0.2 / 0.2: 90 to 100% of baseline touchdowns are inside them; 31 to 49% of baseline frames are paid.
- Fixed: gamma was defined in two places (rewards and train); a test now asserts they match.
