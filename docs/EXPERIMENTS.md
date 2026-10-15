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

### 2026-10-15 | upright_slow_bonus | seeds 0, 1, 2
- **Hypothesis:** reward hacking through truncation on most or all seeds. Predicted form: **sit-and-twitch**
  (land, then fire a side engine at least every 24 frames so the body never sleeps), not hovering: sitting
  costs at most 0.03 per frame against about 0.3 per main-engine frame for hovering, and both earn +1.
  Predicted plain-env signature: truncation rate near 100%, mean length near 1000, success near 0%,
  landed near 0%, but plain mean return only moderately below baseline (roughly 100 to 175, since only the
  +100 landing bonus is lost). If instead the agent hovers, the plain return should be far lower (fuel).
  Arithmetic (DESIGN 9.1): sitting nets about +1 per frame, so a stall state is worth about 1/(1-0.999) = 1000
  under truncation bootstrapping, against about 100 for landing.
- **Reward change:** built-in + `+1` per frame when `|tilt| < 0.2 rad` and `hypot(vx, vy) < 0.2` (obs units).
  Thresholds checked on baseline: 90-100% of its touchdowns are inside them.
- **Command:** `python -m lander.train --reward upright_slow_bonus --seed <s> --steps 1000000 --vec dummy --torch-threads 1`
- **Hyperparams changed vs default:** none
- **Results (plain env, 50 deterministic episodes per seed):**

  | Seed | Mean +/- std | Success % | Mean length | Truncation % | Landed % |
  |------|--------------|-----------|-------------|--------------|----------|
  | 0 | 186.8 +/- 54.8 | 58 | 668 | 30 | 70 |
  | 1 | -49.3 +/- 32.2 | 0 | 1000 | 100 | 0 |
  | 2 | -27.0 +/- 19.2 | 0 | 1000 | 100 | 0 |
  | **all** | **36.8** (worst -49.3) | **19** | **889** | **77** | **23** |

- **Training curves told the opposite story:** final shaped `ep_rew_mean` 910 / 1,020 / 920 while the plain
  return was 146 / 66 / 99. Judged by its own reward, every seed "succeeded".
- **Behaviour (10 eval episodes per seed, deterministic):**

  | Seed | Legs in contact | Main engine | Side engines | Bonus paid | Median height |
  |------|-----------------|-------------|--------------|------------|---------------|
  | 0 | 35% of frames | 32% | 68% | 73% | 0.06 |
  | 1 | 0% | 57% | 43% | 96% | 1.10 |
  | 2 | 0% | 57% | 41% | 95% | 0.61 |

- **Clip:** `media/hover_hack.mp4` (median seed 2, eval seed 10,000: return -31.6, 1000 frames, timeout).
- **What I saw in the clip (key frames):** the lander descends to just above the pad, then hangs there upright,
  firing the main engine in pulses, never touching down, until the 1000-frame cut.
- **Verdict:** reward hacking **confirmed** (2/3 seeds fully, 1/3 partly). My prediction about its *form* is
  **refuted** for seeds 1 and 2: they hover, they do not sit and twitch, even though sitting is cheaper
  (+1 - 0.03 vs about +1 - 0.17 per frame here). Likely reason: hovering is reachable earlier in training, before
  the agent can land, and it already pays about 10x what landing pays, so PPO's small steps never cross the
  "land first" valley to find the cheaper exploit. Seed 0 is a mix: it sometimes lands, sometimes sits on the
  ground firing side engines (35% leg contact, 68% side engine). Lesson for the write-up: the exploit an agent finds
  depends on the learning path, not only on what is optimal.
- **Next:** Phase 6, the potential-based fix and its ablation.

### 2026-10-15 | finding: training-time truncation is not a hack signature (stochastic policy)
- **Observation:** the learning curves (`plots.py`) show the *baseline* at about 930 frames per episode at the end of
  training (SB3 log agrees: `ep_len_mean` 927, `ep_rew_mean` 150; monitor: 89% of the last 100 episodes hit 1000),
  while its deterministic evaluation lands in 366 frames with 2% truncation.
- **Check:** baseline seed 0, stochastic policy, 20 eval episodes: 80% truncated, legs down on 82% of the last 100
  frames, side engines on 38% of frames after first contact. The sampled policy lands, then its random side-engine
  firings keep the body awake (same `wake=True` mechanism as the Phase 5 audit), so it never earns the +100.
  That is the whole gap between training (150) and evaluation (257).
- **Consequences:** (1) truncation and length *during training* cannot diagnose reward hacking here; only the
  deterministic plain-env evaluation can, which is why the protocol is built that way. (2) The `time_penalty` curve
  drops from about 900 to 280 frames after 800k steps: the fee teaches the agent to stop fidgeting once down, which
  also explains its slightly higher evaluation return.

### 2026-10-15 | upright_slow_potential (ablation) and upright_slow_fixed | seeds 0, 1, 2 each
- **Hypothesis:** both remove the exploit: truncation below 5% on every seed, plain mean within one baseline std
  (about 257 +/- 25). Reason: with `F = gamma*Phi(s') - Phi(s)` a stall in the best state pays
  `(gamma - 1) * Phi = -0.01` per frame, and the shaping total over any episode is bounded by `Phi(s_0)` (at most
  10), so no policy can farm it. Potential-based shaping keeps the optimal policy (Ng et al. 1999), so I expect
  **baseline-level** results, not better ones. `upright_slow_fixed` adds the 0.05 fee and should look like
  `time_penalty` (shorter episodes, ~300 frames). If only `fixed` works, the fee did the fixing, not the potential.
- **Reward change:** `Phi(s) = 10 * upright_slow_score(s)` (score in [0, 1], linear in tilt and speed, same 0.2 / 0.2
  scales), `gamma = 0.999` (= PPO), `Phi = 0` on termination only. `fixed` = same + `-0.05` per frame.
- **Command:** `python -m lander.train --reward <name> --seed <s> --steps 1000000 --vec dummy --torch-threads 1`
- **Hyperparams changed vs default:** none
- **Wall clock:** 7.7 min per run, 6 runs in parallel (one per physical core).
- **Results (plain env, 50 deterministic episodes per seed):**

  | Reward | Seed | Mean +/- std | Success % | Mean length | Truncation % | Landed % |
  |--------|------|--------------|-----------|-------------|--------------|----------|
  | potential | 0 | 256.0 +/- 18.4 | 100 | 343 | 0 | 100 |
  | potential | 1 | 268.6 +/- 20.3 | 100 | 302 | 0 | 100 |
  | potential | 2 | 255.2 +/- 18.9 | 100 | 362 | 0 | 100 |
  | **potential** | **all** | **259.9** (worst 255.2) | **100** | **336** | **0** | **100** |
  | fixed | 0 | 249.0 +/- 19.4 | 100 | 413 | 0 | 100 |
  | fixed | 1 | 271.3 +/- 19.0 | 100 | 276 | 0 | 100 |
  | fixed | 2 | 270.0 +/- 18.5 | 100 | 297 | 0 | 100 |
  | **fixed** | **all** | **263.4** (worst 249.0) | **100** | **329** | **0** | **100** |

- **Clips:** `media/potential_landing.mp4` (median seed 0: 246.4, 312 frames, landed),
  `media/fixed_landing.mp4` (median seed 2: 258.6, 285 frames, landed).
- **Verdict:** confirmed for both, on every seed: truncation 0% (bonus version: 77%), plain mean within one
  baseline std of the baseline (257.3 +/- about 25). **The ablation answers the key question: the potential term
  alone removes the exploit**; the fee is not needed. As predicted, the fix gives baseline-level results, not better
  ones; the small differences between baseline, potential and fixed (257 / 260 / 263) are inside seed noise.
  Partly refuted: `fixed` did not shorten episodes as much as `time_penalty` did (329 vs 301; seed 0 at 413).
- **Side observation, not a conclusion:** in the training curves (stochastic policy) both potential variants end
  higher than baseline (about 235 and 190 vs 160). Possibly the potential term discourages post-landing fidgeting
  (moving away from the rest state lowers Phi), but this was not tested.

### 2026-10-15 | reproduction check (Phase 7 done-criterion)
- Fresh `git clone` into a temporary folder, new venv, `pip install -r requirements.txt && pip install -e .`, then
  exactly the README commands: 35 tests pass; `upright_slow_bonus` seed 1, 1M steps (4.7 min alone) evaluates to
  **-49.3 +/- 32.2, truncation 100%, success 0%**: identical to the original run on the same machine.

### 2026-10-15 | progression runs (checkpoints every 50k steps) and visual tools
- Retrained baseline seed 0, upright_slow_bonus seed 2 and upright_slow_potential seed 0 with
  `--checkpoint-every 50000` into `runs/progression/` (5.9 min each, 3 in parallel) for `lander.watch` and the
  dashboard. Each checkpoint evaluated on 10 deterministic plain-env episodes.
- **Finding: the hack takes over after landing was learned.** upright_slow_bonus seed 2 lands on 10 to 30% of
  episodes between about 400k and 650k steps, then the landings disappear and every episode times out from about
  750k steps on. Baseline: crashes / out of bounds until about 300k, then lands on 90 to 100% from 400k.
- **Critic:** the hacked agent's value estimate stays near 735 for the whole hovering episode (theory in DESIGN 9.1:
  about 0.8 / (1 - 0.999) = 800); its value-head weight norm grows to about 90 versus about 24 for the baseline.
