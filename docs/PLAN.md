# Plan

Each phase ends with evidence shown to me. Tick a box only once its done-criteria are met.
Times are estimates on a laptop CPU. Technical design and the reasons behind the amendments: `DESIGN.md`.

**Amendments adopted (2026-10-15, see `DESIGN.md` section 10):**
A venv created with `py -3.12` (default `python` on this machine is 3.14) ·
B `Monitor` sits on the plain env, inside the reward wrapper, so curves show the plain return; the shaped return is logged separately ·
C pass-through and identity tests written together with the base wrapper ·
D dry run compares `SubprocVecEnv` and `DummyVecEnv` ·
E every reward, baseline included, gets 3 seeds ·
F Phase 5 succeeds on any truncation-driven exploit (hover **or** sit-and-fidget) ·
G ablation `upright_slow_potential` (potential term without fee) ·
H the fix is judged against the baseline, with the same step budget for every reward ·
I `record.py` is tried on the dry-run model.

## Tonight (about 3 to 4 h): phases 0 to 4

### [x] Phase 0: Setup (up to 45 min)
- Create `.venv` with `py -3.12 -m venv .venv` (amendment A), install the ranges from `docs/SETUP.md` plus `matplotlib` and `pandas`
  (SB3 2.9 no longer installs them, and `plots.py` imports them).
- Write `requirements.txt`, a minimal `pyproject.toml` (`src/` layout, ruff config), `scripts/smoke_test.py`, then make the first commit.
- **Done when:** `scripts/smoke_test.py` prints `SMOKE TEST OK` (imports gymnasium, stable_baselines3, torch and Box2D,
  creates `LunarLander-v3`, steps it once) and the versions are copied into `docs/EXPERIMENTS.md`.

### [x] Phase 1: Watch it fail (about 20 min)
- `src/lander/envs.py` with `make_env`, and `scripts/random_agent.py` (optional game window with `--render`).
- Print the 8 observation numbers for a few frames and match them to the table in `BRIEF.md`.
- **Done when:** the mean return of 20 random episodes is logged (expect well below 0).

### [x] Phase 2: Train the baseline (30 to 45 min of waiting)
- `rewards.py` (base `ShapingWrapper`, registry, `baseline`) **with its pass-through and identity tests** (amendment C).
- `train.py`, `config.json` writing, Monitor logs on the plain env (amendment B).
- **50k-step dry run** with `SubprocVecEnv` and with `DummyVecEnv`; keep the faster and log steps/s (amendment D).
- Then 1M steps on seeds 0, 1, 2 (amendment E).
- **Done when:** the plain return in `monitor.csv` climbs on all seeds, models are saved and the wall-clock time is noted.

### [x] Phase 3: Evaluate and record (about 30 min)
- `evaluate.py` (plain env, 50 deterministic episodes, seeds 10,000 to 10,049, writes `eval.json` with outcome counts)
  and `record.py` (MP4). Try `record.py` on the dry-run model first (amendment I).
- **Done when:** mean return across the 3 seeds is at least 200 and `media/baseline_landing.mp4` (median seed) exists.
  If a seed is below 200, report it and propose one change (more steps first, for **every** reward: amendment H).

### [x] Phase 4: Tests and a first shaped reward (about 45 min plus training)
- `tests/`: shaping math on hand-made transitions, potential telescoping, registry, env smoke test, eval summary.
- Reward `time_penalty`: a small fee per frame (-0.05, justified in `EXPERIMENTS.md`). 3 seeds.
- **Done when:** tests pass and the agent still lands (plain-env eval close to baseline) with a shorter mean episode length.
- Out of time? Start the training and evaluate tomorrow.

## Tomorrow (about 4 to 6 h): phases 5 to 7

### [x] Phase 5: Break it on purpose
- Reward `upright_slow_bonus`: +1 per frame when |tilt| and speed are under thresholds (write the thresholds in the log).
- Hypothesis to write first: the cheapest exploit is probably to sit on the ground and twitch the side engines
  (0.03 per frame) rather than hover (about 0.3 per main-engine frame). Retrain with 3 seeds and watch the clips.
- **Done when:** at least one seed shows a **truncation-driven exploit** (amendment F): high truncation rate,
  mean length near 1000, success near 0 on the plain env, plus `media/hack.mp4` named by what it shows.
  If no seed exploits it, try a looser threshold or 2M steps before concluding.

### [x] Phase 6: Fix it
- Reward `upright_slow_potential`: `F = gamma*Phi(s') - Phi(s)` on the same "upright and slow" score, no fee (amendment G).
- Reward `upright_slow_fixed`: the same plus a per-frame fee.
- Run the `reward-auditor` agent on `rewards.py` first and address its findings. 3 seeds each.
- **Done when** (amendment H), for both: truncation rate below 5% on every seed, plain-env mean within one baseline
  std of the baseline mean, and `media/fixed_landing.mp4` exists.

### [ ] Phase 7: Write-up
- `plots.py`: learning curves (mean and std over seeds) per reward, plain return and episode length, saved to `media/*.png`.
- README: problem, method, results table, clips, what I learned.
- **Done when:** a stranger can reproduce one result from the README alone.

## Results table (fill as you go; 3 seeds each, mean of per-seed means, worst seed in brackets)
| Reward | Seeds | Mean return (plain env) | Success % | Mean ep. length | Truncation % | Clip |
|--------|-------|-------------------------|-----------|-----------------|--------------|------|
| baseline | 0,1,2 | 257.3 [252.6] | 98 | 366 | 2 | media/baseline_landing.mp4 |
| time_penalty | 0,1,2 | 266.7 [265.8] | 100 | 301 | 0 | media/time_penalty_landing.mp4 |
| upright_slow_bonus | 0,1,2 | 36.8 [-49.3] | 19 | 889 | 77 | media/hover_hack.mp4 |
| upright_slow_potential | 0,1,2 | 259.9 [255.2] | 100 | 336 | 0 | media/potential_landing.mp4 |
| upright_slow_fixed | 0,1,2 | 263.4 [249.0] | 100 | 329 | 0 | media/fixed_landing.mp4 |
