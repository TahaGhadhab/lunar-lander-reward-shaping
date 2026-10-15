# Technical design and critical review

Companion to `PLAN.md` (phases and done-criteria) and `BRIEF.md` (the game and the idea).
This file says **what we build, with what, how the pieces fit, and where the existing plan is weak**.
Written before any code. Section 9 is a critical review of the kit itself; section 10 lists the changes I recommend.

---

## 1. What the project is, in one paragraph

Train a PPO agent (Stable-Baselines3) to land the Box2D lunar lander (`LunarLander-v3`, Gymnasium).
Then use it as a lab for **reward shaping**: add extra reward terms through wrappers, deliberately write one that
the agent can exploit (**reward hacking**: it learns to stall until the 1,000-frame time limit instead of landing),
diagnose it with honest metrics (evaluation on the *plain* reward, truncation rate, video), and fix it with
**potential-based shaping** (Ng, Harada, Russell 1999). The deliverable is a portfolio repo: code, tests,
3 clips, learning curves and a results table that a stranger can reproduce.

The interesting part is not "PPO lands a rocket" (a tutorial does that). It is the **experimental method**:
hypothesis written first, multiple seeds, evaluation that cannot be fooled by the shaped reward, and a fix
justified by theory.

---

## 2. Current state of the machine (checked 2026-10-15)

| Item | State | Consequence |
|------|-------|-------------|
| Project folder | `E:\lunar-lander-claude-kit\lunar-lander-reward-shaping` | Run Claude Code **from this folder** so `CLAUDE.md` and `.claude/` load. The parent `.claude-flow/` is plugin noise, unrelated. |
| Code | none (`src/`, `tests/`, `scripts/` do not exist) | Everything starts at Phase 0. |
| `.venv` | **missing** | SETUP.md step 2 not done. |
| Git | **not a repo** | `git init` needed; `config.json` wants a git hash. |
| Default `python` | **3.14.6** | Trap. Box2D has no wheel there. Always create the venv with `py -3.12`. |
| Python 3.12 | installed (`py -3.12`) | Good. |
| CPU | 12 logical cores, no GPU needed | 16 envs > 12 cores: measure `SubprocVecEnv` vs `DummyVecEnv` in the dry run. |

---

## 3. Stack

| Layer | Choice | Why |
|-------|--------|-----|
| Language | Python 3.12 | Prebuilt Box2D wheel. |
| Environment | `gymnasium[box2d]>=1.2,<1.4`, `LunarLander-v3` | v2 is gone. Cap below 1.4 (released after SB3 2.9). |
| RL library | `stable-baselines3>=2.9,<3` | Reference PPO implementation; handles time-limit bootstrapping correctly (see 5.4). |
| Deep learning | `torch>=2.8` CPU build | MLP of ~10k parameters: CPU is faster than GPU here. |
| Video | `moviepy>=1.0.3` (via `gymnasium.wrappers.RecordVideo`) | MP4 clips. `imageio-ffmpeg` brings ffmpeg. |
| Logging | SB3 `Monitor` CSVs (+ optional `tensorboard`) | CSVs feed `plots.py`. |
| Plots | `matplotlib`, `pandas` | **Not** pulled in by SB3 2.9 (checked: it only requires cloudpickle, gymnasium, numpy, torch). Listed explicitly in `requirements.txt`. |
| Quality | `pytest>=8`, `ruff>=0.6` | Run before every "done". |
| Packaging | `pyproject.toml`, `src/` layout, `pip install -e .` | `python -m lander.train` works from the repo root. |

---

## 4. The neural network

**Type: a small fully connected actor-critic (MLP).** No CNN (the agent never sees pixels, it gets 8 numbers),
no LSTM/recurrence (the 8 numbers describe the physical state completely, so the problem is Markov; see the one
exception in 9.3).

SB3 `PPO("MlpPolicy", ...)` with default architecture for a discrete action space:

```
                 ┌─ policy net (actor) ─ Linear 8→64 ─ tanh ─ Linear 64→64 ─ tanh ─ Linear 64→4 → softmax over 4 actions
obs (8 floats) ──┤
                 └─ value net (critic) ─ Linear 8→64 ─ tanh ─ Linear 64→64 ─ tanh ─ Linear 64→1 → V(s), expected return
```

- Actor and critic do **not** share layers (SB3 default `net_arch=dict(pi=[64,64], vf=[64,64])`).
- Parameters: 2 × (8·64+64 + 64·64+64) + (64·4+4) + (64+1) = **9,797**. Tiny. That is why CPU wins.
- Orthogonal initialisation, Adam, learning rate 3e-4 (SB3 defaults, unchanged).
- Inputs are not normalised: LunarLander observations are already roughly in [-1.5, 1.5]. No `VecNormalize`
  (it would also complicate evaluation, since its statistics must be saved and reloaded).

### PPO hyperparameters (from `CLAUDE.md`, unchanged unless logged in `EXPERIMENTS.md`)

| Param | Value | What it means here |
|-------|-------|--------------------|
| `n_envs` | 16 | 16 landers flown in parallel |
| `n_steps` | 1024 | frames per env per rollout → **16,384** transitions per update |
| `batch_size` | 64 | 256 minibatches per epoch |
| `n_epochs` | 4 | → 1,024 gradient steps per update |
| `gamma` | 0.999 | effective horizon ≈ 1/(1-γ) = 1,000 frames ≈ the episode limit |
| `gae_lambda` | 0.98 | advantage estimation bias/variance trade-off |
| `ent_coef` | 0.01 | small exploration bonus |
| total | 1,000,000 steps | ≈ 62 updates, ≈ 63k gradient steps |

Note: `batch_size=64` on a 16k rollout makes the update phase the slowest part on CPU. A bigger batch would be
faster but changes the learning dynamics. Keep it, measure it, only revisit with a logged reason.

---

## 5. Architecture

### 5.1 Data flow

```
                  TRAINING                                            EVALUATION (never sees shaping)
gym.make("LunarLander-v3")  [TimeLimit 1000 inside]       gym.make("LunarLander-v3")
   └─ Monitor (plain reward → monitor.csv)                   └─ deterministic policy, seeds 10_000+i, 50 episodes
        └─ RewardWrapper(name)  (adds terms, info["shaping"])     └─ eval.json: mean, std, success, length,
             └─ SubprocVecEnv / DummyVecEnv × 16                        truncation, crash, landed
                  └─ PPO (MlpPolicy, cpu) → model.zip + config.json
```

**Key design decision: put `Monitor` *inside* the reward wrapper**, directly on the plain env.
Then `monitor.csv` and the `ep_rew_mean` printed by SB3 are the **plain** return, comparable across all reward
variants, and the "training score rises while the lander hovers" gotcha disappears from the curves.
The shaped return is still logged: the reward wrapper accumulates its terms and exposes the episode total in
`info` at episode end, recorded by a small SB3 callback (or a second CSV).
(The kit's `CLAUDE.md` does not specify the order; this is my recommendation, to confirm in Phase 2.)

### 5.2 Files and functions

```
pyproject.toml          package "lander", src layout, ruff + pytest config
requirements.txt        ranges from SETUP.md step 3
scripts/smoke_test.py   imports + make LunarLander-v3 + one step → "SMOKE TEST OK"
scripts/random_agent.py Phase 1: render 3 random episodes, print obs table, 20 headless episodes
src/lander/
  __init__.py
  envs.py
  rewards.py
  train.py
  evaluate.py
  record.py
  plots.py
tests/
  conftest.py           fake scripted env for wrapper tests
  test_rewards.py
  test_envs.py
  test_evaluate.py
```

#### `envs.py`
| Function | Role |
|----------|------|
| `ENV_ID = "LunarLander-v3"` | single source of truth |
| `make_env(reward_name="baseline", seed=None, render_mode=None, monitor_path=None) -> gym.Env` | `gym.make` → optional `Monitor` → reward wrapper from registry. Seeds via first `reset(seed=...)` and `action_space.seed`. |
| `make_vec_env(reward_name, seed, n_envs, monitor_dir, vec_cls) -> VecEnv` | 16 copies, env *i* seeded `seed*1000 + i` (training seeds never collide with eval seeds ≥ 10,000 as long as seed < 10). |

#### `rewards.py`
| Item | Role |
|------|------|
| `class ShapingWrapper(gym.Wrapper)` | Base. `step()` calls `self.terms(obs, action, terminated, truncated)` → `dict[str, float]`, adds their sum to the built-in reward, writes `info["shaping"] = terms`, passes `terminated`, `truncated`, `info` through. Accumulates per-episode shaping, reset in `reset()`. |
| `REGISTRY: dict[str, type]`, `@register(name)`, `get_reward(name)` | Names → wrapper classes; unknown name raises with the list of valid names. |
| `baseline` | no terms (identity). |
| `upright_slow_score(obs, tilt_max, speed_max) -> float` | Shared "goodness" used by Phases 5 and 6. Pure function, unit-tested. Obs indices: 2 x-speed, 3 y-speed, 4 tilt. |
| `TimePenalty(fee=0.05)` | `{"time_fee": -fee}` each frame. |
| `UprightSlowBonus(tilt_max, speed_max, bonus=1.0)` | `{"upright_slow": +1}` when the state is upright and slow. The deliberate loophole. |
| `UprightSlowPotential(k, gamma=0.999, fee=0.0)` | `F = γ·Φ(s') − Φ(s)`, `Φ = k · upright_slow_score`. `Φ(s')=0` if **terminated**, kept if **truncated**. Stores `Φ(s)` from `reset()`. `fee` optional (see 9.4). `upright_slow_fixed` = this with `fee=0.05`. |

#### `train.py`
`parse_args()` (`--reward --seed --steps --n-envs --out --vec {subproc,dummy}`), `get_versions()`,
`git_hash()`, `build_config(args) -> dict`, `train(config) -> Path`, `main()` under `if __name__ == "__main__":`.
Writes `runs/<reward>/seed<N>/{model.zip, config.json, monitor/*.csv}` and the wall-clock time and steps/s
into `config.json` at the end.

#### `evaluate.py`
`run_episode(model, env, seed) -> EpisodeResult` (return, length, terminated, truncated, outcome),
`classify_outcome(env, terminated, truncated) -> "landed" | "crashed" | "out_of_bounds" | "timeout"`,
`summarize(results) -> dict` (pure, unit-tested), `evaluate(run_dir, n=50) -> dict`, `main()`.
Outcome detection reads the unwrapped env at the end (`lander.awake` false → landed; `game_over` → crash;
`|x| >= 1` → out of bounds; `truncated` → timeout). More reliable than guessing from the last reward.
(Attribute names to verify against the installed Gymnasium source in Phase 3.)

#### `record.py`
`record(run_dir, out_path, seed=10_000, max_episodes=1)`: plain env, `render_mode="rgb_array"`, `RecordVideo`,
deterministic policy, then check the file is under 5 MB.

#### `plots.py`
`load_monitor(run_dir) -> DataFrame`, `to_grid(df, step_grid)` (interpolate each seed onto common timesteps),
`plot_rewards(rewards, runs_root, out)` → mean ± std band across seeds; one plot for plain return,
one for episode length (the stalling signature is clearest there).

#### Tests (minimum set)
| Test | Checks |
|------|--------|
| `test_baseline_identity` | same seed, same actions → `baseline` reward equals plain env reward |
| `test_passthrough` | on a scripted fake env, every wrapper returns `terminated`, `truncated`, original `info` keys unchanged |
| `test_time_penalty_math` | reward = built-in − fee |
| `test_upright_slow_bonus_thresholds` | hand-made obs just inside / just outside each threshold |
| `test_potential_telescoping` | sum of F over a scripted episode = γ^T Φ(s_T) − Φ(s_0); Φ reset on `reset()`; Φ(s')=0 on terminated, not on truncated |
| `test_registry` | every name builds; unknown name raises |
| `test_make_env_smoke` | real `LunarLander-v3`, obs shape (8,), one step |
| `test_summarize` | success / truncation / length on synthetic episode results |

### 5.3 What `config.json` contains
reward name and its constants, seed, total steps, n_envs, vec class, all PPO hyperparams, device,
package versions (python, gymnasium, sb3, torch, numpy), git hash (+ "dirty" flag), ISO date, wall-clock, steps/s.

### 5.4 A subtle but critical detail: time-limit bootstrapping
SB3's PPO treats a **truncated** episode (hit 1,000 frames) as "not over": it adds `γ·V(last state)` to the
last reward. This is correct RL, and it is why the wrapper **must** pass `truncated` through untouched.
It also makes the Phase 5 hack *stronger*: the agent does not see a clock (no time in the 8 observations), so for
it, stalling in a "good" state looks like an endless stream of +1. See 9.1.

---

## 6. Reward variants (formulas)

Built-in per-frame reward `r` is always kept. Thresholds below are proposals, to be finalised in the phase.

| Name | Added term | Expected result |
|------|-----------|-----------------|
| `baseline` | 0 | lands, mean ≥ 200 on most seeds at 1M |
| `time_penalty` | `-0.05` per frame | still lands, shorter episodes. Max cost 50 over 1,000 frames, half the landing bonus: enough to matter, far too small to make crashing (−100) attractive. |
| `upright_slow_bonus` | `+1` if `|tilt| < 0.2 rad` and `‖(vx, vy)‖ < 0.2` (obs units) | stalls until truncation (hack) |
| `upright_slow_potential` (ablation, recommended) | `γΦ(s') − Φ(s)`, `Φ = k·score`, `k = 10` | lands, truncation ≈ 0 |
| `upright_slow_fixed` | same as above + fee `-0.05` | lands, truncation ≈ 0 |

Thresholds note: obs speeds are scaled (not m/s). Before fixing values, Phase 1's printout and a few
baseline landings should be used to see typical values of `|tilt|` and speed near touchdown.
`k = 10` makes the total potential swing (≤ 10) small next to the landing bonus (100): it guides without dominating.

---

## 7. Evaluation protocol (non-negotiable)

- Plain env (`reward_name="baseline"`), `deterministic=True`, 50 episodes, seeds `10_000 … 10_049`.
- Per run: mean ± std return, success rate (return ≥ 200), mean length, **truncation rate**, landed/crash rates.
- Per reward: 3 seeds; report the mean across seeds of the per-seed means, and the worst seed (a fix that works on
  2/3 seeds is not a fix).
- Flags: truncation > 20% → stalling; std > 80 → unstable policy.
- The clip is from the **median** seed, not the best.

---

## 8. Where to start, exactly

**Step 0 (you, PowerShell, ~10 min, in `E:\lunar-lander-claude-kit\lunar-lander-reward-shaping`):**
```powershell
git init
py -3.12 -m venv .venv            # NOT "python -m venv": your default python is 3.14
.\.venv\Scripts\Activate.ps1
python --version                  # must print 3.12.x
python -m pip install --upgrade pip
pip install "gymnasium[box2d]>=1.2,<1.4" "stable-baselines3>=2.9,<3" "torch>=2.8" "moviepy>=1.0.3" "tensorboard>=2.16" "pytest>=8" "ruff>=0.6"
python -c "import Box2D, gymnasium as gym, stable_baselines3 as sb3, torch; e=gym.make('LunarLander-v3'); o,_=e.reset(seed=0); print(o.shape, gym.__version__, sb3.__version__, torch.__version__); print('SMOKE TEST OK')"
```
**Step 1 (Claude Code, Phase 0 prompt from `PROMPTS.md`):** `requirements.txt`, `pyproject.toml`,
`scripts/smoke_test.py`, `pip install -e .`, versions into `EXPERIMENTS.md`, first commit.

Then Phases 1 → 7 as in `PLAN.md`, with the amendments in section 10.

Compute budget (to confirm with the 50k dry run): expect very roughly 5 to 20 min per 1M-step run on this CPU.
With 3 seeds × 4 rewards = 12 runs (+3 for the ablation), plan for **1.5 to 5 h of training in total**,
mostly unattended.

---

## 9. Critical review of the kit (what is right, what is weak, what is wrong)

Overall: the kit is good. The evaluation discipline (plain env, truncation rate, seeds, hypothesis first,
auditor agent) is better than most RL tutorials. The issues below are real but fixable.

### 9.1 The hover arithmetic in `BRIEF.md` is right in direction, understated in size
`BRIEF.md`: hovering earns ≤ +1,000 and costs ≤ ~300 in fuel, so it beats landing. More precisely:
- Undiscounted: hovering nets about `1 − 0.3·p_main` per frame (`p_main` = share of frames with the main engine on,
  plausibly 0.6–0.8) → roughly **+750** over 1,000 frames, vs. roughly +100 (landing bonus) + 20 (legs) + the
  bonus collected on the way down for a landing.
- With `γ = 0.999`, the discounted sum of +1 over 1,000 frames is `(1 − 0.999^1000)/0.001 ≈ 632`.
- With SB3's truncation bootstrapping (5.4), a stalling state is valued at about `0.75/(1 − γ) ≈ 750`
  whatever the frame number, because the agent has no clock. Landing is worth about 100.
- Conclusion: the hack is not just possible, it is **strongly** preferred once found. The real risk in Phase 5 is
  not "no seed hovers", it is "the agent never visits the bonus region" if the thresholds are too tight.

### 9.2 The hack will probably not look like hovering (important for Phase 5)
The episode ends with +100 when the lander comes to rest (the Box2D body goes to sleep). The rule
"+1 per frame upright and slow" is also satisfied **while sitting on the ground**. Firing a side engine costs only
0.03 per frame and keeps the body awake. So the cheapest exploit is probably **"land, then twitch the side engines
forever"** (net ≈ +0.97 per frame), not hovering (net ≈ +0.75 per frame).
Consequences:
- `PLAN.md` Phase 5 done-criterion "at least one seed **hovers**" may never be met literally, even though the
  experiment succeeded. The right criterion is a **truncation-driven exploit**: truncation rate high, length ≈ 1000,
  success rate ≈ 0. Name it by what the clip shows ("sit-and-fidget" or "hover").
- Its plain-env return may be only mildly bad (little fuel burned, legs-contact points kept), so
  **mean return alone could hide it**. Truncation rate and the clip catch it. Strong point for the write-up.
- If you specifically want a hovering clip, a variant paying the bonus only when both legs are off the ground
  (`obs[6] == obs[7] == 0`) forces it. Optional; decide after seeing Phase 5.
- This is a prediction, not a fact. Phase 5's hypothesis entry in `EXPERIMENTS.md` should state it so it can be refuted.
- **Outcome (2026-10-15): refuted for 2 of 3 seeds.** The mechanism is real (side impulses use `wake=True`), but
  seeds 1 and 2 learned to hover; only seed 0 partly sits and twitches. See `EXPERIMENTS.md`.

### 9.3 Partial observability of time
The observation has no time feature, so the agent cannot know the episode will be cut. That is fine and even
desirable here (it is what makes bootstrapping correct). Do **not** add a time feature: it would change the
experiment. Worth one sentence in the README.

### 9.4 Phase 6 mixes two fixes, so we cannot tell which one worked
`upright_slow_fixed` = potential-based term **+** a per-frame fee. Theory says the potential-based term alone is
enough (it cannot change the optimal policy). The fee is not potential-based: it *does* change the optimal policy
(it rewards ending episodes early; harmless at 0.05, but it is a second change).
If both are added, the write-up cannot honestly claim "potential-based shaping fixed it".
**Recommendation:** add the ablation `upright_slow_potential` (same Φ, fee = 0), 3 seeds. It is the most
scientifically valuable run of the project and costs one extra batch of training.
Also be honest in advance: by design, a potential-based bonus is expected to give roughly **baseline-level**
results, not better ones. The claim is "it no longer breaks the agent", not "it lands better".

### 9.5 Potential-based shaping has three implementation traps
1. `γ` in `F` must equal PPO's `γ` (0.999). With `γ_F = 1`, each frame leaks `(1 − 0.999)·Φ` of bias.
   With `k = 10`: up to 0.01 per frame, ~10 over an episode. Small, but exactly the kind of leak the auditor exists for.
2. `Φ(terminal) = 0` on **termination**, but **not** on truncation (the episode is not really over, see 5.4).
3. `Φ(s)` must be recomputed in `reset()`. A stale `Φ` from the previous episode pays a random first-step bonus.
   Side effect worth explaining in the README: at touchdown `F = −Φ(s)`, so landing *looks* penalised by the shaping
   term on the last step. That is the theory working (the potential is "given back"), not a bug.

### 9.6 Seeds are inconsistent across the results table
`CLAUDE.md` says a verdict needs ≥ 3 seeds, but `PLAN.md` trains `baseline` and `time_penalty` on seed 0 only.
The final table would compare 1-seed rows against 3-seed rows, and the baseline is what everything is compared to.
**Recommendation:** baseline and time_penalty on seeds 0, 1, 2 too (cost: 4 extra runs).

### 9.7 "Mean ≥ 200 on every seed" for the fix is a risky criterion at 1M steps
PPO at 1M steps with these hyperparameters lands around or a bit above 200, and some seeds can fall short. If
baseline seed 2 gets 180, the fix cannot be held to a higher bar than the baseline. **Recommendation:** judge
`upright_slow_fixed` *relative to the baseline* (within one baseline std, truncation < 5%), and if the baseline
itself misses 200 on some seeds, move **all** rewards to the same larger budget (e.g. 1.5M), not only the one that
struggles. Same budget for every variant, always.

### 9.8 Where `Monitor` sits is unspecified, and it matters
See 5.1. If `Monitor` wraps the reward wrapper, every curve is the shaped return and the curves of different
rewards are not comparable. Putting it inside fixes both the gotcha and `plots.py`.

### 9.9 Tests come late
Phase 4 introduces tests, but `rewards.py` (base wrapper and registry) is written in Phase 2. The base wrapper's
pass-through is the most important invariant of the project. Write `test_passthrough` and `test_baseline_identity`
in Phase 2 (5 minutes), the rest in Phase 4 as planned.

### 9.10 Small technical risks to check early
- 16 envs on 12 logical cores with `SubprocVecEnv` on Windows (process spawn, pickling): may be *slower* than
  `DummyVecEnv` for such a cheap env. Measure both in the 50k dry run, keep the faster, log it.
- PyTorch threads competing with 16 worker processes: try `torch.set_num_threads(1..4)` if steps/s is poor.
- `RecordVideo` + moviepy 2.x: old import paths were removed in moviepy 2. Gymnasium 1.x handles it, but run
  `record.py` once on the 50k dry-run model rather than for the first time at the end of Phase 3.
- Bit-exact reproducibility across processes on CPU is not guaranteed. Promise "same seed → same ballpark", not identical numbers.
- `.claude/settings.json` allows `Bash(python:*)`. Outside the venv that resolves to Python 3.14. `CLAUDE.md`
  already says to call `.venv/Scripts/python`; stick to it.

### 9.11 Statistical honesty
3 seeds × 50 episodes is fine for a portfolio, not for a paper. Do not write "significant". Report per-seed
numbers and the worst seed so a reader sees the spread. The effect measured (truncation ≈ 100% vs ≈ 0%)
is large, so 3 seeds are enough to make the point.

### 9.12 Time estimates
"Tonight 3–4 h for phases 0–4" is feasible only if Phase 0 has no install problems and training is ~10 min per run.
With amendment E (3 seeds for baseline and time_penalty) Phase 4 will likely end tomorrow. The plan already allows it.

---

## 10. Recommended amendments to `PLAN.md` (for your decision)

| # | Change | Phase | Cost | Value |
|---|--------|-------|------|-------|
| A | Create the venv with `py -3.12` explicitly | 0 | 0 | avoids a broken install |
| B | `Monitor` innermost; log the shaped return separately | 2 | 0 | comparable curves, removes the gotcha |
| C | Pass-through and identity tests written with the base wrapper | 2 | 5 min | catches the worst bug early |
| D | Dry run compares `SubprocVecEnv` vs `DummyVecEnv` | 2 | 2 min | real time saving |
| E | Baseline and time_penalty on 3 seeds | 2–4 | 4 runs | fair results table |
| F | Phase 5 criterion = truncation exploit (hover **or** sit-and-fidget) | 5 | 0 | criterion matches the likely outcome |
| G | Ablation `upright_slow_potential` (no fee) | 6 | 3 runs | proves *which* fix worked |
| H | Fix judged relative to the baseline, same step budget for all | 6 | 0 | fair verdict |
| I | Run `record.py` on the dry-run model | 2–3 | 1 min | catches video issues early |

None of these changes the project's goal or the PPO hyperparameters. A, B, C, D, F, H and I are free;
E and G cost compute and are what make the conclusions defensible.
