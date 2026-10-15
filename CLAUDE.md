# Lunar Lander: reward shaping with PPO

Train a PPO agent to land Gymnasium's `LunarLander-v3`, then study **reward shaping**:
add custom reward terms through a wrapper, deliberately cause **reward hacking** (hovering),
fix it, and document both with clips, curves and numbers. This is a portfolio project.

Full background (observations, actions, built-in reward, the hover trap): `docs/BRIEF.md`.
Phase plan with done-criteria: `docs/PLAN.md`. Experiment log: `docs/EXPERIMENTS.md`.
Read the relevant one before starting a phase.

## How I work with you
- I specify and review; you write the code. Explain non-obvious choices in 1 or 2 lines.
- One phase at a time (see `docs/PLAN.md`). Stop at the end of a phase and show me the evidence
  (command output, numbers, file paths) before moving on.
- Before saying "done": run it, then run `pytest -q` and `ruff check .`. Never claim a result you did not observe.
- If a step needs more than 10 min of compute, tell me the expected duration first.
- I may write to you in French. Code, comments, docs and commits stay in English.

## Environment
- Windows. Python **3.12** in `.venv` (setup steps: `docs/SETUP.md`).
  Your shell is Git Bash: call `.venv/Scripts/python` (and `.venv/Scripts/python -m pytest`, `-m ruff`) explicitly.
  I use PowerShell with `.\.venv\Scripts\Activate.ps1`.
- CPU only. Force `device="cpu"` for PPO (an MLP policy is faster on CPU).
- Dependencies: the ranges in `docs/SETUP.md` step 3 go into `requirements.txt` (create it in Phase 0).
  Ask before adding a new one.

## Code layout (create as needed, keep it this simple)
```
src/lander/
  envs.py       make_env(reward_name, seed, render_mode) -> env: Monitor (plain) then reward wrapper
  rewards.py    one gymnasium.Wrapper subclass per reward variant, registered by name
  train.py      CLI: --reward NAME --seed N --steps N --n-envs N --out runs/
  evaluate.py   CLI: loads a model, evaluates on the PLAIN env (no custom wrapper)
  record.py     CLI: records an MP4 of a model with RecordVideo
  plots.py      learning curves from Monitor CSVs (mean +/- std across seeds)
tests/          pytest: reward math, wrapper pass-through, env smoke test
runs/<reward>/seed<N>/   model.zip, monitor CSVs, eval.json, config.json   (git-ignored)
media/          final clips + plots kept for the README (committed, small files only)
```
Install the package in editable mode (`pip install -e .` with a minimal `pyproject.toml`) so
`python -m lander.train` works from the repo root.

## Hard rules
- Env ID is `LunarLander-v3`. v2 no longer exists; old tutorials are wrong.
- Custom rewards are **wrappers** around the env. Never edit Gymnasium's source.
- A wrapper adds to the built-in reward. It passes `terminated`, `truncated` and `info` through unchanged
  and logs its own terms in `info["shaping"]` so we can see what was paid.
- **Evaluate with the custom wrapper OFF**, deterministic policy, at least 50 episodes, fixed eval seeds.
  Report: mean and std of return, success rate (return >= 200), mean episode length,
  and **truncation rate** (episodes hitting 1000 steps = hover signature).
- Every run writes `config.json` (reward name, seed, steps, hyperparams, package versions, git hash, ISO date).
- Seeds: never trust one run. A verdict on a reward needs at least 3 seeds. Every reward, baseline included, gets
  seeds 0, 1, 2 and the same step budget.
- Windows: any script using `SubprocVecEnv` must be under `if __name__ == "__main__":`.
- Do not commit `runs/`, models or the venv. Keep `media/` files small (MP4 under 5 MB).

## Starting PPO hyperparameters (Hugging Face Deep RL course, Unit 1)
`n_steps=1024, batch_size=64, n_epochs=4, gamma=0.999, gae_lambda=0.98, ent_coef=0.01`, 16 envs,
1M timesteps. Change one only with a reason written in `docs/EXPERIMENTS.md`.

## Known gotchas
- Box2D install error mentioning `swig`: `pip install swig`, then reinstall `gymnasium[box2d]`.
  Python 3.12 normally gets a prebuilt wheel, so this should not happen.
- `RecordVideo` needs `render_mode="rgb_array"` and `moviepy`.
- `Monitor` wraps the plain env, *inside* the reward wrapper (see `docs/DESIGN.md` 5.1). So `ep_rew_mean` and
  `monitor.csv` show the **plain** return; the shaped return is logged as `shaped_ep_rew_mean`. Even so, the plain
  training return is stochastic: judge by the deterministic plain-env evaluation and the clip.
- The default `python` on this machine is 3.14. Never use it for the venv (no Box2D wheel); the venv is 3.12.

## Project skills and agents
- `/run-experiment <reward> [seeds] [steps]`: train, evaluate on the plain env, record, log. See `.claude/skills/`.
- `/evaluate-agent <run dir>`: evaluation only.
- `reward-auditor` agent: read-only review of a reward wrapper for hacking risks and eval leaks.

## Commands
- Setup: `docs/SETUP.md`
- Tests / lint: `.venv/Scripts/python -m pytest -q` and `.venv/Scripts/python -m ruff check .`
- Example: `.venv/Scripts/python -m lander.train --reward baseline --seed 0 --steps 1000000`
