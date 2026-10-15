---
name: evaluate-agent
description: Evaluate a trained Lunar Lander PPO model fairly (plain env with no custom reward wrapper, deterministic, 50 fixed-seed episodes) and report return, success rate, episode length and truncation rate. Use when the user asks how good a model is or wants to compare runs.
argument-hint: "<run dir, e.g. runs/baseline/seed0> [episodes=50]"
---

# Evaluate an agent

Target: `$ARGUMENTS`.

1. Load `model.zip` and `config.json` from the run dir. If either is missing, stop and say which.
2. Build the env with `make_env(reward_name="baseline", ...)`, i.e. **no custom reward wrapper**.
   The evaluation must never see the shaped reward.
3. Run 50 episodes (or the given number), `deterministic=True`, eval seeds `10_000 + i` (fixed and distinct from training seeds).
4. Compute and write to `<run dir>/eval.json`:
   - `mean_return`, `std_return`
   - `success_rate` (share of episodes with return >= 200)
   - `mean_length`
   - `truncation_rate` (share of episodes that hit the 1000-step limit)
   - `crash_rate` (episodes whose final step reward includes the -100 crash penalty), if it can be detected reliably
   - `n_episodes`, `eval_seeds`, ISO date
5. Print one line: `<run>  mean=X +/- Y  success=Z%  len=L  trunc=T%`.
6. Flag in plain words:
   - mean >= 200: solved.
   - truncation rate above 20%: the agent likely hovers (reward-hacking signature).
   - std above 80: unstable policy; check other seeds.
