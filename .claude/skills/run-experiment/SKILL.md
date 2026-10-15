---
name: run-experiment
description: Run one reward-shaping experiment end to end for this Lunar Lander project (train PPO with a named reward wrapper over several seeds, evaluate on the plain env, record a clip, log the result). Use when the user asks to train, retrain, test or compare a reward variant.
argument-hint: "<reward-name> [seeds=0,1,2] [steps=1000000]"
---

# Run a reward experiment

Arguments: `$ARGUMENTS` (reward name, optional comma-separated seeds, optional steps).
Defaults: seeds `0,1,2`, steps `1000000`. Every reward, baseline included, uses the same seeds and step budget.
Training flags: `--vec dummy --torch-threads 1` (fastest on this CPU, see EXPERIMENTS.md); seeds may then run in parallel.

## Steps
1. **Check the reward exists** in `src/lander/rewards.py`. If it does not, stop and ask me for its formula.
   Do not invent one.
2. **Write the hypothesis first**: append a new entry to `docs/EXPERIMENTS.md` using its template
   (hypothesis, formula, constants, command). Do this before training.
3. **Dry run**: train 50,000 steps on the first seed into a throwaway dir. If it crashes, fix and repeat.
   Report steps per second so we can estimate the full run time.
4. **Tell me the estimated wall-clock time** for all seeds. If it is above 10 minutes, wait for my go.
5. **Train** each seed:
   `.venv/Scripts/python -m lander.train --reward <name> --seed <s> --steps <n> --out runs/`
   With `--torch-threads 1` each run uses one core: run the seeds in parallel, in the background.
6. **Evaluate** each seed on the PLAIN env (custom wrapper off), 50 deterministic episodes, fixed eval seeds:
   `.venv/Scripts/python -m lander.evaluate runs/<name>/seed<s>`
7. **Record** one clip for the most representative seed (the median by mean return, not the best):
   `.venv/Scripts/python -m lander.record runs/<name>/seed<s> --out media/<name>.mp4`
8. **Fill the log entry**: per-seed table (mean +/- std, success %, mean length, truncation %), the clip path
   and the verdict. Update the results table in `docs/PLAN.md`.
9. **Report to me** in 5 lines max: numbers per seed, what the clip shows, verdict, and anything that failed or was skipped.

## Rules
- Never report the shaped training reward (`ep_rew_mean`) as the result. The result is the plain-env evaluation.
- High truncation rate or mean length near 1000 means likely hovering. Say so explicitly.
- One seed is never a verdict. With fewer than 3 seeds, the verdict is "inconclusive".
- Do not change PPO hyperparameters inside this skill. If you think one should change, propose it.
