# Prompts to paste into Claude Code

One per phase. Start each phase in a fresh session (`/clear`) once the previous one is committed:
`CLAUDE.md` and the docs carry the context, so the chat history is not needed.
Tip: for Phases 2, 5 and 6 press Shift+Tab to enter **plan mode** first, review the plan, then let it build.

---

## Phase 0: scaffold
```
Read CLAUDE.md, docs/SETUP.md and docs/PLAN.md. We are on Phase 0.
I already created .venv and installed the dependencies (SETUP.md steps 1 to 4).
Create requirements.txt with the ranges from SETUP.md step 3, a minimal pyproject.toml (src layout, package "lander",
ruff config, pytest config) and scripts/smoke_test.py. Install the package in editable mode, run the smoke test,
copy the versions into docs/EXPERIMENTS.md, and commit. Stop and show me the smoke test output.
```

## Phase 1: watch it fail
```
Phase 1 of docs/PLAN.md. Write src/lander/envs.py (make_env) and scripts/random_agent.py.
Show the game window for 3 random episodes, print the 8 observation numbers for the first 5 frames with their
meaning from docs/BRIEF.md, then run 20 episodes headless and log the mean return. Stop with the evidence.
```

## Phase 2: baseline training
```
Phase 2 of docs/PLAN.md. Plan first: show me the design of rewards.py (registry + baseline) and train.py
(CLI, config.json contents, Monitor logs, SubprocVecEnv guarded for Windows) before writing code.
After I approve: write them, do the 50k-step dry run, report steps per second and the estimated time for 1M steps,
then wait for my go.
```

## Phase 3: evaluate and record
```
Phase 3 of docs/PLAN.md. Write evaluate.py and record.py following the evaluate-agent skill. Evaluate
runs/baseline/seed0, record media/baseline_landing.mp4 (under 5 MB), fill the results table. Stop with the numbers.
```

## Phase 4: tests and time penalty
```
Phase 4 of docs/PLAN.md. Write the tests first (shaping math on hand-made transitions, wrapper pass-through,
env smoke test), then add the time_penalty reward. Justify the fee value in 2 lines. Run the reward-auditor agent
on it. Then /run-experiment time_penalty 0 and stop with the results.
```

## Phase 5: break it
```
Phase 5 of docs/PLAN.md. Add upright_slow_bonus exactly as described in docs/BRIEF.md. Propose thresholds
for "upright" and "slow" with reasons, and show me the arithmetic of why hovering should beat landing.
Do not fix the loophole: the point is to observe it. Then /run-experiment upright_slow_bonus.
```

## Phase 6: fix it
```
Phase 6 of docs/PLAN.md. Add upright_slow_fixed: potential-based shaping on the same "upright and slow" score,
plus a per-frame fee. Run the reward-auditor agent first and address its findings. Then
/run-experiment upright_slow_fixed and compare all rewards in one table.
```

## Phase 7: write-up
```
Phase 7 of docs/PLAN.md. Write plots.py (mean +/- std curves over seeds per reward), then a README for recruiters:
problem, method (with the potential-based shaping argument), results table, the three clips, what I learned,
and how to reproduce one result in under 5 commands. Be precise about roles: I specified, reviewed and validated;
the code was written with Claude Code.
```
