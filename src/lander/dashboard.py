"""Recorded experiments, read from runs/ for the studio: results, training curves, checkpoint progression.

`python -m lander.dashboard` opens the live studio (`python -m lander.studio`).
"""

from __future__ import annotations

import json
from pathlib import Path

import numpy as np
import pandas as pd
from stable_baselines3 import PPO

from lander.envs import make_env
from lander.evaluate import EVAL_SEED0, run_episode, summarize
from lander.introspect import list_checkpoints
from lander.plots import DEFAULT_REWARDS, load_monitor, to_grid

__all__ = ["DEFAULT_REWARDS", "collect_curves", "collect_progression", "collect_results"]


def r(x, nd=3):
    """Round floats (and arrays) for a compact JSON."""
    return np.round(np.asarray(x, dtype=float), nd).tolist()


def collect_results(runs: Path, rewards: list[str]) -> dict:
    """Per reward, the eval.json of each seed (episode list replaced by the returns only)."""
    out = {}
    for reward in rewards:
        seeds = []
        for run_dir in sorted((runs / reward).glob("seed*")):
            ev = run_dir / "eval.json"
            if ev.exists():
                d = json.loads(ev.read_text())
                episodes = d.pop("episodes", [])
                d["seed"] = run_dir.name
                d["returns"] = r([e["ret"] for e in episodes], 1)
                seeds.append(d)
        if seeds:
            out[reward] = seeds
    return out


def collect_curves(runs: Path, rewards: list[str], steps: int = 1_000_000) -> dict:
    """Plain return and episode length (mean, std over seeds) on a common grid, plus shaped vs plain."""
    grid = np.linspace(10_000, steps, 100)
    curves = {"grid": r(grid, 0), "rewards": {}}
    for reward in rewards:
        run_dirs = sorted((runs / reward).glob("seed*"))
        if not run_dirs:
            continue
        entry = {}
        for col in ("r", "l"):
            c = np.array([to_grid(load_monitor(d), grid, col) for d in run_dirs])
            entry[col] = {"mean": r(np.nanmean(c, 0), 1), "std": r(np.nanstd(c, 0), 1)}
        shaped, plain, ts = [], [], None
        for d in run_dirs:
            prog = pd.read_csv(d / "progress.csv")
            if "rollout/shaped_ep_rew_mean" in prog:
                ts = prog["time/total_timesteps"].to_numpy()
                shaped.append(prog["rollout/shaped_ep_rew_mean"].to_numpy())
                plain.append(prog["rollout/ep_rew_mean"].to_numpy())
        if shaped:
            n = min(map(len, shaped))
            entry["shaped_vs_plain"] = {"t": r(ts[:n], 0),
                                        "shaped": r(np.mean([s[:n] for s in shaped], 0), 1),
                                        "plain": r(np.mean([p[:n] for p in plain], 0), 1)}
        curves["rewards"][reward] = entry
    return curves


def collect_progression(runs: Path, cache_dir: Path, n_episodes: int = 10) -> dict:
    """Each checkpoint of runs/progression/* evaluated on 10 fixed episodes (cached on disk)."""
    cache_file = cache_dir / "progression.json"
    cache = json.loads(cache_file.read_text()) if cache_file.exists() else {}
    out = {}
    env = make_env("baseline")
    for run_dir in sorted((runs / "progression").glob("*/seed*")):
        label = f"{run_dir.parent.name}/{run_dir.name}"
        points = []
        for steps, path in list_checkpoints(run_dir):
            stamp = f"{path.stat().st_mtime_ns}|{n_episodes}"
            key = f"{path.relative_to(runs).as_posix()}|{stamp}"
            legacy = f"{Path('runs') / path.relative_to(runs)}|{stamp}"  # keys written by older versions
            if key not in cache and legacy in cache:
                cache[key] = cache.pop(legacy)
            if key not in cache:
                model = PPO.load(path, device="cpu")
                s = summarize([run_episode(model, env, EVAL_SEED0 + i) for i in range(n_episodes)])
                cache[key] = {k: s[k] for k in ("mean_return", "landed_rate", "crashed_rate",
                                                "out_of_bounds_rate", "timeout_rate", "mean_length")}
            points.append({"steps": steps, **cache[key]})
        out[label] = points
    env.close()
    cache_dir.mkdir(parents=True, exist_ok=True)
    cache_file.write_text(json.dumps(cache))
    return out


def main(argv: list[str] | None = None) -> None:
    from lander.studio.__main__ import main as studio_main

    studio_main(argv)


if __name__ == "__main__":
    main()
