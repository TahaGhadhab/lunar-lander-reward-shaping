"""Evaluate a trained model on the PLAIN env (no custom reward wrapper), deterministic, fixed seeds.

Example: python -m lander.evaluate runs/baseline/seed0 [--episodes 50]
Writes <run dir>/eval.json and prints one summary line.
"""

from __future__ import annotations

import argparse
import json
from dataclasses import asdict, dataclass
from datetime import datetime
from pathlib import Path

import gymnasium as gym
import numpy as np
from stable_baselines3 import PPO

from lander.envs import make_env

EVAL_SEED0 = 10_000  # distinct from training seeds (seed * 1000 + i, seed < 10)
SUCCESS_RETURN = 200.0
OUTCOMES = ("landed", "crashed", "out_of_bounds", "timeout")


@dataclass
class EpisodeResult:
    seed: int
    ret: float
    length: int
    terminated: bool
    truncated: bool
    outcome: str


def classify_outcome(env: gym.Env, terminated: bool, truncated: bool) -> str:
    """Mirror LunarLander's own ending logic: the 'at rest' check (+100) overrides the crash check (-100)."""
    if not terminated:
        return "timeout" if truncated else "running"
    lander_env = env.unwrapped
    if not lander_env.lander.awake:
        return "landed"
    if lander_env.game_over:
        return "crashed"
    return "out_of_bounds"


def run_episode(model: PPO, env: gym.Env, seed: int) -> EpisodeResult:
    obs, _ = env.reset(seed=seed)
    ret, length, terminated, truncated = 0.0, 0, False, False
    while not (terminated or truncated):
        action, _ = model.predict(obs, deterministic=True)
        obs, reward, terminated, truncated, _ = env.step(int(action))
        ret += float(reward)
        length += 1
    outcome = classify_outcome(env, terminated, truncated)
    return EpisodeResult(seed, ret, length, terminated, truncated, outcome)


def summarize(results: list[EpisodeResult]) -> dict:
    returns = np.array([r.ret for r in results])
    summary = {
        "n_episodes": len(results),
        "mean_return": float(returns.mean()),
        "std_return": float(returns.std()),
        "min_return": float(returns.min()),
        "success_rate": float((returns >= SUCCESS_RETURN).mean()),
        "mean_length": float(np.mean([r.length for r in results])),
        "truncation_rate": float(np.mean([r.truncated and not r.terminated for r in results])),
    }
    for outcome in OUTCOMES:
        summary[f"{outcome}_rate"] = float(np.mean([r.outcome == outcome for r in results]))
    return summary


def flags(summary: dict) -> list[str]:
    out = []
    if summary["mean_return"] >= SUCCESS_RETURN:
        out.append("solved (mean >= 200)")
    if summary["truncation_rate"] > 0.20:
        out.append("truncation > 20%: likely stalling (reward-hacking signature)")
    if summary["std_return"] > 80:
        out.append("std > 80: unstable policy, check other seeds")
    return out


def evaluate(run_dir: Path, n_episodes: int = 50, model_name: str = "model.zip") -> dict:
    model = PPO.load(run_dir / model_name, device="cpu")
    config_path = run_dir / "config.json"
    trained_reward = json.loads(config_path.read_text())["reward"] if config_path.exists() else None

    env = make_env("baseline")  # plain built-in reward: evaluation never sees the shaped reward
    seeds = [EVAL_SEED0 + i for i in range(n_episodes)]
    results = [run_episode(model, env, s) for s in seeds]
    env.close()

    summary = summarize(results)
    report = {
        **summary,
        "flags": flags(summary),
        "model": str(run_dir / model_name),
        "trained_reward": trained_reward,
        "eval_env": "LunarLander-v3 plain (baseline wrapper, no extra terms)",
        "deterministic": True,
        "eval_seeds": [seeds[0], seeds[-1]],
        "date": datetime.now().astimezone().isoformat(timespec="seconds"),
        "episodes": [asdict(r) for r in results],
    }
    (run_dir / "eval.json").write_text(json.dumps(report, indent=2))
    return report


def summary_line(name: str, s: dict) -> str:
    return (f"{name}  mean={s['mean_return']:.1f} +/- {s['std_return']:.1f}  "
            f"success={100 * s['success_rate']:.0f}%  len={s['mean_length']:.0f}  "
            f"trunc={100 * s['truncation_rate']:.0f}%  landed={100 * s['landed_rate']:.0f}%")


def main(argv: list[str] | None = None) -> None:
    p = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    p.add_argument("run_dir", type=Path)
    p.add_argument("--episodes", type=int, default=50)
    p.add_argument("--model", default="model.zip")
    args = p.parse_args(argv)

    report = evaluate(args.run_dir, args.episodes, args.model)
    print(summary_line(str(args.run_dir), report))
    for flag in report["flags"]:
        print(f"  - {flag}")


if __name__ == "__main__":
    main()
