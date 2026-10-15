"""Environment factories.

Wrapper order (outermost last): LunarLander-v3 (with its 1000-step TimeLimit) -> Monitor -> reward wrapper.
Monitor sits on the plain env, so monitor CSVs and SB3's `ep_rew_mean` report the plain return,
comparable across reward variants (docs/DESIGN.md 5.1).
"""

from __future__ import annotations

from collections.abc import Callable
from pathlib import Path

import gymnasium as gym
from stable_baselines3.common.monitor import Monitor
from stable_baselines3.common.vec_env import DummyVecEnv, SubprocVecEnv, VecEnv

from lander.rewards import make_reward_wrapper

ENV_ID = "LunarLander-v3"


def make_env(
    reward_name: str = "baseline",
    seed: int | None = None,
    render_mode: str | None = None,
    monitor_file: str | Path | None = None,
) -> gym.Env:
    env = gym.make(ENV_ID, render_mode=render_mode)
    if monitor_file is not None:
        env = Monitor(env, str(monitor_file))
    env = make_reward_wrapper(env, reward_name)
    if seed is not None:
        env.reset(seed=seed)
        env.action_space.seed(seed)
    return env


def make_vec_env(
    reward_name: str,
    seed: int,
    n_envs: int,
    monitor_dir: str | Path | None = None,
    vec: str = "subproc",
) -> VecEnv:
    """`n_envs` copies; env i gets seed `seed * 1000 + i` (disjoint from eval seeds 10_000+ if seed < 10)."""
    if monitor_dir is not None:
        Path(monitor_dir).mkdir(parents=True, exist_ok=True)

    def factory(i: int) -> Callable[[], gym.Env]:
        def _init() -> gym.Env:
            monitor_file = None if monitor_dir is None else Path(monitor_dir) / str(i)
            return make_env(reward_name, monitor_file=monitor_file)

        return _init

    fns = [factory(i) for i in range(n_envs)]
    vec_env = SubprocVecEnv(fns, start_method="spawn") if vec == "subproc" else DummyVecEnv(fns)
    vec_env.seed(seed * 1000)
    return vec_env
