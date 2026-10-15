"""Reward variants as gymnasium wrappers, registered by name.

Each wrapper keeps the built-in reward and adds named terms. It passes `terminated`, `truncated`
and the existing `info` keys through unchanged, and adds:
- `info["shaping"]`: the terms paid on this step (dict name -> float);
- `info["shaped_episode"]` on the last step of an episode: shaped return, shaping total and length.
"""

from __future__ import annotations

from collections.abc import Callable
from typing import Any

import gymnasium as gym
import numpy as np

# Observation indices (0-based), see docs/BRIEF.md.
X, Y, VX, VY, TILT, ANG_VEL, LEFT_LEG, RIGHT_LEG = range(8)


class ShapingWrapper(gym.Wrapper):
    """Base wrapper: adds `sum(self.terms(...))` to the built-in reward. No terms by default."""

    def __init__(self, env: gym.Env):
        super().__init__(env)
        self._episode_return = 0.0
        self._episode_shaping = 0.0
        self._episode_length = 0

    def params(self) -> dict[str, Any]:
        """Constants of this reward, written to config.json."""
        return {}

    def on_reset(self, obs: np.ndarray) -> None:
        """Hook to reset per-episode state (e.g. the previous potential)."""

    def terms(
        self, obs: np.ndarray, action: Any, terminated: bool, truncated: bool
    ) -> dict[str, float]:
        """Extra reward terms for the transition that produced `obs`."""
        return {}

    def reset(self, **kwargs):
        obs, info = self.env.reset(**kwargs)
        self._episode_return = 0.0
        self._episode_shaping = 0.0
        self._episode_length = 0
        self.on_reset(obs)
        return obs, info

    def step(self, action):
        obs, reward, terminated, truncated, info = self.env.step(action)
        terms = self.terms(obs, action, terminated, truncated)
        shaping = float(sum(terms.values()))
        shaped_reward = float(reward) + shaping

        self._episode_return += shaped_reward
        self._episode_shaping += shaping
        self._episode_length += 1

        info["shaping"] = terms
        if terminated or truncated:
            info["shaped_episode"] = {
                "r": self._episode_return,
                "shaping": self._episode_shaping,
                "l": self._episode_length,
            }
        return obs, shaped_reward, terminated, truncated, info


# name -> factory(env) -> wrapper. Constants live in the factory so a name always means one formula.
REWARDS: dict[str, Callable[[gym.Env], ShapingWrapper]] = {}


def register(name: str, factory: Callable[[gym.Env], ShapingWrapper]) -> None:
    if name in REWARDS:
        raise ValueError(f"reward {name!r} already registered")
    REWARDS[name] = factory


def make_reward_wrapper(env: gym.Env, name: str) -> ShapingWrapper:
    try:
        factory = REWARDS[name]
    except KeyError:
        raise ValueError(f"unknown reward {name!r}; valid: {sorted(REWARDS)}") from None
    return factory(env)


register("baseline", ShapingWrapper)
