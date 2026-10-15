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


def upright_slow_score(obs: np.ndarray, tilt_max: float, speed_max: float) -> float:
    """1.0 when upright and still, falling linearly to 0 at |tilt| = tilt_max or speed = speed_max."""
    speed = float(np.hypot(obs[VX], obs[VY]))
    upright = max(0.0, 1.0 - abs(float(obs[TILT])) / tilt_max)
    slow = max(0.0, 1.0 - speed / speed_max)
    return upright * slow


def is_upright_and_slow(obs: np.ndarray, tilt_max: float, speed_max: float) -> bool:
    return abs(float(obs[TILT])) < tilt_max and float(np.hypot(obs[VX], obs[VY])) < speed_max


class TimePenalty(ShapingWrapper):
    """A fixed fee per frame: waiting costs something."""

    def __init__(self, env: gym.Env, fee: float = 0.05):
        super().__init__(env)
        self.fee = fee

    def params(self) -> dict[str, Any]:
        return {"fee": self.fee}

    def terms(self, obs, action, terminated, truncated):
        return {"time_fee": -self.fee}


class UprightSlowBonus(ShapingWrapper):
    """DELIBERATELY HACKABLE: +bonus on every frame the lander is upright and slow (a state, not progress)."""

    def __init__(self, env: gym.Env, tilt_max: float = 0.2, speed_max: float = 0.2, bonus: float = 1.0):
        super().__init__(env)
        self.tilt_max, self.speed_max, self.bonus = tilt_max, speed_max, bonus

    def params(self) -> dict[str, Any]:
        return {"tilt_max": self.tilt_max, "speed_max": self.speed_max, "bonus": self.bonus}

    def terms(self, obs, action, terminated, truncated):
        paid = self.bonus if is_upright_and_slow(obs, self.tilt_max, self.speed_max) else 0.0
        return {"upright_slow": paid}


class UprightSlowPotential(ShapingWrapper):
    """Potential-based shaping (Ng, Harada, Russell 1999): F = gamma * Phi(s') - Phi(s), Phi = k * score.

    Phi(s') is 0 on a true termination and kept on truncation (SB3 bootstraps truncated episodes).
    `gamma` must equal PPO's gamma, otherwise each frame leaks (1 - gamma) * Phi of bias.
    An optional per-frame `fee` is NOT potential-based; it is a separate, policy-changing term.
    """

    def __init__(self, env: gym.Env, k: float = 10.0, gamma: float = 0.999, fee: float = 0.0,
                 tilt_max: float = 0.2, speed_max: float = 0.2):
        super().__init__(env)
        self.k, self.gamma, self.fee = k, gamma, fee
        self.tilt_max, self.speed_max = tilt_max, speed_max
        self.prev_phi = 0.0

    def params(self) -> dict[str, Any]:
        return {"k": self.k, "gamma": self.gamma, "fee": self.fee,
                "tilt_max": self.tilt_max, "speed_max": self.speed_max}

    def phi(self, obs: np.ndarray) -> float:
        return self.k * upright_slow_score(obs, self.tilt_max, self.speed_max)

    def on_reset(self, obs):
        self.prev_phi = self.phi(obs)

    def terms(self, obs, action, terminated, truncated):
        next_phi = 0.0 if terminated else self.phi(obs)
        shaping = self.gamma * next_phi - self.prev_phi
        self.prev_phi = next_phi
        out = {"potential": shaping}
        if self.fee:
            out["time_fee"] = -self.fee
        return out


register("baseline", ShapingWrapper)
register("time_penalty", TimePenalty)
register("upright_slow_bonus", UprightSlowBonus)
register("upright_slow_potential", UprightSlowPotential)
register("upright_slow_fixed", lambda env: UprightSlowPotential(env, fee=0.05))
