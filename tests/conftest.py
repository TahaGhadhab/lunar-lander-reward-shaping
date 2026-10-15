"""A scripted fake env, so wrapper tests do not depend on Box2D physics."""

from __future__ import annotations

import gymnasium as gym
import numpy as np
import pytest


class ScriptedEnv(gym.Env):
    """Replays a fixed list of (obs, reward, terminated, truncated, info) transitions."""

    observation_space = gym.spaces.Box(-np.inf, np.inf, shape=(8,), dtype=np.float32)
    action_space = gym.spaces.Discrete(4)

    def __init__(self, reset_obs, transitions):
        self.reset_obs = np.asarray(reset_obs, dtype=np.float32)
        self.transitions = transitions
        self.t = 0

    def reset(self, *, seed=None, options=None):
        super().reset(seed=seed)
        self.t = 0
        return self.reset_obs.copy(), {"from_reset": True}

    def step(self, action):
        obs, reward, terminated, truncated, info = self.transitions[self.t]
        self.t += 1
        return np.asarray(obs, dtype=np.float32), reward, terminated, truncated, dict(info)


def obs(x=0.0, y=1.0, vx=0.0, vy=0.0, tilt=0.0, ang=0.0, left=0.0, right=0.0):
    return [x, y, vx, vy, tilt, ang, left, right]


@pytest.fixture
def scripted():
    """Factory: scripted(reset_obs, transitions) -> ScriptedEnv."""
    return ScriptedEnv
