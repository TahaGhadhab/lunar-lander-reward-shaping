import numpy as np
import pytest
from conftest import obs

from lander.envs import make_env
from lander.rewards import REWARDS, make_reward_wrapper

ENDINGS = [(False, False), (True, False), (False, True)]


@pytest.mark.parametrize("name", sorted(REWARDS))
@pytest.mark.parametrize("terminated,truncated", ENDINGS)
def test_passthrough(scripted, name, terminated, truncated):
    """Every reward keeps terminated, truncated, obs and the existing info keys unchanged."""
    transitions = [(obs(y=0.5), -1.5, terminated, truncated, {"key": 42})]
    env = make_reward_wrapper(scripted(obs(), transitions), name)
    reset_obs, reset_info = env.reset(seed=0)
    assert reset_info == {"from_reset": True}
    o, r, term, trunc, info = env.step(0)

    assert term is terminated and trunc is truncated
    np.testing.assert_array_equal(o, np.asarray(obs(y=0.5), dtype=np.float32))
    assert info["key"] == 42
    assert isinstance(info["shaping"], dict)
    assert r == pytest.approx(-1.5 + sum(info["shaping"].values()))
    assert ("shaped_episode" in info) == (terminated or truncated)


def test_shaped_episode_totals(scripted):
    transitions = [(obs(), 1.0, False, False, {}), (obs(), 2.0, True, False, {})]
    env = make_reward_wrapper(scripted(obs(), transitions), "baseline")
    env.reset()
    env.step(0)
    *_, info = env.step(0)
    assert info["shaped_episode"] == {"r": 3.0, "shaping": 0.0, "l": 2}


def test_baseline_identity():
    """Same seed, same actions: `baseline` pays exactly the plain env reward."""
    import gymnasium as gym

    plain = gym.make("LunarLander-v3")
    wrapped = make_env("baseline")
    plain.reset(seed=3)
    wrapped.reset(seed=3)
    rng = np.random.default_rng(0)
    for _ in range(200):
        a = int(rng.integers(4))
        o1, r1, te1, tr1, _ = plain.step(a)
        o2, r2, te2, tr2, _ = wrapped.step(a)
        np.testing.assert_array_equal(o1, o2)
        assert r1 == r2 and te1 == te2 and tr1 == tr2
        if te1 or tr1:
            break


def test_unknown_reward_raises(scripted):
    with pytest.raises(ValueError, match="unknown reward"):
        make_reward_wrapper(scripted(obs(), []), "does_not_exist")
