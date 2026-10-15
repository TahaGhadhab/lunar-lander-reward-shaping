import numpy as np
import pytest
from conftest import obs

from lander.envs import make_env
from lander.rewards import REWARDS, UprightSlowPotential, make_reward_wrapper, upright_slow_score

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
    """At the end of an episode the wrapper reports the shaped return, shaping total and length."""
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
    """Asking for a reward that does not exist fails loudly with the valid names."""
    with pytest.raises(ValueError, match="unknown reward"):
        make_reward_wrapper(scripted(obs(), []), "does_not_exist")


# --- shaped rewards: math on hand-made transitions -------------------------------------------


def step_once(scripted, name, next_obs, terminated=False, truncated=False, reset_obs=None):
    transitions = [(next_obs, 0.0, terminated, truncated, {})]
    env = make_reward_wrapper(scripted(reset_obs or obs(), transitions), name)
    env.reset()
    _, r, *_, info = env.step(0)
    return r, info["shaping"]


def test_time_penalty_math(scripted):
    """time_penalty subtracts exactly 0.05 per frame from the built-in reward."""
    r, terms = step_once(scripted, "time_penalty", obs())
    assert terms == {"time_fee": -0.05} and r == pytest.approx(-0.05)


@pytest.mark.parametrize(
    "state,paid",
    [
        (obs(tilt=0.19, vx=0.0, vy=-0.19), 1.0),  # just inside both thresholds
        (obs(tilt=0.21), 0.0),  # too tilted
        (obs(vx=0.15, vy=-0.15), 0.0),  # speed 0.212 > 0.2
        (obs(tilt=-0.1, vy=0.1, left=1, right=1), 1.0),  # sitting on the ground also pays: the loophole
    ],
)
def test_upright_slow_bonus_thresholds(scripted, state, paid):
    """The +1 bonus is paid just inside the tilt and speed limits, not outside, and also on the ground."""
    _, terms = step_once(scripted, "upright_slow_bonus", state)
    assert terms["upright_slow"] == paid


def test_score_shape():
    """The upright-and-slow score is 1 when still and upright, falling linearly to 0 at the limits."""
    assert upright_slow_score(np.array(obs()), 0.2, 0.2) == 1.0
    assert upright_slow_score(np.array(obs(tilt=0.1)), 0.2, 0.2) == pytest.approx(0.5)
    assert upright_slow_score(np.array(obs(tilt=0.5)), 0.2, 0.2) == 0.0


def test_potential_telescopes(scripted):
    """With gamma=1, the shaping sum over an episode is Phi(s_T) - Phi(s_0)."""
    states = [obs(tilt=0.15), obs(tilt=0.1), obs(tilt=0.05), obs()]
    transitions = [(s, 0.0, False, False, {}) for s in states]
    env = UprightSlowPotential(scripted(obs(tilt=0.18), transitions), k=10.0, gamma=1.0)
    env.reset()
    total = sum(env.step(0)[4]["shaping"]["potential"] for _ in states)
    assert total == pytest.approx(10.0 * 1.0 - 10.0 * 0.1)


def test_potential_terminal_is_zero_but_truncation_is_not(scripted):
    """The potential drops to 0 when an episode really ends, but not when it is only cut by the time limit."""
    start = obs()  # Phi = 10
    _, term = step_once(scripted, "upright_slow_potential", obs(), terminated=True, reset_obs=start)
    _, trunc = step_once(scripted, "upright_slow_potential", obs(), truncated=True, reset_obs=start)
    assert term["potential"] == pytest.approx(-10.0)  # potential "given back" at landing
    assert trunc["potential"] == pytest.approx(0.999 * 10.0 - 10.0)


def test_potential_resets_between_episodes(scripted):
    """The potential is recomputed at each new episode, so no bonus leaks between episodes."""
    transitions = [(obs(tilt=0.5), 0.0, True, False, {})]
    env = UprightSlowPotential(scripted(obs(), transitions), k=10.0, gamma=1.0)
    env.reset()
    env.step(0)
    env.reset()  # Phi(s_0) must be recomputed, not left at the terminal 0
    assert env.prev_phi == pytest.approx(10.0)


def test_stalling_pays_nothing_under_potential(scripted):
    """Staying in the best state forever earns (gamma - 1) * Phi per frame: slightly negative, no loophole."""
    transitions = [(obs(), 0.0, False, False, {})] * 100
    env = make_reward_wrapper(scripted(obs(), transitions), "upright_slow_potential")
    env.reset()
    total = sum(env.step(0)[1] for _ in range(100))
    assert total == pytest.approx(100 * (0.999 - 1.0) * 10.0)


def test_fixed_adds_fee_to_potential(scripted):
    """upright_slow_fixed is the potential-based term plus the 0.05 fee per frame."""
    _, terms = step_once(scripted, "upright_slow_fixed", obs())
    assert set(terms) == {"potential", "time_fee"} and terms["time_fee"] == -0.05


def test_potential_gamma_matches_ppo(scripted):
    """The shaping gamma must equal PPO's, or every frame leaks (1 - gamma) * Phi of bias (audit finding)."""
    from lander.train import PPO_HYPERPARAMS

    for name in ("upright_slow_potential", "upright_slow_fixed"):
        env = make_reward_wrapper(scripted(obs(), []), name)
        assert env.params()["gamma"] == PPO_HYPERPARAMS["gamma"]
