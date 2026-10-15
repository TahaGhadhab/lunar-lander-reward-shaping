from stable_baselines3.common.monitor import Monitor

from lander.envs import ENV_ID, make_env


def test_make_env_smoke():
    """LunarLander-v3 is created, returns 8 observations, and keeps its 1,000-frame time limit."""
    env = make_env("baseline", seed=0)
    assert env.spec.id == ENV_ID
    o, _ = env.reset(seed=0)
    assert o.shape == (8,)
    o, r, terminated, truncated, info = env.step(0)
    assert o.shape == (8,) and "shaping" in info
    assert env.spec.max_episode_steps == 1000


def test_monitor_is_inside_reward_wrapper(tmp_path):
    """Monitor must see the plain reward: it is the reward wrapper's inner env."""
    env = make_env("baseline", monitor_file=tmp_path / "0")
    assert isinstance(env.env, Monitor)
