import numpy as np

from lander.envs import make_env
from lander.scene import SceneRenderer


def rollout(render: bool, steps: int = 120):
    env = make_env("baseline", render_mode="rgb_array")
    env.reset(seed=42)
    scene = SceneRenderer() if render else None
    rng = np.random.default_rng(0)
    trace, frame = [], None
    for t in range(steps):
        obs, reward, terminated, truncated, _ = env.step(int(rng.integers(4)))
        env.render()
        if scene:
            frame = scene.render(env, 2, t)
        trace.append((obs.copy(), reward))
        if terminated or truncated:
            break
    return trace, frame


def test_scene_is_purely_cosmetic():
    """Same seed and actions: the textured renderer must not change a single observation or reward."""
    plain, _ = rollout(render=False)
    pretty, frame = rollout(render=True)
    assert len(plain) == len(pretty)
    for (o1, r1), (o2, r2) in zip(plain, pretty, strict=True):
        np.testing.assert_array_equal(o1, o2)
        assert r1 == r2
    assert frame.shape == (400, 600, 3) and frame.dtype == np.uint8
    assert frame.std() > 10  # an actual picture, not a blank frame
