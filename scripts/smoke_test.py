"""Check that the toolchain works: imports, LunarLander-v3 creation and one step."""

import platform
import sys

import Box2D  # noqa: F401  (import check only: the physics engine must load)
import gymnasium as gym
import stable_baselines3 as sb3
import torch


def main() -> None:
    env = gym.make("LunarLander-v3")
    obs, _ = env.reset(seed=0)
    obs, reward, terminated, truncated, _ = env.step(0)
    env.close()
    assert obs.shape == (8,), obs.shape

    print(f"python      {sys.version.split()[0]}")
    print(f"gymnasium   {gym.__version__}")
    print(f"sb3         {sb3.__version__}")
    print(f"torch       {torch.__version__}")
    print(f"cpu         {platform.processor()}")
    print(f"obs shape   {obs.shape}, first reward {reward:.3f}")
    print("SMOKE TEST OK")


if __name__ == "__main__":
    main()
