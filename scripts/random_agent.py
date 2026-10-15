"""Phase 1: a random agent, to see the observations and how badly it does.

Usage: python scripts/random_agent.py [--render] [--episodes 20]
"""

import argparse

import numpy as np

from lander.envs import make_env

OBS_NAMES = [
    "x position (rel. to pad)",
    "height above pad",
    "horizontal speed",
    "vertical speed",
    "tilt angle (rad)",
    "angular speed",
    "left leg contact",
    "right leg contact",
]


def print_first_frames(n_frames: int = 5, seed: int = 0) -> None:
    env = make_env(seed=seed)
    obs, _ = env.reset(seed=seed)
    print(f"{'frame':>5}  " + "  ".join(f"{i:>7d}" for i in range(8)))
    for t in range(n_frames):
        print(f"{t:>5}  " + "  ".join(f"{v:7.3f}" for v in obs))
        obs, *_ = env.step(env.action_space.sample())
    env.close()
    print("columns: " + "; ".join(f"{i} {name}" for i, name in enumerate(OBS_NAMES)))


def run_episodes(n: int, render: bool, seed0: int = 100) -> list[float]:
    env = make_env(render_mode="human" if render else None)
    env.action_space.seed(seed0)
    returns, lengths = [], []
    for i in range(n):
        env.reset(seed=seed0 + i)
        done, ret, length = False, 0.0, 0
        while not done:
            _, r, terminated, truncated, _ = env.step(env.action_space.sample())
            ret += r
            length += 1
            done = terminated or truncated
        returns.append(ret)
        lengths.append(length)
    env.close()
    print(f"{n} random episodes: return mean {np.mean(returns):.1f} +/- {np.std(returns):.1f}, "
          f"min {np.min(returns):.1f}, max {np.max(returns):.1f}, mean length {np.mean(lengths):.0f}")
    return returns


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--render", action="store_true", help="show 3 episodes in a game window first")
    parser.add_argument("--episodes", type=int, default=20)
    args = parser.parse_args()

    print_first_frames()
    if args.render:
        run_episodes(3, render=True)
    run_episodes(args.episodes, render=False)


if __name__ == "__main__":
    main()
