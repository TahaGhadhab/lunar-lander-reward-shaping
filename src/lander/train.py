"""Train PPO on LunarLander-v3 with a named reward.

Example: python -m lander.train --reward baseline --seed 0 --steps 1000000
Writes runs/<reward>/seed<N>/: model.zip, config.json, monitor/*.monitor.csv (plain return), progress.csv.
"""

from __future__ import annotations

import argparse
import json
import platform
import subprocess
import time
from datetime import datetime
from pathlib import Path

import numpy as np
import torch
from stable_baselines3 import PPO
from stable_baselines3.common.callbacks import BaseCallback, CallbackList, CheckpointCallback
from stable_baselines3.common.logger import configure

from lander.envs import ENV_ID, make_env, make_vec_env

# Hugging Face Deep RL course, Unit 1. Change only with a reason logged in docs/EXPERIMENTS.md.
PPO_HYPERPARAMS = {
    "n_steps": 1024,
    "batch_size": 64,
    "n_epochs": 4,
    "gamma": 0.999,
    "gae_lambda": 0.98,
    "ent_coef": 0.01,
}


class ShapedReturnCallback(BaseCallback):
    """Logs the shaped episode return (`info["shaped_episode"]`) next to SB3's plain `ep_rew_mean`."""

    def __init__(self, window: int = 100):
        super().__init__()
        self.window = window
        self.returns: list[float] = []

    def _on_step(self) -> bool:
        for info in self.locals["infos"]:
            if "shaped_episode" in info:
                self.returns.append(info["shaped_episode"]["r"])
        return True

    def _on_rollout_end(self) -> None:
        if self.returns:
            self.logger.record("rollout/shaped_ep_rew_mean", float(np.mean(self.returns[-self.window :])))


def get_versions() -> dict[str, str]:
    import gymnasium
    import stable_baselines3

    return {
        "python": platform.python_version(),
        "gymnasium": gymnasium.__version__,
        "stable_baselines3": stable_baselines3.__version__,
        "torch": torch.__version__,
        "numpy": np.__version__,
    }


def git_hash() -> str:
    try:
        head = subprocess.run(["git", "rev-parse", "HEAD"], capture_output=True, text=True, check=True)
        dirty = subprocess.run(["git", "status", "--porcelain"], capture_output=True, text=True, check=True)
    except (OSError, subprocess.CalledProcessError):
        return "unknown"
    return head.stdout.strip() + ("-dirty" if dirty.stdout.strip() else "")


def build_config(args: argparse.Namespace) -> dict:
    reward_params = make_env(args.reward).params()
    return {
        "env_id": ENV_ID,
        "reward": args.reward,
        "reward_params": reward_params,
        "seed": args.seed,
        "total_timesteps": args.steps,
        "n_envs": args.n_envs,
        "vec_env": args.vec,
        "torch_threads": args.torch_threads,
        "checkpoint_every": args.checkpoint_every,
        "device": "cpu",
        "policy": "MlpPolicy",
        "ppo": PPO_HYPERPARAMS,
        "versions": get_versions(),
        "git": git_hash(),
        "started": datetime.now().astimezone().isoformat(timespec="seconds"),
    }


def train(args: argparse.Namespace) -> Path:
    run_dir = Path(args.run_dir) if args.run_dir else Path(args.out) / args.reward / f"seed{args.seed}"
    if (run_dir / "model.zip").exists() and not args.force:
        raise SystemExit(f"{run_dir}/model.zip exists; use --force to overwrite")
    run_dir.mkdir(parents=True, exist_ok=True)

    config = build_config(args)
    (run_dir / "config.json").write_text(json.dumps(config, indent=2))

    if args.torch_threads:
        torch.set_num_threads(args.torch_threads)
    env = make_vec_env(args.reward, args.seed, args.n_envs, monitor_dir=run_dir / "monitor", vec=args.vec)
    model = PPO("MlpPolicy", env, seed=args.seed, device="cpu", verbose=1, **PPO_HYPERPARAMS)
    model.set_logger(configure(str(run_dir), ["stdout", "csv"]))

    callbacks = [ShapedReturnCallback()]
    if args.checkpoint_every:
        # save_freq counts vec-env calls (n_envs steps each), not timesteps.
        callbacks.append(CheckpointCallback(max(args.checkpoint_every // args.n_envs, 1),
                                            str(run_dir / "checkpoints"), name_prefix="model"))
    start = time.perf_counter()
    model.learn(total_timesteps=args.steps, callback=CallbackList(callbacks))
    elapsed = time.perf_counter() - start
    model.save(run_dir / "model.zip")
    env.close()

    config.update(
        finished=datetime.now().astimezone().isoformat(timespec="seconds"),
        wall_clock_s=round(elapsed, 1),
        steps_per_s=round(model.num_timesteps / elapsed, 1),
        timesteps_done=int(model.num_timesteps),
    )
    (run_dir / "config.json").write_text(json.dumps(config, indent=2))
    print(f"saved {run_dir / 'model.zip'}  {elapsed / 60:.1f} min  {config['steps_per_s']} steps/s")
    return run_dir


def parse_args(argv: list[str] | None = None) -> argparse.Namespace:
    p = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    p.add_argument("--reward", default="baseline")
    p.add_argument("--seed", type=int, default=0)
    p.add_argument("--steps", type=int, default=1_000_000)
    p.add_argument("--n-envs", type=int, default=16)
    p.add_argument("--vec", choices=["subproc", "dummy"], default="subproc")
    p.add_argument("--torch-threads", type=int, default=0, help="0 = PyTorch default")
    p.add_argument("--out", default="runs")
    p.add_argument("--run-dir", default=None, help="override output dir (e.g. for dry runs)")
    p.add_argument("--force", action="store_true")
    p.add_argument("--checkpoint-every", type=int, default=0,
                   help="save checkpoints/model_<steps>_steps.zip every N timesteps (0 = off)")
    return p.parse_args(argv)


def main(argv: list[str] | None = None) -> None:
    train(parse_args(argv))


if __name__ == "__main__":
    main()
