"""Record an MP4 of a trained model on the plain env (deterministic policy).

Example: python -m lander.record runs/baseline/seed0 --out media/baseline_landing.mp4
"""

from __future__ import annotations

import argparse
import shutil
import tempfile
from pathlib import Path

from gymnasium.wrappers import RecordVideo
from stable_baselines3 import PPO

from lander.envs import make_env
from lander.evaluate import EVAL_SEED0, run_episode

MAX_MB = 5.0


def record(run_dir: Path, out: Path, seed: int = EVAL_SEED0, model_name: str = "model.zip") -> Path:
    model = PPO.load(run_dir / model_name, device="cpu")
    out.parent.mkdir(parents=True, exist_ok=True)
    with tempfile.TemporaryDirectory() as tmp:
        video_dir = Path(tmp) / "video"  # must not exist yet, or RecordVideo warns about overwriting
        env = RecordVideo(make_env("baseline", render_mode="rgb_array"), str(video_dir),
                          episode_trigger=lambda _: True, name_prefix="clip", disable_logger=True)
        result = run_episode(model, env, seed)
        env.close()
        videos = sorted(video_dir.glob("*.mp4"))
        if not videos:
            raise RuntimeError("RecordVideo produced no file")
        shutil.move(str(videos[0]), out)

    size_mb = out.stat().st_size / 1e6
    print(f"{out}  seed={seed}  return={result.ret:.1f}  length={result.length}  "
          f"outcome={result.outcome}  size={size_mb:.2f} MB")
    if size_mb > MAX_MB:
        print(f"  warning: above {MAX_MB} MB, do not commit as is")
    return out


def main(argv: list[str] | None = None) -> None:
    p = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    p.add_argument("run_dir", type=Path)
    p.add_argument("--out", type=Path, required=True)
    p.add_argument("--seed", type=int, default=EVAL_SEED0)
    p.add_argument("--model", default="model.zip")
    args = p.parse_args(argv)
    record(args.run_dir, args.out, args.seed, args.model)


if __name__ == "__main__":
    main()
