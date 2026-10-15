"""Record an MP4 of a trained model on the plain env (deterministic policy).

Example: python -m lander.record runs/baseline/seed0 --out media/baseline_landing.mp4 [--style classic]
`--style scene` (default) uses the textured renderer in lander.scene; `classic` is Gymnasium's own drawing.
"""

from __future__ import annotations

import argparse
import shutil
import tempfile
from pathlib import Path

import imageio.v2 as imageio
import numpy as np
from gymnasium.wrappers import RecordVideo
from stable_baselines3 import PPO

from lander.envs import make_env
from lander.evaluate import EVAL_SEED0, EpisodeResult, classify_outcome, run_episode
from lander.scene import SceneRenderer

MAX_MB = 5.0
FPS = 50


def record_classic(model: PPO, out: Path, seed: int) -> EpisodeResult:
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
    return result


def record_scene(model: PPO, out: Path, seed: int) -> EpisodeResult:
    scene = SceneRenderer()
    env = make_env("baseline", render_mode="rgb_array")
    obs, _ = env.reset(seed=seed)
    frames, ret, t, terminated, truncated = [], 0.0, 0, False, False
    while not (terminated or truncated):
        action = int(model.predict(obs, deterministic=True)[0])
        obs, reward, terminated, truncated, _ = env.step(action)
        ret += float(reward)
        t += 1
        env.render()  # keeps Gymnasium's particle bookkeeping identical to the classic path
        frames.append(scene.render(env, action, t))
    outcome = classify_outcome(env, terminated, truncated)
    tail = range(35) if outcome == "crashed" else range(25)  # short hold on the ending
    frames += [scene.render(env, 0, t + k, crash_frame=k if outcome == "crashed" else None) for k in tail]
    env.close()
    imageio.mimwrite(out, np.asarray(frames), fps=FPS, codec="libx264", quality=7, macro_block_size=8)
    return EpisodeResult(seed, ret, t, terminated, truncated, outcome)


def record(run_dir: Path, out: Path, seed: int = EVAL_SEED0, model_name: str = "model.zip",
           style: str = "scene") -> Path:
    model = PPO.load(run_dir / model_name, device="cpu")
    out.parent.mkdir(parents=True, exist_ok=True)
    result = record_scene(model, out, seed) if style == "scene" else record_classic(model, out, seed)
    size_mb = out.stat().st_size / 1e6
    print(f"{out}  seed={seed}  return={result.ret:.1f}  length={result.length}  "
          f"outcome={result.outcome}  size={size_mb:.2f} MB  style={style}")
    if size_mb > MAX_MB:
        print(f"  warning: above {MAX_MB} MB, do not commit as is")
    return out


def main(argv: list[str] | None = None) -> None:
    p = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    p.add_argument("run_dir", type=Path)
    p.add_argument("--out", type=Path, required=True)
    p.add_argument("--seed", type=int, default=EVAL_SEED0)
    p.add_argument("--model", default="model.zip")
    p.add_argument("--style", choices=["scene", "classic"], default="scene")
    args = p.parse_args(argv)
    record(args.run_dir, args.out, args.seed, args.model, args.style)


if __name__ == "__main__":
    main()
