"""Learning curves from Monitor CSVs: plain return and episode length, mean +/- std across seeds.

Example: python -m lander.plots --rewards baseline upright_slow_bonus upright_slow_potential --out media/
Monitor sits on the plain env, so these curves are the plain return for every reward (docs/DESIGN.md 5.1).
"""

from __future__ import annotations

import argparse
from pathlib import Path

import matplotlib

matplotlib.use("Agg")
import matplotlib.pyplot as plt  # noqa: E402
import numpy as np  # noqa: E402
import pandas as pd  # noqa: E402

DEFAULT_REWARDS = [
    "baseline", "time_penalty", "upright_slow_bonus", "upright_slow_potential", "upright_slow_fixed",
]


def load_monitor(run_dir: Path) -> pd.DataFrame:
    """All episodes of one run, in completion order, with the cumulative timestep at which each ended."""
    frames = [pd.read_csv(f, skiprows=1) for f in sorted((run_dir / "monitor").glob("*.monitor.csv"))]
    if not frames:
        raise FileNotFoundError(f"no monitor CSVs in {run_dir / 'monitor'}")
    df = pd.concat(frames).sort_values("t").reset_index(drop=True)
    df["timesteps"] = df["l"].cumsum()
    return df


def to_grid(df: pd.DataFrame, grid: np.ndarray, column: str, window: int = 100) -> np.ndarray:
    """Rolling mean of the last `window` episodes at each grid timestep (NaN before the first episode)."""
    smooth = df[column].rolling(window, min_periods=1).mean().to_numpy()
    idx = np.searchsorted(df["timesteps"].to_numpy(), grid, side="right") - 1
    return np.where(idx >= 0, smooth[np.clip(idx, 0, None)], np.nan)


def seed_curves(runs_root: Path, reward: str, grid: np.ndarray, column: str) -> np.ndarray:
    runs = sorted((runs_root / reward).glob("seed*"))
    return np.array([to_grid(load_monitor(r), grid, column) for r in runs])


def plot(runs_root: Path, rewards: list[str], out_dir: Path, steps: int = 1_000_000) -> list[Path]:
    out_dir.mkdir(parents=True, exist_ok=True)
    grid = np.linspace(10_000, steps, 100)
    panels = [("r", "Plain return (rolling mean of 100 episodes)", "learning_curves_return.png"),
              ("l", "Episode length (frames; 1000 = time limit)", "learning_curves_length.png")]
    written = []
    for column, ylabel, name in panels:
        fig, ax = plt.subplots(figsize=(8, 4.5))
        for reward in rewards:
            if not (runs_root / reward).exists():
                continue
            curves = seed_curves(runs_root, reward, grid, column)
            mean, std = np.nanmean(curves, axis=0), np.nanstd(curves, axis=0)
            line, = ax.plot(grid, mean, label=f"{reward} (n={len(curves)})")
            ax.fill_between(grid, mean - std, mean + std, color=line.get_color(), alpha=0.2)
        if column == "r":
            ax.axhline(200, color="grey", linestyle="--", linewidth=1, label="solved (200)")
        ax.set_xlabel("Training timesteps")
        ax.set_ylabel(ylabel)
        ax.grid(alpha=0.3)
        ax.legend(fontsize=8)
        fig.tight_layout()
        path = out_dir / name
        fig.savefig(path, dpi=120)
        plt.close(fig)
        written.append(path)
    return written


def main(argv: list[str] | None = None) -> None:
    p = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    p.add_argument("--runs", type=Path, default=Path("runs"))
    p.add_argument("--rewards", nargs="+", default=DEFAULT_REWARDS)
    p.add_argument("--out", type=Path, default=Path("media"))
    args = p.parse_args(argv)
    for path in plot(args.runs, args.rewards, args.out):
        print(path)


if __name__ == "__main__":
    main()
