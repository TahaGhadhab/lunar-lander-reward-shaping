"""Build the machine-learning dashboard: one local HTML page with results, curves, progression,
the network replayed frame by frame next to the video, and the test suite.

  python -m lander.dashboard            # builds dashboard/index.html and opens it in the browser
  python -m lander.dashboard --no-open  # build only

Reads runs/ (eval.json, monitor CSVs, progress.csv, checkpoints). Progression evaluations are cached in
dashboard/cache/ so rebuilding is fast.
"""

from __future__ import annotations

import argparse
import base64
import io
import json
import subprocess
import sys
import tempfile
import webbrowser
import xml.etree.ElementTree as ET
from pathlib import Path

import numpy as np
import pandas as pd
from PIL import Image
from stable_baselines3 import PPO

from lander.envs import make_env
from lander.evaluate import EVAL_SEED0, classify_outcome, run_episode, summarize
from lander.introspect import ACTION_NAMES, OBS_NAMES, list_checkpoints, policy_snapshot
from lander.plots import DEFAULT_REWARDS, load_monitor, to_grid
from lander.train import PPO_HYPERPARAMS

TEMPLATE = Path(__file__).with_name("dashboard_template.html")
REPLAYS = [  # (id, title, model zip, frame stride)
    ("baseline", "Baseline agent (built-in reward)", "runs/baseline/seed2/model.zip", 1),
    ("hack", "Hacked reward: +1 per upright frame (hovers)", "runs/upright_slow_bonus/seed2/model.zip", 2),
    ("fixed", "Potential-based fix", "runs/upright_slow_potential/seed0/model.zip", 1),
    ("early", "Baseline after 150k steps (still learning)",
     "runs/progression/baseline/seed0/checkpoints/model_150000_steps.zip", 1),
]


def r(x, nd=3):
    """Round floats (and arrays) for a compact JSON."""
    return np.round(np.asarray(x, dtype=float), nd).tolist()


# --- results and curves --------------------------------------------------------------------------

def collect_results(runs: Path, rewards: list[str]) -> dict:
    out = {}
    for reward in rewards:
        seeds = []
        for run_dir in sorted((runs / reward).glob("seed*")):
            ev = run_dir / "eval.json"
            if ev.exists():
                d = json.loads(ev.read_text())
                episodes = d.pop("episodes", [])
                d["seed"] = run_dir.name
                d["returns"] = r([e["ret"] for e in episodes], 1)
                seeds.append(d)
        if seeds:
            out[reward] = seeds
    return out


def collect_curves(runs: Path, rewards: list[str], steps: int = 1_000_000) -> dict:
    grid = np.linspace(10_000, steps, 100)
    curves = {"grid": r(grid, 0), "rewards": {}}
    for reward in rewards:
        run_dirs = sorted((runs / reward).glob("seed*"))
        if not run_dirs:
            continue
        entry = {}
        for col in ("r", "l"):
            c = np.array([to_grid(load_monitor(d), grid, col) for d in run_dirs])
            entry[col] = {"mean": r(np.nanmean(c, 0), 1), "std": r(np.nanstd(c, 0), 1)}
        shaped, plain, ts = [], [], None
        for d in run_dirs:
            prog = pd.read_csv(d / "progress.csv")
            if "rollout/shaped_ep_rew_mean" in prog:
                ts = prog["time/total_timesteps"].to_numpy()
                shaped.append(prog["rollout/shaped_ep_rew_mean"].to_numpy())
                plain.append(prog["rollout/ep_rew_mean"].to_numpy())
        if shaped:
            n = min(map(len, shaped))
            entry["shaped_vs_plain"] = {"t": r(ts[:n], 0),
                                        "shaped": r(np.mean([s[:n] for s in shaped], 0), 1),
                                        "plain": r(np.mean([p[:n] for p in plain], 0), 1)}
        curves["rewards"][reward] = entry
    return curves


# --- progression over checkpoints ----------------------------------------------------------------

def collect_progression(runs: Path, cache_dir: Path, n_episodes: int = 10) -> dict:
    cache_file = cache_dir / "progression.json"
    cache = json.loads(cache_file.read_text()) if cache_file.exists() else {}
    out = {}
    env = make_env("baseline")
    for run_dir in sorted((runs / "progression").glob("*/seed*")):
        label = f"{run_dir.parent.name}/{run_dir.name}"
        points = []
        prev_weights = None
        for steps, path in list_checkpoints(run_dir):
            key = f"{path}|{path.stat().st_mtime_ns}|{n_episodes}"
            model = PPO.load(path, device="cpu")
            weights = {name: p.detach().numpy().copy() for name, p in model.policy.named_parameters()}
            if key not in cache:
                print(f"  evaluating {label} @ {steps}")
                s = summarize([run_episode(model, env, EVAL_SEED0 + i) for i in range(n_episodes)])
                cache[key] = {k: s[k] for k in ("mean_return", "landed_rate", "crashed_rate",
                                                "out_of_bounds_rate", "timeout_rate", "mean_length")}
            point = {"steps": steps, **cache[key]}
            point["weight_norms"] = {
                "actor L1": float(np.linalg.norm(weights["mlp_extractor.policy_net.0.weight"])),
                "actor L2": float(np.linalg.norm(weights["mlp_extractor.policy_net.2.weight"])),
                "action head": float(np.linalg.norm(weights["action_net.weight"])),
                "critic L1": float(np.linalg.norm(weights["mlp_extractor.value_net.0.weight"])),
                "value head": float(np.linalg.norm(weights["value_net.weight"])),
            }
            if prev_weights is not None:
                point["weight_change"] = float(np.sqrt(sum(
                    np.sum((weights[k] - prev_weights[k]) ** 2) for k in weights)))
            prev_weights = weights
            points.append(point)
        out[label] = points
    env.close()
    cache_dir.mkdir(parents=True, exist_ok=True)
    cache_file.write_text(json.dumps(cache))
    return out


# --- network structure and replays ---------------------------------------------------------------

def network_info(model: PPO) -> dict:
    sd = {k: v.detach().numpy() for k, v in model.policy.state_dict().items()}
    w1 = sd["mlp_extractor.policy_net.0.weight"]  # (64, 8)
    w2 = sd["mlp_extractor.policy_net.2.weight"]  # (64, 64)
    w3 = sd["action_net.weight"]  # (4, 64)
    n_params = int(sum(v.size for v in sd.values()))
    return {"w1": r(w1), "w2": r(w2), "w3": r(w3), "n_params": n_params,
            "input_importance": r(np.abs(w1).mean(0)),
            "layers": [{"name": k, "shape": list(v.shape)} for k, v in sd.items()]}


def frame_to_b64(frame: np.ndarray, width: int = 300, quality: int = 60) -> str:
    img = Image.fromarray(frame).resize((width, int(frame.shape[0] * width / frame.shape[1])))
    buf = io.BytesIO()
    img.save(buf, format="JPEG", quality=quality)
    return base64.b64encode(buf.getvalue()).decode()


def record_replay(model_path: Path, stride: int, seed: int = EVAL_SEED0) -> dict:
    model = PPO.load(model_path, device="cpu")
    env = make_env("baseline", render_mode="rgb_array")
    obs, _ = env.reset(seed=seed)
    frames, steps = [], []
    t, ret, done = 0, 0.0, False
    while not done:
        snap = policy_snapshot(model, obs)
        action = int(np.argmax(snap["probs"]))
        if t % stride == 0:
            frames.append(frame_to_b64(env.render()))
            steps.append({"t": t, "obs": r(obs), "h1": r(snap["h1"], 2), "h2": r(snap["h2"], 2),
                          "p": r(snap["probs"]), "v": round(snap["value"], 1), "a": action,
                          "ret": round(ret, 1)})
        obs, reward, terminated, truncated, _ = env.step(action)
        ret += float(reward)
        t += 1
        done = terminated or truncated
    outcome = classify_outcome(env, terminated, truncated)
    env.close()
    return {"frames": frames, "steps": steps, "outcome": outcome, "return": round(ret, 1), "length": t,
            "network": network_info(model)}


# --- tests ----------------------------------------------------------------------------------------

def collect_tests() -> dict:
    with tempfile.TemporaryDirectory() as tmp:
        xml_path = Path(tmp) / "junit.xml"
        subprocess.run([sys.executable, "-m", "pytest", "-q", f"--junitxml={xml_path}"],
                       capture_output=True, text=True)
        root = ET.parse(xml_path).getroot()  # trusted: written by our own pytest run just above
    cases = []
    for case in root.iter("testcase"):
        status = "passed"
        for child in case:
            if child.tag in ("failure", "error"):
                status = "failed"
            elif child.tag == "skipped":
                status = "skipped"
        cases.append({"file": case.get("classname", "").split(".")[-1], "name": case.get("name"),
                      "time": round(float(case.get("time", 0)), 3), "status": status})
    return {"cases": cases, "passed": sum(c["status"] == "passed" for c in cases),
            "failed": sum(c["status"] == "failed" for c in cases)}


def build(runs: Path, out_dir: Path) -> Path:
    print("collecting results and curves")
    data = {
        "results": collect_results(runs, DEFAULT_REWARDS),
        "curves": collect_curves(runs, DEFAULT_REWARDS),
        "hyperparams": PPO_HYPERPARAMS,
        "action_names": ACTION_NAMES,
        "obs_names": OBS_NAMES,
    }
    print("evaluating checkpoints (cached after the first build)")
    data["progression"] = collect_progression(runs, out_dir / "cache")
    data["replays"] = {}
    for rid, title, path, stride in REPLAYS:
        if Path(path).exists():
            print(f"recording replay: {title}")
            data["replays"][rid] = {"title": title, "path": path, "stride": stride,
                                    **record_replay(Path(path), stride)}
    print("running the test suite")
    data["tests"] = collect_tests()
    html = TEMPLATE.read_text(encoding="utf-8").replace("/*__DATA__*/null", json.dumps(data))
    out_dir.mkdir(parents=True, exist_ok=True)
    out = out_dir / "index.html"
    out.write_text(html, encoding="utf-8")
    print(f"wrote {out} ({out.stat().st_size / 1e6:.1f} MB)")
    return out


def main(argv: list[str] | None = None) -> None:
    p = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    p.add_argument("--runs", type=Path, default=Path("runs"))
    p.add_argument("--out", type=Path, default=Path("dashboard"))
    p.add_argument("--no-open", action="store_true")
    args = p.parse_args(argv)
    out = build(args.runs, args.out)
    if not args.no_open:
        webbrowser.open(out.resolve().as_uri())


if __name__ == "__main__":
    main()
