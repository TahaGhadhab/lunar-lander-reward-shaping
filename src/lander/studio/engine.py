"""Background jobs behind the studio: live training, the live rocket view, evaluation and unit tests.

Everything that happens is published on a `Bus` as small JSON events, streamed to the browser (SSE).
Training uses the same env factory, rewards and PPO as `lander.train`; the live view and the evaluation fly a
*copy* of the policy on the plain env, so watching never touches what is being learned.
"""

from __future__ import annotations

import ast
import base64
import copy
import io
import json
import queue
import re
import subprocess
import sys
import threading
import time
import traceback
from dataclasses import asdict
from datetime import datetime
from pathlib import Path

import numpy as np
import torch
from PIL import Image
from stable_baselines3 import PPO
from stable_baselines3.common.callbacks import BaseCallback
from stable_baselines3.common.logger import Logger

from lander.envs import make_env, make_vec_env
from lander.evaluate import EpisodeResult, classify_outcome, summarize
from lander.introspect import actor_weights, policy_snapshot
from lander.scene import SceneRenderer
from lander.studio.params import reward_kwargs, validate
from lander.train import get_versions, git_hash

ROOT = Path(__file__).resolve().parents[3]
ACTIVATIONS = {
    "tanh": torch.nn.Tanh,
    "relu": torch.nn.ReLU,
    "elu": torch.nn.ELU,
    "leaky_relu": torch.nn.LeakyReLU,
}
MAX_DRAWN = 24  # neurons drawn per layer in the network view
PHYSICS_FPS = 50


# ---------------------------------------------------------------- event bus


class Bus:
    """Fan-out of server-sent events to every connected browser tab."""

    def __init__(self):
        self._subs: list[queue.Queue] = []
        self._lock = threading.Lock()

    def subscribe(self) -> queue.Queue:
        q: queue.Queue = queue.Queue(maxsize=600)
        with self._lock:
            self._subs.append(q)
        return q

    def unsubscribe(self, q: queue.Queue) -> None:
        with self._lock:
            if q in self._subs:
                self._subs.remove(q)

    def publish(self, kind: str, data) -> None:
        msg = f"event: {kind}\ndata: {json.dumps(data, separators=(',', ':'))}\n\n"
        with self._lock:
            subs = list(self._subs)
        for q in subs:
            try:
                q.put_nowait(msg)
            except queue.Full:  # a slow tab drops frames rather than blocking training
                pass


# ---------------------------------------------------------------- helpers


def jpeg_b64(frame: np.ndarray, quality: int = 78) -> str:
    buf = io.BytesIO()
    Image.fromarray(frame).save(buf, format="JPEG", quality=quality)
    return base64.b64encode(buf.getvalue()).decode()


def r(x, nd: int = 3):
    return np.round(np.asarray(x, dtype=float), nd).tolist()


def net_layout(policy) -> dict:
    """Layer sizes, which neurons are drawn (at most MAX_DRAWN per layer) and the weights between them."""
    ws = actor_weights(policy)
    sizes = [ws[0].shape[1]] + [w.shape[0] for w in ws]
    idx = [
        list(range(n)) if n <= MAX_DRAWN else np.linspace(0, n - 1, MAX_DRAWN).round().astype(int).tolist()
        for n in sizes
    ]
    sub = [r(w[np.ix_(idx[i + 1], idx[i])]) for i, w in enumerate(ws)]
    return {"sizes": sizes, "idx": idx, "w": sub}


def weight_names(policy) -> dict[str, str]:
    """Readable names for every weight matrix: actor L1.., action head, critic L1.., value head."""
    names, a, c = {}, 0, 0
    for name, _ in policy.named_parameters():
        if not name.endswith("weight"):
            continue
        if name.startswith("mlp_extractor.policy_net"):
            a += 1
            names[name] = f"actor L{a}"
        elif name.startswith("mlp_extractor.value_net"):
            c += 1
            names[name] = f"critic L{c}"
        elif name.startswith("action_net"):
            names[name] = "action head"
        elif name.startswith("value_net"):
            names[name] = "value head"
    return names


def weight_stats(policy, prev: dict | None) -> tuple[dict, dict]:
    names = weight_names(policy)
    cur = {n: p.detach().numpy().copy() for n, p in policy.named_parameters() if n in names}
    ws = actor_weights(policy)
    actor = np.concatenate([w.ravel() for w in ws])
    w1 = ws[0]
    rows = w1 if w1.shape[0] <= 64 else w1[np.linspace(0, w1.shape[0] - 1, 64).round().astype(int)]
    change = None
    if prev is not None:
        change = round(float(np.sqrt(sum(np.sum((cur[n] - prev[n]) ** 2) for n in cur))), 4)
    stats = {
        "norms": {names[n]: round(float(np.linalg.norm(w)), 3) for n, w in cur.items()},
        "change": change,
        "hist": np.histogram(actor, bins=40, range=(-1.2, 1.2))[0].tolist(),
        "w1": r(rows, 2),
    }
    return stats, cur


def test_docs(tests_dir: Path) -> dict[tuple[str, str], str]:
    """(file name, test function) -> first line of its docstring."""
    docs = {}
    for path in tests_dir.glob("test_*.py"):
        for node in ast.walk(ast.parse(path.read_text(encoding="utf-8"))):
            if isinstance(node, ast.FunctionDef) and node.name.startswith("test_"):
                doc = ast.get_docstring(node)
                if doc:
                    docs[(path.name, node.name)] = doc.strip().splitlines()[0]
    return docs


def list_models(root: Path = ROOT) -> list[dict]:
    runs = root / "runs"
    out = []
    for path in sorted(runs.glob("*/seed*/model.zip")):
        if path.parts[-3] in ("progression", "_dry"):
            continue
        out.append(
            {
                "path": path.relative_to(root).as_posix(),
                "label": f"{path.parts[-3]} · {path.parts[-2]} (final)",
            }
        )
    ckpts = sorted(
        runs.glob("progression/*/seed*/checkpoints/model_*_steps.zip"),
        key=lambda p: (p.parts[-4], int(re.findall(r"\d+", p.name)[0])),
    )
    for path in ckpts:
        steps = int(re.findall(r"\d+", path.name)[0])
        out.append(
            {
                "path": path.relative_to(root).as_posix(),
                "label": f"{path.parts[-4]} · {path.parts[-3]} after {steps // 1000}k steps",
            }
        )
    for path in sorted(runs.glob("studio/*/model.zip")):
        out.append({"path": path.relative_to(root).as_posix(), "label": f"studio · {path.parts[-2]}"})
    return out


def safe_model_path(rel: str, root: Path = ROOT) -> Path:
    path = (root / rel).resolve()
    if not (path.suffix == ".zip" and path.is_file() and (root / "runs").resolve() in path.parents):
        raise ValueError("model must be a .zip under runs/")
    return path


def parse_failures(lines: list[str], order: list[str]) -> dict[str, str]:
    """Map test id -> its short traceback, from pytest's FAILURES / ERRORS sections."""
    by_name = {i.split("::", 1)[1]: i for i in order}
    header = re.compile(r"^_{3,} (?:ERROR at \w+ of )?(\S+) _{3,}$")
    section = re.compile(r"^={3,} (FAILURES|ERRORS) ={3,}$")
    out, current, buf, inside = {}, None, [], False
    for line in lines:
        m = header.match(line) if inside else None
        if section.match(line) or m or (inside and line.startswith("====")):
            if current:
                out[current] = "\n".join(buf).strip()
            current, buf = (by_name.get(m.group(1)) if m else None), []
            inside = bool(section.match(line)) or bool(m)
            continue
        if current:
            buf.append(line)
    if current:
        out[current] = "\n".join(buf).strip()
    return out


# ---------------------------------------------------------------- the studio


class Studio:
    def __init__(self, root: Path = ROOT):
        torch.set_num_threads(1)
        self.root = root
        self.bus = Bus()
        self.pilot_lock = threading.Lock()
        self.pilot: dict | None = None  # {"policy", "label", "layout", "training"}
        self.train_pilot: dict | None = None
        self.view = {"enabled": True, "speed": 1.0}
        self.train_state: dict = {"status": "idle"}
        self.history: dict = {"progress": [], "losses": [], "weights": [], "last_weights": None}
        self.stop_train = threading.Event()
        self.train_thread: threading.Thread | None = None
        self.eval_state: dict = {"status": "idle", "episodes": [], "summary": None}
        self.stop_eval = threading.Event()
        self.eval_running = False
        self.tests: dict = {"items": [], "status": {}, "failures": {}, "running": False, "summary": None}
        self.progression: dict | None = None
        self._load_default_pilot()
        threading.Thread(target=self._viewer, daemon=True, name="viewer").start()

    # ---------- pilots (what the live view flies)
    def _load_default_pilot(self) -> None:
        default = self.root / "runs" / "baseline" / "seed2" / "model.zip"
        if default.exists():
            self.set_pilot(
                PPO.load(default, device="cpu").policy,
                "baseline · seed2 (final)",
                path="runs/baseline/seed2/model.zip",
            )
        else:
            env = make_vec_env("baseline", 0, 1, vec="dummy")
            self.set_pilot(PPO("MlpPolicy", env, device="cpu").policy, "untrained network (random weights)")

    def _publish_pilot(self, pilot: dict) -> None:
        self.bus.publish(
            "pilot",
            {
                "label": pilot["label"],
                "training": pilot["training"],
                "layout": pilot["layout"],
                "path": pilot.get("path"),
            },
        )

    def set_pilot(self, policy, label: str, training: bool = False, path: str | None = None) -> None:
        with self.pilot_lock:
            self.pilot = {
                "policy": policy,
                "label": label,
                "layout": net_layout(policy),
                "training": training,
                "path": path,
            }
            if training:
                self.train_pilot = self.pilot
        self._publish_pilot(self.pilot)

    def load_pilot(self, rel: str) -> str:
        if rel == "training":
            if not self.train_pilot:
                raise ValueError("no training in this session yet")
            with self.pilot_lock:
                self.pilot = self.train_pilot
            self._publish_pilot(self.pilot)
            return self.pilot["label"]
        path = safe_model_path(rel, self.root)
        label = next((m["label"] for m in list_models(self.root) if m["path"] == rel), rel)
        self.set_pilot(PPO.load(path, device="cpu").policy, label, path=rel)
        return label

    # ---------- flying one episode (live view and evaluation share this)
    def fly(
        self,
        env,
        scene,
        seed: int,
        get_pilot,
        deterministic: bool,
        speed,
        stop,
        source: str,
        episode: int,
        rng: np.random.Generator,
    ) -> EpisodeResult | None:
        """Play one episode. speed() > 0 streams frames at that multiple of real time; 0 = full speed."""
        obs, _ = env.reset(seed=seed)
        t, ret, terminated, truncated = 0, 0.0, False, False
        start = time.perf_counter()
        while not (terminated or truncated):
            if stop():
                return None
            pilot = get_pilot()
            with self.pilot_lock:
                snap = policy_snapshot(pilot["policy"], obs)
            probs = snap["probs"]
            action = int(np.argmax(probs)) if deterministic else int(rng.choice(4, p=probs / probs.sum()))
            prev_obs = obs
            obs, reward, terminated, truncated, _ = env.step(action)
            ret += float(reward)
            t += 1
            s = speed()
            if s > 0:
                env.render()  # Gymnasium's particle bookkeeping (particles never touch the lander)
                every = 1 if s < 1 else max(2, round(2 * s))
                if t % every == 0 or terminated or truncated:
                    idx = pilot["layout"]["idx"]
                    act = [r(np.asarray(h)[idx[i + 1]], 2) for i, h in enumerate(snap["hidden"])]
                    self.bus.publish(
                        "frame",
                        {
                            "src": source,
                            "img": jpeg_b64(scene.render(env, action, t)),
                            "t": t,
                            "ret": round(ret, 1),
                            "action": action,
                            "probs": r(probs),
                            "value": round(snap["value"], 1),
                            "obs": r(prev_obs),
                            "act": act,
                            "label": pilot["label"],
                            "episode": episode,
                            "seed": seed,
                        },
                    )
                lag = t / (PHYSICS_FPS * s) - (time.perf_counter() - start)
                if lag > 0:
                    time.sleep(lag)
        outcome = classify_outcome(env, terminated, truncated)
        if speed() > 0:
            for k in range(22 if outcome == "crashed" else 12):  # hold the ending on screen
                if stop():
                    break
                frame = scene.render(env, 0, t + k, crash_frame=k if outcome == "crashed" else None)
                self.bus.publish(
                    "frame",
                    {
                        "src": source,
                        "img": jpeg_b64(frame),
                        "t": t,
                        "ret": round(ret, 1),
                        "outcome": outcome,
                        "label": get_pilot()["label"],
                        "episode": episode,
                        "seed": seed,
                    },
                )
                time.sleep(0.04)
        return EpisodeResult(seed, ret, t, terminated, truncated, outcome)

    def _viewer(self) -> None:
        torch.set_num_threads(1)
        scene = SceneRenderer()
        env = make_env("baseline", render_mode="rgb_array")
        rng = np.random.default_rng(0)
        seed, episode = 20_000, 0

        def idle():
            return not self.view["enabled"] or self.eval_running

        while True:
            if idle() or self.pilot is None:
                time.sleep(0.2)
                continue
            episode += 1
            try:
                res = self.fly(
                    env,
                    scene,
                    seed,
                    lambda: self.pilot,
                    True,
                    lambda: self.view["speed"],
                    idle,
                    "view",
                    episode,
                    rng,
                )
            except Exception:  # never let the view thread die
                traceback.print_exc()
                time.sleep(1)
                continue
            if res:
                self.bus.publish(
                    "view_episode", {**asdict(res), "label": self.pilot["label"], "episode": episode}
                )
            seed += 1

    # ---------- training
    def start_training(self, raw: dict) -> dict:
        if self.train_thread and self.train_thread.is_alive():
            raise ValueError("a training run is already in progress")
        cfg = validate(raw)
        self.stop_train.clear()
        self.history = {"progress": [], "losses": [], "weights": [], "last_weights": None}
        self.train_state = {"status": "starting", "cfg": cfg}
        self.train_thread = threading.Thread(target=self._train, args=(cfg,), daemon=True, name="train")
        self.train_thread.start()
        return cfg

    def record(self, kind: str, data: dict) -> None:
        self.history[kind].append(data)
        self.bus.publish(kind, data)

    def _train(self, cfg: dict) -> None:
        torch.set_num_threads(1)
        stamp = datetime.now().strftime("%Y%m%d-%H%M%S")
        run_dir = self.root / "runs" / "studio" / f"{stamp}_{cfg['reward']}_s{cfg['seed']}"
        env = None
        try:
            rkw = reward_kwargs(cfg)
            env = make_vec_env(
                cfg["reward"],
                cfg["seed"],
                cfg["n_envs"],
                monitor_dir=run_dir / "monitor",
                vec="dummy",
                reward_kwargs=rkw,
            )
            ppo = {
                k: cfg[k]
                for k in (
                    "learning_rate",
                    "n_steps",
                    "batch_size",
                    "n_epochs",
                    "gamma",
                    "gae_lambda",
                    "ent_coef",
                    "clip_range",
                    "vf_coef",
                    "max_grad_norm",
                )
            }
            policy_kwargs = {
                "net_arch": {"pi": cfg["actor_layers"], "vf": cfg["critic_layers"]},
                "activation_fn": ACTIVATIONS[cfg["activation"]],
            }
            model = PPO(
                "MlpPolicy",
                env,
                seed=cfg["seed"],
                device="cpu",
                verbose=0,
                policy_kwargs=policy_kwargs,
                **ppo,
            )
            model.set_logger(Logger(folder=None, output_formats=[]))
            n_params = int(sum(p.numel() for p in model.policy.parameters()))
            self.train_state = {
                "status": "running",
                "cfg": cfg,
                "n_params": n_params,
                "run_dir": run_dir.relative_to(self.root).as_posix(),
                "total": cfg["total_timesteps"],
            }
            self.bus.publish("train_start", self.train_state)
            self.set_pilot(copy.deepcopy(model.policy), "training: untrained (step 0)", training=True)
            started = datetime.now().astimezone().isoformat(timespec="seconds")
            t0 = time.perf_counter()
            model.learn(cfg["total_timesteps"], callback=LiveCallback(self, cfg))
            elapsed = time.perf_counter() - t0
            stopped = self.stop_train.is_set()
            model.save(run_dir / "model.zip")
            config = {
                "source": "studio",
                "env_id": "LunarLander-v3",
                "reward": cfg["reward"],
                "reward_params": rkw,
                "seed": cfg["seed"],
                "total_timesteps": cfg["total_timesteps"],
                "n_envs": cfg["n_envs"],
                "vec_env": "dummy",
                "device": "cpu",
                "policy": "MlpPolicy",
                "ppo": ppo,
                "policy_kwargs": {"net_arch": policy_kwargs["net_arch"], "activation": cfg["activation"]},
                "versions": get_versions(),
                "git": git_hash(),
                "started": started,
                "finished": datetime.now().astimezone().isoformat(timespec="seconds"),
                "wall_clock_s": round(elapsed, 1),
                "timesteps_done": int(model.num_timesteps),
                "stopped_early": stopped,
            }
            (run_dir / "config.json").write_text(json.dumps(config, indent=2))
            self.train_state.update(
                status="stopped" if stopped else "done",
                elapsed=round(elapsed, 1),
                timesteps=int(model.num_timesteps),
                model=(run_dir / "model.zip").relative_to(self.root).as_posix(),
            )
            with self.pilot_lock:
                if self.train_pilot:
                    self.train_pilot["label"] = (
                        f"trained in studio: {int(model.num_timesteps) // 1000}k steps"
                    )
            self.bus.publish("train_done", self.train_state)
        except Exception as exc:
            traceback.print_exc()
            self.train_state = {"status": "error", "message": str(exc)}
            self.bus.publish("train_error", self.train_state)
        finally:
            if env is not None:
                env.close()

    def stop_training(self) -> None:
        self.stop_train.set()

    # ---------- evaluation ("test the agent")
    def start_eval(self, raw: dict) -> dict:
        if self.eval_running:
            raise ValueError("an evaluation is already running")
        cfg = validate({k: raw[k] for k in ("eval_episodes", "eval_seed", "deterministic") if k in raw})
        source = raw.get("model", "pilot")
        if source == "pilot":
            with self.pilot_lock:
                policy, label = copy.deepcopy(self.pilot["policy"]), self.pilot["label"]
        else:
            policy = PPO.load(safe_model_path(source, self.root), device="cpu").policy
            label = next((m["label"] for m in list_models(self.root) if m["path"] == source), source)
        watch = bool(raw.get("watch", True))
        speed = max(0.25, min(16.0, float(raw.get("speed", 4.0))))
        self.stop_eval.clear()
        self.eval_running = True
        job = {"policy": policy, "label": label, "layout": net_layout(policy), "training": False}
        threading.Thread(target=self._eval, args=(cfg, job, watch, speed), daemon=True, name="eval").start()
        return {"label": label, **{k: cfg[k] for k in ("eval_episodes", "eval_seed", "deterministic")}}

    def _eval(self, cfg: dict, pilot: dict, watch: bool, speed: float) -> None:
        torch.set_num_threads(1)
        try:
            n, seed0 = cfg["eval_episodes"], cfg["eval_seed"]
            self.eval_state = {
                "status": "running",
                "label": pilot["label"],
                "n": n,
                "seed0": seed0,
                "deterministic": cfg["deterministic"],
                "watch": watch,
                "episodes": [],
                "summary": None,
            }
            self.bus.publish("eval_start", self.eval_state)
            if watch:
                self._publish_pilot(pilot)
            scene = SceneRenderer() if watch else None
            env = make_env("baseline", render_mode="rgb_array" if watch else None)
            rng = np.random.default_rng(seed0)
            results = []
            for i in range(n):
                res = self.fly(
                    env,
                    scene,
                    seed0 + i,
                    lambda: pilot,
                    cfg["deterministic"],
                    lambda: speed if watch else 0,
                    self.stop_eval.is_set,
                    "eval",
                    i + 1,
                    rng,
                )
                if res is None:
                    break
                results.append(res)
                ep = {**asdict(res), "i": i + 1}
                self.eval_state["episodes"].append(ep)
                self.bus.publish("eval_episode", ep)
            env.close()
            self.eval_state.update(
                status="stopped" if self.stop_eval.is_set() else "done",
                summary=summarize(results) if results else None,
            )
            self.bus.publish("eval_done", self.eval_state)
        except Exception as exc:
            traceback.print_exc()
            self.eval_state = {"status": "error", "message": str(exc), "episodes": [], "summary": None}
            self.bus.publish("eval_error", self.eval_state)
        finally:
            self.eval_running = False
            if self.pilot:
                self._publish_pilot(self.pilot)

    # ---------- unit tests
    def collect_tests(self) -> list[dict]:
        out = subprocess.run(
            [sys.executable, "-m", "pytest", "--collect-only", "-q", "-p", "no:cacheprovider"],
            capture_output=True,
            text=True,
            cwd=self.root,
        )
        docs = test_docs(self.root / "tests")
        items = []
        for line in out.stdout.splitlines():
            line = line.strip()
            if "::" not in line:
                continue
            file_part, name = line.split("::", 1)
            func = name.split("[")[0]
            doc = docs.get((Path(file_part).name, func)) or func.removeprefix("test_").replace("_", " ") + "."
            items.append({"id": line, "file": Path(file_part).name, "name": name, "doc": doc})
        self.tests["items"] = items
        return items

    def run_tests(self, ids: list[str] | None) -> list[str]:
        if self.tests["running"]:
            raise ValueError("tests are already running")
        known = [t["id"] for t in (self.tests["items"] or self.collect_tests())]
        if ids:
            unknown = [i for i in ids if i not in known]
            if unknown:
                raise ValueError(f"unknown test id: {unknown[0]}")
            order = [i for i in known if i in set(ids)]
        else:
            order = known
        self.tests["running"] = True
        threading.Thread(target=self._run_tests, args=(order, not ids), daemon=True, name="tests").start()
        return order

    def _run_tests(self, order: list[str], everything: bool) -> None:
        status = self.tests["status"]
        for i in order:
            status[i] = "queued"
            self.tests["failures"].pop(i, None)
        if order:
            status[order[0]] = "running"
        self.bus.publish("tests_start", {"ids": order, "status": status})
        cmd = [
            sys.executable,
            "-u",
            "-m",
            "pytest",
            "-v",
            "--tb=short",
            "--color=no",
            "-p",
            "no:cacheprovider",
        ]
        cmd += [] if everything else order
        start = time.perf_counter()
        lines: list[str] = []
        try:
            proc = subprocess.Popen(
                cmd,
                cwd=self.root,
                stdout=subprocess.PIPE,
                stderr=subprocess.STDOUT,
                text=True,
                bufsize=1,
                encoding="utf-8",
                errors="replace",
            )
            pattern = re.compile(r"^(tests/\S+::\S+)\s+(PASSED|FAILED|ERROR|SKIPPED|XFAIL|XPASS)")
            pending = list(order)
            for line in proc.stdout:
                lines.append(line.rstrip("\n"))
                m = pattern.match(line)
                if not m:
                    continue
                test_id, result = m.group(1), m.group(2).lower()
                status[test_id] = "passed" if result in ("passed", "xfail", "xpass") else result
                if test_id in pending:
                    pending.remove(test_id)
                nxt = pending[0] if pending else None
                if nxt:
                    status[nxt] = "running"
                self.bus.publish("test_result", {"id": test_id, "status": status[test_id], "next": nxt})
            proc.wait()
        finally:
            failures = parse_failures(lines, order)
            self.tests["failures"].update(failures)
            for test_id in order:
                if status.get(test_id) in ("queued", "running"):
                    status[test_id] = "error"
            counts = {
                k: sum(1 for i in order if status.get(i) == k)
                for k in ("passed", "failed", "error", "skipped")
            }
            self.tests["summary"] = {
                **counts,
                "duration": round(time.perf_counter() - start, 2),
                "ran": len(order),
            }
            self.tests["running"] = False
            self.bus.publish(
                "tests_done", {"summary": self.tests["summary"], "failures": failures, "status": status}
            )

    # ---------- recorded checkpoint progression (computed once, cached on disk)
    def load_progression(self) -> None:
        from lander.dashboard import collect_progression

        try:
            self.progression = collect_progression(self.root / "runs", self.root / "dashboard" / "cache")
        except Exception:
            traceback.print_exc()
            self.progression = {}
        self.bus.publish("progression_ready", {"runs": len(self.progression)})

    # ---------- snapshot for a newly opened page
    def state(self) -> dict:
        with self.pilot_lock:
            pilot = (
                None
                if self.pilot is None
                else {
                    "label": self.pilot["label"],
                    "training": self.pilot["training"],
                    "layout": self.pilot["layout"],
                    "path": self.pilot.get("path"),
                }
            )
        return {
            "train": self.train_state,
            "history": self.history,
            "pilot": pilot,
            "view": self.view,
            "eval": self.eval_state,
            "tests": self.tests,
        }


class LiveCallback(BaseCallback):
    """Streams training progress, losses and weights; refreshes the live view's copy of the policy."""

    def __init__(self, studio: Studio, cfg: dict):
        super().__init__()
        self.studio, self.cfg = studio, cfg
        self.shaped: list[float] = []
        self.prev_w = None
        self.t0 = time.perf_counter()

    def _publish_weights(self) -> None:
        stats, self.prev_w = weight_stats(self.model.policy, self.prev_w)
        stats["t"] = int(self.num_timesteps)
        self.studio.history["last_weights"] = stats
        self.studio.record("weights", {"t": stats["t"], "norms": stats["norms"], "change": stats["change"]})
        self.studio.bus.publish("weights_detail", stats)

    def _on_training_start(self) -> None:
        self._publish_weights()

    def _on_step(self) -> bool:
        for info in self.locals["infos"]:
            if "shaped_episode" in info:
                self.shaped.append(info["shaped_episode"]["r"])
        return not self.studio.stop_train.is_set()

    def _on_rollout_start(self) -> None:
        v = self.model.logger.name_to_value
        if "train/value_loss" in v:
            self.studio.record(
                "losses",
                {
                    "t": int(self.num_timesteps),
                    "policy_loss": round(float(v.get("train/policy_gradient_loss", 0)), 5),
                    "value_loss": round(float(v.get("train/value_loss", 0)), 3),
                    "entropy": round(-float(v.get("train/entropy_loss", 0)), 4),
                    "approx_kl": round(float(v.get("train/approx_kl", 0)), 5),
                    "clip_fraction": round(float(v.get("train/clip_fraction", 0)), 4),
                    "explained_variance": round(float(v.get("train/explained_variance", 0)), 4),
                },
            )

    def _on_rollout_end(self) -> None:
        eps = list(self.model.ep_info_buffer)
        elapsed = time.perf_counter() - self.t0
        prog = {
            "t": int(self.num_timesteps),
            "total": self.cfg["total_timesteps"],
            "elapsed": round(elapsed, 1),
            "fps": round(self.num_timesteps / max(elapsed, 1e-6)),
            "episodes": len(eps),
        }
        if eps:
            lengths = np.array([e["l"] for e in eps])
            prog.update(
                ep_rew_mean=round(float(np.mean([e["r"] for e in eps])), 1),
                ep_len_mean=round(float(lengths.mean()), 1),
                trunc_rate=round(float((lengths >= 1000).mean()), 3),
            )
        if self.shaped:
            prog["shaped_rew_mean"] = round(float(np.mean(self.shaped[-100:])), 1)
        self.studio.record("progress", prog)
        self._publish_weights()
        studio, current = self.studio, False
        with studio.pilot_lock:  # the live view now flies the freshly updated policy
            if studio.train_pilot:
                studio.train_pilot["policy"].load_state_dict(self.model.policy.state_dict())
                studio.train_pilot["label"] = f"training: after {self.num_timesteps // 1000}k steps"
                studio.train_pilot["layout"] = net_layout(studio.train_pilot["policy"])
                current = studio.pilot is studio.train_pilot
        if current:
            studio._publish_pilot(studio.train_pilot)
