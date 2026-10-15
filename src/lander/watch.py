"""Watch the lander fly, live, with what the network is thinking next to it.

  python -m lander.watch --random                                        # untrained: tumbles and crashes
  python -m lander.watch --progression runs/progression/baseline/seed0   # every checkpoint, crash -> landing
  python -m lander.watch --model runs/upright_slow_bonus/seed2           # the hacked agent hovering
  python -m lander.watch --compare                                       # random, baseline, hack, fix in turn

Keys: SPACE pause, N or RIGHT next episode, UP/DOWN speed, ESC quit.
Always the plain env (built-in reward), deterministic policy, so the numbers match `lander.evaluate`.
"""

from __future__ import annotations

import argparse
import os
from dataclasses import dataclass
from pathlib import Path

import numpy as np
from stable_baselines3 import PPO

from lander.envs import make_env
from lander.evaluate import EVAL_SEED0, classify_outcome
from lander.introspect import ACTION_NAMES, OBS_NAMES, list_checkpoints, policy_snapshot
from lander.scene import SceneRenderer

GAME_W, GAME_H, PANEL_W, WIN_H = 600, 400, 380, 440
OUTCOME_COLORS = {"landed": (60, 200, 90), "crashed": (230, 70, 60), "out_of_bounds": (230, 140, 40),
                  "timeout": (240, 200, 40)}
COMPARE = [("random", None), ("baseline (fixed reward)", "runs/baseline/seed2"),
           ("hacked reward: +1 per upright frame", "runs/upright_slow_bonus/seed2"),
           ("fixed: potential-based", "runs/upright_slow_potential/seed0")]


@dataclass
class Pilot:
    label: str
    model: PPO | None  # None = random actions


def pilots_from_args(args: argparse.Namespace) -> list[Pilot]:
    if args.random:
        return [Pilot("random agent (untrained)", None)]
    if args.model:
        run = Path(args.model)
        return [Pilot(f"{run.parent.name} / {run.name}", PPO.load(run / "model.zip", device="cpu"))]
    if args.progression:
        run = Path(args.progression)
        ckpts = list_checkpoints(run)[:: args.stride]
        if not ckpts:
            raise SystemExit(f"no checkpoints in {run}; train with --checkpoint-every")
        return [Pilot(f"{run.parent.name}  after {steps / 1000:.0f}k training steps",
                      PPO.load(path, device="cpu")) for steps, path in ckpts]
    pilots = []
    for label, path in COMPARE:
        if path is None:
            pilots.append(Pilot(label, None))
        elif (Path(path) / "model.zip").exists():
            pilots.append(Pilot(label, PPO.load(Path(path) / "model.zip", device="cpu")))
    return pilots


class Viewer:
    def __init__(self, headless: bool = False):
        if headless:
            os.environ["SDL_VIDEODRIVER"] = "dummy"
        import pygame

        self.pg = pygame
        pygame.init()
        self.screen = pygame.display.set_mode((GAME_W + PANEL_W, WIN_H))
        pygame.display.set_caption("Lunar Lander: PPO agent")
        self.font = pygame.font.SysFont("consolas", 14)
        self.small = pygame.font.SysFont("consolas", 12)
        self.big = pygame.font.SysFont("consolas", 34, bold=True)
        self.clock = pygame.time.Clock()

    def text(self, s, pos, color=(230, 230, 230), font=None):
        self.screen.blit((font or self.font).render(s, True, color), pos)

    def bar(self, x, y, w, h, frac, color, signed=False):
        pg = self.pg
        pg.draw.rect(self.screen, (55, 55, 65), (x, y, w, h))
        if signed:
            mid = x + w // 2
            span = int(np.clip(frac, -1, 1) * w / 2)
            pg.draw.rect(self.screen, color, (min(mid, mid + span), y, abs(span), h))
            pg.draw.line(self.screen, (120, 120, 130), (mid, y), (mid, y + h))
        else:
            pg.draw.rect(self.screen, color, (x, y, int(np.clip(frac, 0, 1) * w), h))

    def grid(self, values, x, y, cell=9):
        for i, v in enumerate(values):
            r, c = divmod(i, 8)
            v = float(np.clip(v, -1, 1))
            color = (int(60 + 190 * v), 70, 90) if v > 0 else (60, 90, int(90 + 160 * -v))
            self.pg.draw.rect(self.screen, color, (x + c * (cell + 1), y + r * (cell + 1), cell, cell))

    def draw(self, frame, pilot, ep, t, obs, snap, action, ret, speed, outcome=None):
        pg = self.pg
        self.screen.blit(pg.surfarray.make_surface(frame.swapaxes(0, 1)), (0, 0))
        if outcome:
            col = OUTCOME_COLORS[outcome]
            label = self.big.render(outcome.replace("_", " ").upper(), True, col)
            self.screen.blit(label, (GAME_W // 2 - label.get_width() // 2, 150))
        x0 = GAME_W + 12
        pg.draw.rect(self.screen, (25, 25, 32), (GAME_W, 0, PANEL_W, WIN_H))
        pg.draw.rect(self.screen, (15, 15, 20), (0, GAME_H, GAME_W, WIN_H - GAME_H))
        self.text("plain env reward, deterministic policy (same as lander.evaluate)", (10, GAME_H + 13),
                  (120, 120, 130), self.small)
        self.text(pilot.label[:46], (x0, 8), (255, 220, 120))
        self.text(f"episode {ep}   frame {t:4d}   speed x{speed:g}", (x0, 28))
        self.text(f"return so far {ret:8.1f}", (x0, 46), (150, 220, 255))
        value = f"{snap['value']:8.1f}" if snap else "     n/a"
        self.text(f"value V(s) {value}", (x0, 64), (150, 220, 255))
        self.text("action probabilities", (x0, 90), (180, 180, 190))
        probs = snap["probs"] if snap else np.full(4, 0.25)
        for i, name in enumerate(ACTION_NAMES):
            y = 108 + i * 18
            chosen = i == action
            color = (255, 255, 255) if chosen else (160, 160, 170)
            self.text(("> " if chosen else "  ") + name, (x0, y), color)
            self.bar(x0 + 150, y + 3, 150, 11, probs[i], (90, 200, 120) if chosen else (90, 110, 160))
            self.text(f"{probs[i]:.2f}", (x0 + 308, y), font=self.small)
        self.text("observation (8 inputs)", (x0, 186), (180, 180, 190))
        for i, name in enumerate(OBS_NAMES):
            y = 204 + i * 15
            self.text(f"{name:9s}{obs[i]:7.2f}", (x0, y), font=self.small)
            self.bar(x0 + 130, y + 3, 120, 9, obs[i] / 1.5, (200, 170, 90), signed=True)
        self.text("hidden layer 1 / 2 (64 tanh each)", (x0, 330), (180, 180, 190))
        if snap:
            self.grid(snap["h1"], x0, 350, cell=6)
            self.grid(snap["h2"], x0 + 80, 350, cell=6)
            self.text("red > 0, blue < 0", (x0 + 170, 370), (120, 120, 130), self.small)
        else:
            self.text("(random agent: no network)", (x0, 350), (120, 120, 130), self.small)
        self.text("SPACE pause  N next  UP/DOWN speed  ESC quit", (x0, 418), (110, 110, 120), self.small)
        pg.display.flip()

    def events(self, state):
        pg = self.pg
        for e in pg.event.get():
            if e.type == pg.QUIT or (e.type == pg.KEYDOWN and e.key == pg.K_ESCAPE):
                state["quit"] = True
            elif e.type == pg.KEYDOWN:
                if e.key == pg.K_SPACE:
                    state["paused"] = not state["paused"]
                elif e.key in (pg.K_n, pg.K_RIGHT):
                    state["skip"] = True
                elif e.key == pg.K_UP:
                    state["speed"] = min(state["speed"] * 2, 16)
                elif e.key == pg.K_DOWN:
                    state["speed"] = max(state["speed"] / 2, 0.25)


def run(pilots: list[Pilot], episodes: int, seed0: int, speed: float, headless: bool = False,
        max_frames: int = 0, screenshot: Path | None = None, classic: bool = False) -> None:
    viewer = Viewer(headless)
    scene = None if classic else SceneRenderer()
    env = make_env("baseline", render_mode="rgb_array")
    env.action_space.seed(seed0)
    state = {"quit": False, "paused": False, "skip": False, "speed": speed}
    total_frames = 0
    for pilot in pilots:
        for ep in range(episodes):
            obs, _ = env.reset(seed=seed0 + ep)
            t, ret, done, state["skip"] = 0, 0.0, False, False
            while not done and not state["quit"] and not state["skip"]:
                viewer.events(state)
                if state["paused"]:
                    viewer.clock.tick(20)
                    continue
                snap = policy_snapshot(pilot.model, obs) if pilot.model else None
                action = int(np.argmax(snap["probs"])) if snap else int(env.action_space.sample())
                prev_obs = obs
                obs, reward, terminated, truncated, _ = env.step(action)
                ret += float(reward)
                t += 1
                total_frames += 1
                done = terminated or truncated
                outcome = classify_outcome(env, terminated, truncated) if done else None
                frame = env.render()  # always called: keeps Gymnasium's particle bookkeeping unchanged
                if scene:
                    frame = scene.render(env, action, t)
                viewer.draw(frame, pilot, ep + 1, t, prev_obs, snap, action, ret, state["speed"], outcome)
                if screenshot and total_frames == max_frames:
                    viewer.pg.image.save(viewer.screen, str(screenshot))
                if max_frames and total_frames >= max_frames:
                    state["quit"] = True
                viewer.clock.tick(int(50 * state["speed"]))
            if done and not state["quit"]:
                print(f"{pilot.label:55s} ep {ep + 1}: {outcome:13s} return {ret:7.1f}  {t} frames")
                for k in range(40 if not headless else 0):  # hold the result ~0.8 s; crashes explode
                    viewer.events(state)
                    if scene and outcome == "crashed":
                        viewer.draw(scene.render(env, 0, t + k, crash_frame=k), pilot, ep + 1, t, prev_obs,
                                    snap, action, ret, state["speed"], outcome)
                    viewer.clock.tick(50)
            if state["quit"]:
                break
        if state["quit"]:
            break
    env.close()
    viewer.pg.quit()


def main(argv: list[str] | None = None) -> None:
    p = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    mode = p.add_mutually_exclusive_group()
    mode.add_argument("--random", action="store_true")
    mode.add_argument("--model", help="run dir containing model.zip")
    mode.add_argument("--progression", help="run dir trained with --checkpoint-every")
    mode.add_argument("--compare", action="store_true", help="(default) random, baseline, hack, fix")
    p.add_argument("--episodes", type=int, default=1, help="episodes per pilot / checkpoint")
    p.add_argument("--stride", type=int, default=1, help="progression: show every k-th checkpoint")
    p.add_argument("--seed", type=int, default=EVAL_SEED0, help="first episode seed (same for every pilot)")
    p.add_argument("--speed", type=float, default=1.0, help="1 = real time (50 frames/s)")
    p.add_argument("--headless", action="store_true", help="no window (tests)")
    p.add_argument("--max-frames", type=int, default=0)
    p.add_argument("--screenshot", type=Path, default=None, help="save the window at --max-frames")
    p.add_argument("--classic", action="store_true", help="Gymnasium's original flat graphics")
    args = p.parse_args(argv)
    run(pilots_from_args(args), args.episodes, args.seed, args.speed, args.headless, args.max_frames,
        args.screenshot, args.classic)


if __name__ == "__main__":
    main()
