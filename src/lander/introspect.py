"""Look inside a trained PPO policy: action probabilities, value estimate, hidden activations, checkpoints."""

from __future__ import annotations

import json
import re
from pathlib import Path

import numpy as np
import torch
from stable_baselines3 import PPO

ACTION_NAMES = ["do nothing", "left engine", "main engine", "right engine"]
OBS_NAMES = ["x", "height", "x speed", "y speed", "tilt", "spin", "left leg", "right leg"]


@torch.no_grad()
def policy_snapshot(model: PPO, obs: np.ndarray) -> dict:
    """One forward pass, layer by layer, for the default MlpPolicy (separate 2x64 tanh actor and critic)."""
    x = torch.as_tensor(np.asarray(obs, dtype=np.float32)).unsqueeze(0)
    actor = model.policy.mlp_extractor.policy_net  # Linear(8,64) Tanh Linear(64,64) Tanh
    h1 = actor[1](actor[0](x))
    h2 = actor[3](actor[2](h1))
    logits = model.policy.action_net(h2)
    value = model.policy.value_net(model.policy.mlp_extractor.value_net(x))
    return {
        "h1": h1[0].numpy(),
        "h2": h2[0].numpy(),
        "probs": torch.softmax(logits, dim=-1)[0].numpy(),
        "value": float(value[0, 0]),
    }


def list_checkpoints(run_dir: Path) -> list[tuple[int, Path]]:
    """(timesteps, path) for checkpoints/model_<N>_steps.zip, sorted, then the final model.zip if present."""
    found = []
    for path in (run_dir / "checkpoints").glob("model_*_steps.zip"):
        match = re.fullmatch(r"model_(\d+)_steps\.zip", path.name)
        if match:
            found.append((int(match.group(1)), path))
    found.sort()
    final = run_dir / "model.zip"
    if final.exists():
        steps = found[-1][0] + 1 if found else 0
        config = run_dir / "config.json"
        if config.exists():
            steps = json.loads(config.read_text()).get("timesteps_done", steps)
        if not found or steps > found[-1][0]:
            found.append((int(steps), final))
    return found
