"""Look inside a trained PPO policy: action probabilities, value estimate, hidden activations, checkpoints."""

from __future__ import annotations

import json
import re
from pathlib import Path

import numpy as np
import torch

ACTION_NAMES = ["do nothing", "left engine", "main engine", "right engine"]
OBS_NAMES = ["x", "height", "x speed", "y speed", "tilt", "spin", "left leg", "right leg"]


@torch.no_grad()
def policy_snapshot(model, obs: np.ndarray) -> dict:
    """One forward pass, layer by layer, for any SB3 MlpPolicy (any depth, width or activation).

    `model` is a PPO model or its policy. `hidden` lists the actor's activations after each hidden layer;
    `h1` and `h2` alias the first two for the default 2-layer network.
    """
    policy = getattr(model, "policy", model)
    x = torch.as_tensor(np.asarray(obs, dtype=np.float32)).unsqueeze(0)
    hidden, h = [], x
    for module in policy.mlp_extractor.policy_net:  # Linear, activation, Linear, activation, ...
        h = module(h)
        if not isinstance(module, torch.nn.Linear):
            hidden.append(h[0].numpy())
    logits = policy.action_net(h)
    value = policy.value_net(policy.mlp_extractor.value_net(x))
    snap = {"hidden": hidden, "probs": torch.softmax(logits, dim=-1)[0].numpy(), "value": float(value[0, 0])}
    for i, layer in enumerate(hidden[:2], start=1):
        snap[f"h{i}"] = layer
    return snap


def actor_weights(model) -> list[np.ndarray]:
    """Weight matrices of the actor, input to output: hidden Linear layers then the action head."""
    policy = getattr(model, "policy", model)
    linears = [m for m in policy.mlp_extractor.policy_net if isinstance(m, torch.nn.Linear)]
    return [m.weight.detach().numpy().copy() for m in linears + [policy.action_net]]


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
