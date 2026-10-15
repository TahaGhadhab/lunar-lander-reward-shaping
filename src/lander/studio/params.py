"""Every setting the studio exposes: default, range, and why the default is what it is."""

from __future__ import annotations

from typing import Any

from lander.rewards import REWARDS

SHAPED_FEE = ["time_penalty", "upright_slow_fixed"]  # not upright_slow_potential: it is the no-fee ablation
UPRIGHT = ["upright_slow_bonus", "upright_slow_potential", "upright_slow_fixed"]

# kind: choice | float | int | layers | bool. "applies": only shown for these rewards.
PARAMS: list[dict[str, Any]] = [
    # ---- reward
    {
        "key": "reward",
        "group": "Reward",
        "label": "Reward used for training",
        "kind": "choice",
        "choices": sorted(REWARDS),
        "default": "baseline",
        "why": "What PPO is paid for while it learns. Evaluation always uses the plain built-in reward, "
        "so every variant is judged on the same scale.",
    },
    {
        "key": "fee",
        "group": "Reward",
        "label": "Fee per frame",
        "kind": "float",
        "default": 0.05,
        "min": 0.0,
        "max": 1.0,
        "step": 0.01,
        "applies": SHAPED_FEE,
        "why": "Waiting costs something. 0.05 x ~350 frames = ~18, small next to +100 for landing, "
        "and at most 50 over 1,000 frames, so a -100 crash never looks attractive.",
    },
    {
        "key": "bonus",
        "group": "Reward",
        "label": "Bonus per upright frame",
        "kind": "float",
        "default": 1.0,
        "min": 0.0,
        "max": 5.0,
        "step": 0.1,
        "applies": ["upright_slow_bonus"],
        "why": "The deliberate loophole: up to ~1,000 over an episode, ten times the landing bonus, "
        "so hovering until the time limit pays more than landing.",
    },
    {
        "key": "tilt_max",
        "group": "Reward",
        "label": "Upright: max tilt (rad)",
        "kind": "float",
        "default": 0.2,
        "min": 0.02,
        "max": 1.0,
        "step": 0.01,
        "applies": UPRIGHT,
        "why": "90-100% of the baseline's touchdowns are below 0.2 rad, so the agent easily finds the bonus.",
    },
    {
        "key": "speed_max",
        "group": "Reward",
        "label": "Slow: max speed (obs units)",
        "kind": "float",
        "default": 0.2,
        "min": 0.02,
        "max": 1.0,
        "step": 0.01,
        "applies": UPRIGHT,
        "why": "The baseline touches down at about 0.14 in these units.",
    },
    {
        "key": "k",
        "group": "Reward",
        "label": "Potential scale k",
        "kind": "float",
        "default": 10.0,
        "min": 0.0,
        "max": 100.0,
        "step": 1.0,
        "applies": ["upright_slow_potential", "upright_slow_fixed"],
        "why": "Phi = k x (upright x slow score). The shaping over a whole episode is bounded by k, "
        "so 10 guides the agent without outweighing the +100 landing bonus.",
    },
    # ---- PPO
    {
        "key": "learning_rate",
        "group": "PPO",
        "label": "Learning rate",
        "kind": "float",
        "default": 3e-4,
        "min": 1e-6,
        "max": 1e-2,
        "step": 1e-5,
        "why": "Adam step size (SB3 default). Larger learns faster but less stably; try 3e-3 to see it.",
    },
    {
        "key": "n_steps",
        "group": "PPO",
        "label": "Steps per env per update",
        "kind": "int",
        "default": 1024,
        "min": 32,
        "max": 8192,
        "step": 32,
        "why": "1,024 x 16 envs = 16,384 transitions per update: long enough to contain whole landings.",
    },
    {
        "key": "batch_size",
        "group": "PPO",
        "label": "Minibatch size",
        "kind": "int",
        "default": 64,
        "min": 8,
        "max": 8192,
        "step": 8,
        "why": "Small minibatches = many noisy gradient steps (256 per epoch here). Course default.",
    },
    {
        "key": "n_epochs",
        "group": "PPO",
        "label": "Epochs per update",
        "kind": "int",
        "default": 4,
        "min": 1,
        "max": 30,
        "step": 1,
        "why": "How many times each rollout is reused. More reuse, more risk of drifting away from the data.",
    },
    {
        "key": "gamma",
        "group": "PPO",
        "label": "Discount gamma",
        "kind": "float",
        "default": 0.999,
        "min": 0.9,
        "max": 0.9999,
        "step": 0.0005,
        "why": "Horizon ~ 1/(1-gamma) = 1,000 frames, the episode limit. The landing bonus comes late, "
        "so the agent must look far ahead. Also used by the potential-based reward.",
    },
    {
        "key": "gae_lambda",
        "group": "PPO",
        "label": "GAE lambda",
        "kind": "float",
        "default": 0.98,
        "min": 0.8,
        "max": 1.0,
        "step": 0.01,
        "why": "Advantage estimation: closer to 1 = less bias, more variance.",
    },
    {
        "key": "ent_coef",
        "group": "PPO",
        "label": "Entropy bonus",
        "kind": "float",
        "default": 0.01,
        "min": 0.0,
        "max": 0.2,
        "step": 0.005,
        "why": "Keeps the agent exploring. Zero: it may lock into an early habit; too high: never commits.",
    },
    {
        "key": "clip_range",
        "group": "PPO",
        "label": "Clip range",
        "kind": "float",
        "default": 0.2,
        "min": 0.05,
        "max": 0.5,
        "step": 0.05,
        "why": "PPO's 'small steps': caps how much the policy may change in one update (SB3 default).",
    },
    {
        "key": "vf_coef",
        "group": "PPO",
        "label": "Value loss weight",
        "kind": "float",
        "default": 0.5,
        "min": 0.1,
        "max": 2.0,
        "step": 0.1,
        "why": "Weight of the critic's error in the loss (SB3 default).",
    },
    {
        "key": "max_grad_norm",
        "group": "PPO",
        "label": "Gradient clip",
        "kind": "float",
        "default": 0.5,
        "min": 0.1,
        "max": 10.0,
        "step": 0.1,
        "why": "Caps the size of each gradient step, protecting against rare huge errors (SB3 default).",
    },
    # ---- network
    {
        "key": "actor_layers",
        "group": "Network",
        "label": "Actor hidden layers",
        "kind": "layers",
        "default": [64, 64],
        "why": "The policy network: 8 inputs -> these layers -> 4 actions. 2 x 64 (~5k weights) is plenty "
        "for 8 numbers in and 4 choices out. Try 16 or 256, 256 to see under- and over-sizing.",
    },
    {
        "key": "critic_layers",
        "group": "Network",
        "label": "Critic hidden layers",
        "kind": "layers",
        "default": [64, 64],
        "why": "The value network V(s), separate from the actor so its errors do not disturb the policy.",
    },
    {
        "key": "activation",
        "group": "Network",
        "label": "Activation",
        "kind": "choice",
        "choices": ["tanh", "relu", "elu", "leaky_relu"],
        "default": "tanh",
        "why": "tanh keeps every neuron in [-1, 1] (SB3's PPO default, smooth and stable). "
        "ReLU often learns faster but can be less stable here.",
    },
    # ---- training
    {
        "key": "total_timesteps",
        "group": "Training",
        "label": "Training steps",
        "kind": "int",
        "default": 1_000_000,
        "min": 10_000,
        "max": 5_000_000,
        "step": 10_000,
        "why": "PPO starts landing after ~400k steps and lands reliably at 1M (~6-8 min on this laptop). "
        "Use 300k for a quick demo: you will see it learn, not finish.",
    },
    {
        "key": "n_envs",
        "group": "Training",
        "label": "Parallel landers",
        "kind": "int",
        "default": 16,
        "min": 1,
        "max": 32,
        "step": 1,
        "why": "More landers = more varied experience per update.",
    },
    {
        "key": "seed",
        "group": "Training",
        "label": "Seed",
        "kind": "int",
        "default": 0,
        "min": 0,
        "max": 9,
        "step": 1,
        "why": "Another seed is another run. A verdict needs at least 3 seeds.",
    },
    # ---- evaluation ("tests" of the agent)
    {
        "key": "eval_episodes",
        "group": "Evaluation",
        "label": "Test episodes",
        "kind": "int",
        "default": 50,
        "min": 1,
        "max": 500,
        "step": 1,
        "why": "50 fixed episodes separate skill from luck (std of the mean ~3-5 points).",
    },
    {
        "key": "eval_seed",
        "group": "Evaluation",
        "label": "First test seed",
        "kind": "int",
        "default": 10_000,
        "min": 10_000,
        "max": 99_999,
        "step": 1,
        "why": "Fixed and never used in training: every model faces the same landings.",
    },
    {
        "key": "deterministic",
        "group": "Evaluation",
        "label": "Deterministic policy",
        "kind": "bool",
        "default": True,
        "why": "Takes the most likely action: what the agent learned, without exploration noise.",
    },
]

PRESETS: dict[str, dict[str, Any]] = {
    "Project defaults (baseline)": {},
    "Quick demo: 300k steps": {"total_timesteps": 300_000},
    "Reward hacking: +1 per upright frame": {"reward": "upright_slow_bonus"},
    "The fix: potential-based shaping": {"reward": "upright_slow_potential"},
    "Fix + fee per frame": {"reward": "upright_slow_fixed"},
    "Time penalty": {"reward": "time_penalty"},
    "Tiny network: 1 x 16": {"actor_layers": [16], "critic_layers": [16]},
    "Big network: 2 x 256": {"actor_layers": [256, 256], "critic_layers": [256, 256]},
    "ReLU instead of tanh": {"activation": "relu"},
    "Learning rate x10 (unstable)": {"learning_rate": 3e-3},
    "No exploration bonus": {"ent_coef": 0.0},
    "Short-sighted: gamma 0.95": {"gamma": 0.95},
}

SPEC = {p["key"]: p for p in PARAMS}


def defaults() -> dict[str, Any]:
    return {p["key"]: (list(p["default"]) if p["kind"] == "layers" else p["default"]) for p in PARAMS}


def parse_layers(value: Any) -> list[int]:
    message = "layers: 1 to 6 whole numbers between 1 and 1024, separated by commas, e.g. 64, 64"
    if isinstance(value, str):
        value = [v for v in value.replace(";", ",").split(",") if v.strip()]
    try:
        layers = [int(str(v).strip()) for v in value]
    except (TypeError, ValueError):
        raise ValueError(message) from None
    if not 1 <= len(layers) <= 6 or any(not 1 <= n <= 1024 for n in layers):
        raise ValueError(message)
    return layers


def validate(cfg: dict[str, Any]) -> dict[str, Any]:
    """Defaults overlaid with `cfg`; raises ValueError with a readable message on bad input."""
    out = defaults()
    for key, value in cfg.items():
        if key not in SPEC:
            continue
        p = SPEC[key]
        if p["kind"] == "choice":
            if value not in p["choices"]:
                raise ValueError(f"{p['label']}: choose one of {p['choices']}")
            out[key] = value
        elif p["kind"] == "bool":
            out[key] = bool(value)
        elif p["kind"] == "layers":
            out[key] = parse_layers(value)
        else:
            num = float(value) if p["kind"] == "float" else int(float(value))
            if not p["min"] <= num <= p["max"]:
                raise ValueError(f"{p['label']}: must be between {p['min']} and {p['max']}")
            out[key] = num
    if out["batch_size"] > out["n_steps"] * out["n_envs"]:
        raise ValueError("Minibatch size cannot exceed steps per env x parallel landers")
    return out


def reward_kwargs(cfg: dict[str, Any]) -> dict[str, Any]:
    """Constants for the selected reward only (the potential's gamma is tied to PPO's gamma)."""
    reward = cfg["reward"]
    kw = {
        p["key"]: cfg[p["key"]]
        for p in PARAMS
        if p["group"] == "Reward" and p["key"] != "reward" and reward in p.get("applies", [])
    }
    if reward in ("upright_slow_potential", "upright_slow_fixed"):
        kw["gamma"] = cfg["gamma"]
    return kw
