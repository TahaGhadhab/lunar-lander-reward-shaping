import json

import numpy as np
import pytest
import torch
from stable_baselines3 import PPO

from lander.introspect import list_checkpoints, policy_snapshot


@pytest.fixture(scope="module")
def model():
    return PPO("MlpPolicy", "LunarLander-v3", device="cpu", seed=0, n_steps=64, batch_size=64)


def test_snapshot_matches_sb3(model):
    """Layer-by-layer forward pass must give exactly what SB3 itself computes."""
    rng = np.random.default_rng(0)
    for _ in range(20):
        obs = rng.uniform(-1.5, 1.5, size=8).astype(np.float32)
        snap = policy_snapshot(model, obs)
        action, _ = model.predict(obs, deterministic=True)
        assert int(np.argmax(snap["probs"])) == int(action)
        assert snap["probs"].sum() == pytest.approx(1.0, abs=1e-6)
        with torch.no_grad():
            value = model.policy.predict_values(torch.as_tensor(obs).unsqueeze(0))
        assert snap["value"] == pytest.approx(float(value), abs=1e-5)
        assert snap["h1"].shape == (64,) and snap["h2"].shape == (64,)
        assert np.all(np.abs(snap["h1"]) <= 1) and np.all(np.abs(snap["h2"]) <= 1)  # tanh


def test_list_checkpoints_sorted_with_final(tmp_path):
    """Training checkpoints are listed in step order, ending with the final model."""
    (tmp_path / "checkpoints").mkdir()
    for steps in (100_000, 50_000, 1_000_000):
        (tmp_path / "checkpoints" / f"model_{steps}_steps.zip").touch()
    (tmp_path / "checkpoints" / "model_bad.zip").touch()
    (tmp_path / "model.zip").touch()
    (tmp_path / "config.json").write_text(json.dumps({"timesteps_done": 1_015_808}))
    assert [s for s, _ in list_checkpoints(tmp_path)] == [50_000, 100_000, 1_000_000, 1_015_808]


def test_list_checkpoints_empty(tmp_path):
    """A run without checkpoints lists none instead of failing."""
    assert list_checkpoints(tmp_path) == []


def test_snapshot_any_architecture():
    """A custom network (3 ReLU layers of different widths) is read correctly, layer by layer."""
    from lander.introspect import actor_weights

    custom = PPO("MlpPolicy", "LunarLander-v3", device="cpu", seed=1, n_steps=64, batch_size=64,
                 policy_kwargs={"net_arch": {"pi": [32, 48, 16], "vf": [24]}, "activation_fn": torch.nn.ReLU})
    obs = np.linspace(-1, 1, 8).astype(np.float32)
    snap = policy_snapshot(custom, obs)
    assert [len(h) for h in snap["hidden"]] == [32, 48, 16]
    assert all(np.all(h >= 0) for h in snap["hidden"])  # ReLU
    assert int(np.argmax(snap["probs"])) == int(custom.predict(obs, deterministic=True)[0])
    assert [w.shape for w in actor_weights(custom)] == [(32, 8), (48, 32), (16, 48), (4, 16)]
