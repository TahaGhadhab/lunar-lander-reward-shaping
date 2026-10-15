import pytest
from stable_baselines3 import PPO

from lander.studio.engine import net_layout, parse_failures, safe_model_path
from lander.studio.params import PRESETS, defaults, reward_kwargs, validate


def test_defaults_match_training_script():
    """The studio's defaults are the project's PPO settings, so Reset really restores them."""
    from lander.train import PPO_HYPERPARAMS

    d = defaults()
    assert {k: d[k] for k in PPO_HYPERPARAMS} == PPO_HYPERPARAMS
    assert d["actor_layers"] == [64, 64] and d["activation"] == "tanh" and d["total_timesteps"] == 1_000_000


def test_every_preset_is_valid():
    """Every preset in the studio passes validation."""
    for name, overrides in PRESETS.items():
        assert validate(overrides), name


@pytest.mark.parametrize("bad", [{"gamma": 1.5}, {"reward": "nope"}, {"actor_layers": "64, x"},
                                 {"batch_size": 8192, "n_steps": 32, "n_envs": 1}])
def test_validate_refuses_bad_settings(bad):
    """Out-of-range or malformed settings are refused with a readable message."""
    with pytest.raises(ValueError):
        validate(bad)


def test_reward_constants_only_for_the_chosen_reward():
    """The no-fee ablation never receives the fee; the potential's gamma is tied to PPO's gamma."""
    cfg = validate({"reward": "upright_slow_potential", "gamma": 0.99})
    assert reward_kwargs(cfg) == {"tilt_max": 0.2, "speed_max": 0.2, "k": 10.0, "gamma": 0.99}
    assert reward_kwargs(validate({"reward": "baseline"})) == {}
    assert reward_kwargs(validate({"reward": "upright_slow_fixed"}))["fee"] == 0.05


def test_net_layout_samples_wide_layers():
    """Wide layers are drawn as an even sample of 24 neurons, with the matching weights."""
    model = PPO("MlpPolicy", "LunarLander-v3", device="cpu", n_steps=64, batch_size=64,
                policy_kwargs={"net_arch": {"pi": [256, 16], "vf": [8]}})
    lay = net_layout(model.policy)
    assert lay["sizes"] == [8, 256, 16, 4]
    assert [len(i) for i in lay["idx"]] == [8, 24, 16, 4]
    assert [(len(w), len(w[0])) for w in lay["w"]] == [(24, 8), (16, 24), (4, 16)]


def test_parse_failures_maps_traceback_to_test():
    """A failing test's traceback is attached to that test, and nothing else."""
    bad = "tests/test_a.py::test_bad"
    lines = ["tests/test_a.py::test_ok PASSED", f"{bad} FAILED",
             "=================================== FAILURES ===================================",
             "___________________________________ test_bad ___________________________________",
             "    assert 1 == 2", "E   assert 1 == 2",
             "=========================== short test summary info ============================",
             f"FAILED {bad} - assert 1 == 2"]
    out = parse_failures(lines, ["tests/test_a.py::test_ok", bad])
    assert list(out) == [bad]
    assert "E   assert 1 == 2" in out[bad] and "short test summary" not in out[bad]


def test_model_paths_cannot_escape_runs(tmp_path):
    """The studio only loads .zip models from inside runs/."""
    (tmp_path / "runs").mkdir()
    (tmp_path / "secret.zip").write_bytes(b"x")
    with pytest.raises(ValueError):
        safe_model_path("runs/../secret.zip", tmp_path)
    with pytest.raises(ValueError):
        safe_model_path("secret.zip", tmp_path)


def test_server_refuses_other_websites():
    """Other websites (wrong Host, foreign Origin, non-JSON POST) are refused; the page itself works."""
    import http.client
    import threading
    from http.server import ThreadingHTTPServer

    from lander.studio.server import make_handler

    class FakeStudio:
        stopped = False

        def stop_training(self):
            FakeStudio.stopped = True

    srv = ThreadingHTTPServer(("127.0.0.1", 0), None)
    port = srv.server_address[1]
    srv.RequestHandlerClass = make_handler(FakeStudio(), port)
    threading.Thread(target=srv.serve_forever, daemon=True).start()

    def post(headers):
        conn = http.client.HTTPConnection("127.0.0.1", port, timeout=5)
        conn.request("POST", "/api/train/stop", body="{}", headers=headers)
        return conn.getresponse().status

    host, js = {"Host": f"127.0.0.1:{port}"}, {"Content-Type": "application/json"}
    try:
        assert post({"Host": f"evil.example:{port}", **js}) == 403  # DNS rebinding
        assert post({**host, **js, "Origin": "https://evil.example"}) == 403  # cross-site
        assert post({**host, "Content-Type": "text/plain"}) == 415  # simple CORS request
        assert not FakeStudio.stopped
        assert post({**host, **js, "Origin": f"http://127.0.0.1:{port}"}) == 200
        assert FakeStudio.stopped
    finally:
        srv.shutdown()
