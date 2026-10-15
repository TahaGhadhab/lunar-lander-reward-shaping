import pytest

from lander.evaluate import EpisodeResult, flags, summarize


def ep(ret, length=300, terminated=True, truncated=False, outcome="landed"):
    return EpisodeResult(0, ret, length, terminated, truncated, outcome)


def test_summarize_rates():
    """Evaluation metrics (mean, success, truncation, landed, crashed) are computed correctly."""
    results = [
        ep(250.0),
        ep(210.0),
        ep(-100.0, length=120, outcome="crashed"),
        ep(-50.0, length=1000, terminated=False, truncated=True, outcome="timeout"),
    ]
    s = summarize(results)
    assert s["n_episodes"] == 4
    assert s["mean_return"] == pytest.approx(77.5)
    assert s["success_rate"] == pytest.approx(0.5)
    assert s["truncation_rate"] == pytest.approx(0.25)
    assert s["mean_length"] == pytest.approx(430.0)
    assert s["landed_rate"] == pytest.approx(0.5)
    assert s["crashed_rate"] == pytest.approx(0.25)
    assert s["timeout_rate"] == pytest.approx(0.25)


def test_success_threshold_is_inclusive():
    """A score of exactly 200 counts as a success, as in Gymnasium."""
    assert summarize([ep(200.0)])["success_rate"] == 1.0


def test_flags_detect_stalling():
    """Many episodes cut at 1,000 frames raise the reward-hacking flag."""
    stalling = [ep(50.0, length=1000, terminated=False, truncated=True, outcome="timeout")] * 3
    assert any("truncation" in f for f in flags(summarize(stalling)))
