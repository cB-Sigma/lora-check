"""Statistical behaviour on the simulator (a small version of scripts/evaluate_sim.py)."""

from __future__ import annotations

import pytest

from lora_check.sim.evaluate import SCENARIOS, Cell, evaluate, markdown_table, run_scenario

BY_KEY = {s.key: s for s in SCENARIOS}
TRIALS = 2


def _rate(key: str, noise: float) -> float:
    scenario = BY_KEY[key]
    hits = sum(
        scenario.detected(run_scenario(scenario, noise, seed=500 + t)) for t in range(TRIALS)
    )
    return hits / TRIALS


@pytest.mark.slow
@pytest.mark.parametrize("noise", [0.002, 0.02])
@pytest.mark.parametrize("key", ["healthy", "healthy-fp", "healthy-salt", "healthy-narrow"])
def test_no_false_alarms_at_realistic_noise(key: str, noise: float) -> None:
    assert _rate(key, noise) == 0.0


@pytest.mark.slow
@pytest.mark.parametrize("noise", [0.0, 0.02])
@pytest.mark.parametrize(
    "key",
    [
        "ignored",
        "duplicate",
        "leak",
        "batch-mix",
        "scale",
        "scale-odd",
        "swap",
        "cache",
        "cache-fp",
        "stale",
    ],
)
def test_faults_are_detected_at_realistic_noise(key: str, noise: float) -> None:
    assert _rate(key, noise) == 1.0


@pytest.mark.slow
@pytest.mark.parametrize("key", ["scale-plain", "swap-plain", "cache-plain"])
def test_documented_blind_spots_without_reference(key: str) -> None:
    """These faults are invisible without a fingerprint or cache control - by design."""
    assert _rate(key, 0.002) == 0.0


@pytest.mark.slow
@pytest.mark.parametrize("key", ["healthy", "ignored", "duplicate", "leak", "swap"])
def test_heavy_noise_still_separates_the_obvious_cases(key: str) -> None:
    expected = 0.0 if BY_KEY[key].healthy else 1.0
    assert _rate(key, 0.1) == expected


def test_table_rendering() -> None:
    scenarios = [BY_KEY["healthy"], BY_KEY["ignored"]]
    table = evaluate(trials=1, noise_levels=(0.0,), scenarios=scenarios)
    assert table["healthy"][0.0] == Cell(0, 1)
    assert table["ignored"][0.0].rate == 1.0
    text = markdown_table(table, scenarios)
    assert "SIMULATED" in text
    assert "| adapter ignored -> APPLIED fails | 1/1 |" in text
