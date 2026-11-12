from __future__ import annotations

import math
import random

import pytest

from lora_check.dist import (
    clean,
    cross_validated_effect,
    distance,
    fit_scale,
    max_within_distance,
    mean_cross_distance,
    median,
    percentile,
    top_token,
    truncate,
)


def test_distance_identical_is_zero() -> None:
    p = {"a": -0.1, "b": -2.5, "c": -4.0}
    assert distance(p, dict(p)) == 0.0


def test_distance_is_symmetric_and_in_nats() -> None:
    p = {"a": -0.5, "b": -1.5}
    q = {"a": -0.7, "b": -1.2}
    assert distance(p, q) == pytest.approx(distance(q, p))
    assert distance(p, q) == pytest.approx((0.2 + 0.3) / 2)


def test_distance_missing_token_uses_floor_of_other_side() -> None:
    p = {"a": -0.1, "b": -3.0}
    q = {"a": -0.1, "c": -2.0}
    # b is missing in q -> q's floor (-2.0); c is missing in p -> p's floor (-3.0)
    assert distance(p, q) == pytest.approx((0.0 + 1.0 + 1.0) / 3)


def test_distance_empty_inputs() -> None:
    assert distance({}, {}) == 0.0
    assert math.isinf(distance({"a": -1.0}, {}))


def test_clean_clamps_and_drops_nan() -> None:
    out = clean({"a": float("-inf"), "b": float("nan"), "c": 0.5, "d": -1.0})
    assert out == {"a": -50.0, "c": 0.0, "d": -1.0}


def test_truncate_and_top_token() -> None:
    d = {"a": -3.0, "b": -1.0, "c": -2.0}
    assert truncate(d, 2) == {"b": -1.0, "c": -2.0}
    assert truncate(d, 0) == d
    assert top_token(d) == "b"
    assert top_token({}) == ""


def test_percentile_and_median() -> None:
    values = [1.0, 2.0, 3.0, 4.0, 5.0]
    assert median(values) == 3.0
    assert percentile(values, 0) == 1.0
    assert percentile(values, 100) == 5.0
    assert percentile(values, 90) == pytest.approx(4.6)
    assert percentile([], 50) == 0.0


def test_within_and_cross_distance() -> None:
    a = [{"x": -1.0, "y": -2.0}, {"x": -1.1, "y": -2.0}]
    b = [{"x": -2.0, "y": -1.0}]
    assert max_within_distance(a) == pytest.approx(0.05)
    assert mean_cross_distance(a, b) == pytest.approx((1.0 + 0.95) / 2)


def _triple(
    scale: float, rng: random.Random, noise: float = 0.0
) -> tuple[dict[str, float], dict[str, float], dict[str, float]]:
    tokens = [f"t{i}" for i in range(12)]
    base = {t: rng.uniform(-6, -1) for t in tokens}
    delta = {t: rng.gauss(0, 0.5) for t in tokens}
    shift = rng.uniform(-0.3, 0.3)  # per-prompt normaliser change
    adapter = {t: base[t] + delta[t] + 0.1 for t in tokens}
    observed = {t: base[t] + scale * delta[t] + shift + rng.gauss(0, noise) for t in tokens}
    return base, adapter, observed


@pytest.mark.parametrize("scale", [0.0, 0.25, 0.5, 1.0, 2.0])
def test_fit_scale_recovers_scale_despite_normaliser_shift(scale: float) -> None:
    rng = random.Random(3)
    fit = fit_scale(_triple(scale, rng) for _ in range(10))
    assert fit.usable
    assert fit.scale == pytest.approx(scale, abs=1e-9)
    assert fit.r2 == pytest.approx(1.0)


def test_fit_scale_with_noise_is_close() -> None:
    rng = random.Random(4)
    fit = fit_scale(_triple(0.5, rng, noise=0.05) for _ in range(30))
    assert fit.scale == pytest.approx(0.5, abs=0.05)
    assert fit.r2 > 0.8


def test_fit_scale_degenerate() -> None:
    same = {"a": -1.0, "b": -2.0, "c": -3.0}
    fit = fit_scale([(same, same, same)])
    assert math.isnan(fit.scale)
    assert not fit.usable


def _repeats(
    rng: random.Random, delta: float, noise: float, probes: int = 24, repeats: int = 3
) -> tuple[list[list[dict[str, float]]], list[list[dict[str, float]]]]:
    a_all, b_all = [], []
    for _ in range(probes):
        tokens = [f"t{i}" for i in range(20)]
        base = {t: rng.uniform(-6, -1) for t in tokens}
        shift = {t: rng.gauss(0, delta) for t in tokens}
        a_all.append(
            [{t: base[t] + shift[t] + rng.gauss(0, noise) for t in tokens} for _ in range(repeats)]
        )
        b_all.append([{t: base[t] + rng.gauss(0, noise) for t in tokens} for _ in range(repeats)])
    return a_all, b_all


def test_cross_validated_effect_finds_sub_noise_difference() -> None:
    a, b = _repeats(random.Random(5), delta=0.08, noise=0.1)
    effect = cross_validated_effect(a, b)
    assert effect.significant()
    assert effect.rms == pytest.approx(0.08, rel=0.3)


def test_cross_validated_effect_null_is_not_significant() -> None:
    false_alarms = 0
    for seed in range(40):
        a, b = _repeats(random.Random(100 + seed), delta=0.0, noise=0.1)
        false_alarms += cross_validated_effect(a, b).significant()
    assert false_alarms == 0


def test_cross_validated_effect_deterministic_cases() -> None:
    a, b = _repeats(random.Random(6), delta=0.0, noise=0.0)
    assert not cross_validated_effect(a, b).significant()
    a, b = _repeats(random.Random(6), delta=0.2, noise=0.0)
    assert cross_validated_effect(a, b).significant()
    assert cross_validated_effect([], []).probes == 0
