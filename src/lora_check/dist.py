"""Truncated next-token distributions and the statistics built on them.

An endpoint only reveals the top-K next-token log-probabilities, so every
comparison in lora-check works on *truncated* distributions: a mapping from a
token key to its log-probability.  This module contains the distance used by
all checks, the percentile helper used for noise estimates, and the
least-squares fit that estimates the effective adapter scale.
"""

from __future__ import annotations

import math
from collections.abc import Iterable, Mapping, Sequence
from dataclasses import dataclass

Dist = dict[str, float]
"""Token key -> log-probability (natural log) for the top-K next tokens."""

LOGPROB_FLOOR = -50.0
"""Log-probabilities below this (including ``-inf``) are clamped to it."""


def clean(raw: Mapping[str, float]) -> Dist:
    """Return a copy with non-finite or absurdly small values clamped."""
    out: Dist = {}
    for token, value in raw.items():
        lp = float(value)
        if math.isnan(lp):
            continue
        out[token] = max(LOGPROB_FLOOR, min(lp, 0.0))
    return out


def truncate(dist: Mapping[str, float], k: int) -> Dist:
    """Keep the ``k`` most likely tokens (ties broken by token key)."""
    if k <= 0 or len(dist) <= k:
        return dict(dist)
    items = sorted(dist.items(), key=lambda kv: (-kv[1], kv[0]))
    return dict(items[:k])


def top_token(dist: Mapping[str, float]) -> str:
    """Most likely token key (ties broken by key); empty string if empty."""
    if not dist:
        return ""
    return min(dist.items(), key=lambda kv: (-kv[1], kv[0]))[0]


def distance(p: Mapping[str, float], q: Mapping[str, float]) -> float:
    """Mean absolute log-probability difference over the union of tokens.

    A token missing from one side is known to be *at most* as likely as that
    side's least likely listed token, so that minimum is substituted.  The
    substitution can only shrink the true difference, which makes the distance
    conservative: it never invents a difference that the full distributions
    do not have.

    The unit is nats, the same unit the upstream bug reports use when they
    quote "mean |delta logprob|".
    """
    if not p or not q:
        return 0.0 if not p and not q else float("inf")
    p_floor = min(p.values())
    q_floor = min(q.values())
    total = 0.0
    count = 0
    for token, lp in p.items():
        total += abs(lp - q.get(token, q_floor))
        count += 1
    for token, lq in q.items():
        if token not in p:
            total += abs(p_floor - lq)
            count += 1
    return total / count


def mean(values: Iterable[float]) -> float:
    """Arithmetic mean; 0.0 for an empty input."""
    items = list(values)
    return sum(items) / len(items) if items else 0.0


def percentile(values: Sequence[float], q: float) -> float:
    """Linear-interpolated percentile, ``q`` in [0, 100]; 0.0 if empty."""
    if not values:
        return 0.0
    ordered = sorted(values)
    if len(ordered) == 1:
        return ordered[0]
    pos = (len(ordered) - 1) * (q / 100.0)
    lo = math.floor(pos)
    hi = min(lo + 1, len(ordered) - 1)
    frac = pos - lo
    return ordered[lo] * (1.0 - frac) + ordered[hi] * frac


def median(values: Sequence[float]) -> float:
    """Median; 0.0 if empty."""
    return percentile(values, 50.0)


def mean_cross_distance(
    a: Sequence[Mapping[str, float]], b: Sequence[Mapping[str, float]]
) -> float:
    """Mean distance over all pairs drawn from two groups of repeats."""
    return mean(distance(x, y) for x in a for y in b)


def max_within_distance(group: Sequence[Mapping[str, float]]) -> float:
    """Largest pairwise distance among repeats of the same request."""
    worst = 0.0
    for i in range(len(group)):
        for j in range(i + 1, len(group)):
            worst = max(worst, distance(group[i], group[j]))
    return worst


@dataclass(frozen=True)
class Effect:
    """Cross-validated estimate of how far apart two models are."""

    rms: float
    """Estimated root-mean-square log-prob difference, with noise removed."""
    t: float
    """t-statistic over probes for "the difference is real"."""
    probes: int

    def significant(self, t_critical: float = 4.0, floor: float = 1e-3) -> bool:
        """True when the difference is both statistically clear and non-trivial."""
        return self.probes >= 4 and self.t >= t_critical and self.rms >= floor


def cross_validated_effect(
    a: Sequence[Sequence[Mapping[str, float]]], b: Sequence[Sequence[Mapping[str, float]]]
) -> Effect:
    """Detect a systematic difference that is smaller than the per-request noise.

    ``a[i]`` and ``b[i]`` are the repeated answers of two models to probe
    ``i``.  For repeat ``r`` let ``d_r = a_r - b_r`` over the tokens every
    answer lists.  If the models are the same, the ``d_r`` are independent
    noise and the inner product of two *different* repeats averages to zero;
    if they differ by ``delta`` it averages to ``|delta|^2``.  Averaging those
    cross products therefore estimates the squared effect without the noise
    bias that a plain distance carries, and a t-test over probes says whether
    it is distinguishable from zero.
    """
    per_probe: list[float] = []
    for group_a, group_b in zip(a, b):
        repeats = min(len(group_a), len(group_b))
        if repeats < 2:
            continue
        shared = set(group_a[0])
        for dist in [*group_a[:repeats], *group_b[:repeats]]:
            shared &= set(dist)
        if len(shared) < 2:
            continue
        tokens = sorted(shared)
        total = [0.0] * len(tokens)
        squares = 0.0
        for r in range(repeats):
            for j, token in enumerate(tokens):
                d = group_a[r][token] - group_b[r][token]
                total[j] += d
                squares += d * d
        cross = sum(x * x for x in total) - squares
        per_probe.append(cross / (repeats * (repeats - 1) * len(tokens)))
    n = len(per_probe)
    if n == 0:
        return Effect(rms=0.0, t=0.0, probes=0)
    avg = sum(per_probe) / n
    rms = math.sqrt(avg) if avg > 0 else 0.0
    if n < 2:
        return Effect(rms=rms, t=0.0, probes=n)
    var = sum((x - avg) ** 2 for x in per_probe) / (n - 1)
    if var <= 0.0:
        return Effect(rms=rms, t=float("inf") if avg > 0 else 0.0, probes=n)
    return Effect(rms=rms, t=avg / math.sqrt(var / n), probes=n)


@dataclass(frozen=True)
class ScaleFit:
    """Result of fitting ``observed ~ base + scale * (adapter - base)``."""

    scale: float
    r2: float
    points: int

    @property
    def usable(self) -> bool:
        """True when the fit rests on enough points to mean anything."""
        return self.points >= 8 and not math.isnan(self.scale)


def fit_scale(
    triples: Iterable[tuple[Mapping[str, float], Mapping[str, float], Mapping[str, float]]],
) -> ScaleFit:
    """Estimate the effective adapter scale from (base, adapter, observed) triples.

    To first order a LoRA applied at scale ``s`` moves the logits by
    ``s * delta``.  Log-probabilities are logits minus a per-prompt
    normaliser, so within one prompt we centre both ``adapter - base`` and
    ``observed - base`` over the tokens all three distributions share; the
    normaliser cancels and the slope of a no-intercept regression is the
    scale.  ``r2`` says how much of the observed movement the single-scale
    model explains; a low value means "not simply a rescaled adapter".
    """
    sxy = 0.0
    sxx = 0.0
    syy = 0.0
    points = 0
    for base, adapter, observed in triples:
        shared = [t for t in base if t in adapter and t in observed]
        if len(shared) < 3:
            continue
        xs = [adapter[t] - base[t] for t in shared]
        ys = [observed[t] - base[t] for t in shared]
        mx = sum(xs) / len(xs)
        my = sum(ys) / len(ys)
        for x, y in zip(xs, ys):
            sxy += (x - mx) * (y - my)
            sxx += (x - mx) ** 2
            syy += (y - my) ** 2
        points += len(shared)
    if sxx <= 1e-18:
        return ScaleFit(scale=float("nan"), r2=0.0, points=points)
    scale = sxy / sxx
    if syy <= 1e-18:
        # Observed equals base: a perfect fit of scale zero.
        return ScaleFit(scale=0.0, r2=1.0, points=points)
    r2 = (sxy * sxy) / (sxx * syy)
    return ScaleFit(scale=scale, r2=max(0.0, min(1.0, r2)), points=points)
