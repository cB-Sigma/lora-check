"""Black-box verification of a live multi-LoRA endpoint.

The run has four phases, all using greedy one-token requests with log-probs:

1. **Isolated** - every model (base and each adapter) answers every probe
   ``repeats`` times, one model at a time.  Repeats of identical requests
   give the run-to-run noise that every later threshold is scaled by.
2. **Same-model burst** - each model answers all probes concurrently, alone.
   This measures how much answers move merely because requests are batched.
3. **Interleaved** - a strictly sequential pass that switches model on every
   request, then a concurrent burst that mixes all models on the same
   prompts.  Answers must stay where phase 1 put them.
4. **Cache bypass** (optional) - each request repeated with a unique
   ``cache_salt``; answers must not depend on the prefix cache.

From these the checks APPLIED, DISTINCT, STABLE, CACHE and (with a
fingerprint) REFERENCE are derived.
"""

from __future__ import annotations

import uuid
from collections.abc import Callable, Sequence
from concurrent.futures import ThreadPoolExecutor
from dataclasses import dataclass, field
from typing import Any

from lora_check.client import Client, EndpointError, ModelInfo
from lora_check.dist import (
    Dist,
    Effect,
    cross_validated_effect,
    distance,
    max_within_distance,
    mean,
    mean_cross_distance,
    median,
    percentile,
)
from lora_check.fingerprint import Fingerprint, compare_to_reference
from lora_check.probes import Probe

PASS = "PASS"
WARN = "WARN"
FAIL = "FAIL"
SKIP = "SKIP"
ERROR = "ERROR"

_RANK = {SKIP: 0, PASS: 1, WARN: 2, FAIL: 3, ERROR: 4}

CHECKS = ("applied", "distinct", "stable", "cache", "reference")


@dataclass
class VerifyConfig:
    """Knobs of a verification run."""

    probes: list[Probe]
    adapters: list[str] | None = None
    """Adapter ids to check; ``None`` discovers them from ``/v1/models``."""
    base_model: str | None = None
    adapter_model_format: str = "{adapter}"
    """How an adapter is addressed in the ``model`` field.  vLLM uses the
    adapter name; SGLang documents ``{base}:{adapter}``."""
    repeats: int = 3
    concurrency: int = 8
    noise_multiplier: float = 3.0
    min_delta: float = 1e-3
    """Smallest distance (nats) ever treated as a real difference."""
    t_critical: float = 4.0
    """t-statistic a sub-noise difference must reach to count as real."""
    weak_fraction: float = 0.25
    """Below this share of differing probes APPLIED is only a WARN."""
    interleave: bool = True
    cache_salt: bool = False
    fingerprint: Fingerprint | None = None
    name_map: dict[str, str] = field(default_factory=dict)
    """Served adapter id -> adapter name inside the fingerprint."""
    reference_tolerance: float = 0.02
    scale_tolerance: float = 0.15
    progress: Callable[[str], None] | None = None


@dataclass
class CheckResult:
    """Outcome of one check for one model."""

    name: str
    status: str
    summary: str
    evidence: dict[str, Any] = field(default_factory=dict)

    def to_json(self) -> dict[str, Any]:
        return {"status": self.status, "summary": self.summary, "evidence": self.evidence}


@dataclass
class ModelReport:
    """All checks for one model (an adapter, or the base)."""

    name: str
    model_id: str
    is_base: bool
    noise: float = 0.0
    checks: dict[str, CheckResult] = field(default_factory=dict)

    @property
    def verdict(self) -> str:
        worst = SKIP
        for check in self.checks.values():
            if _RANK[check.status] > _RANK[worst]:
                worst = check.status
        return PASS if worst == SKIP else worst

    def to_json(self) -> dict[str, Any]:
        return {
            "name": self.name,
            "model_id": self.model_id,
            "verdict": self.verdict,
            "noise": round(self.noise, 6),
            "checks": {name: check.to_json() for name, check in self.checks.items()},
        }


@dataclass
class VerifyReport:
    """Result of :func:`run_verify`."""

    endpoint: str
    base: ModelReport
    adapters: list[ModelReport]
    probes: int
    repeats: int
    requests: int
    api: str
    notes: list[str] = field(default_factory=list)

    @property
    def verdict(self) -> str:
        worst = PASS
        for report in [self.base, *self.adapters]:
            if _RANK[report.verdict] > _RANK[worst]:
                worst = report.verdict
        return worst

    def exit_code(self, strict: bool = False) -> int:
        """0 healthy, 1 a check failed, 2 the run could not be completed."""
        verdict = self.verdict
        if verdict == ERROR:
            return 2
        if verdict == FAIL or (strict and verdict == WARN):
            return 1
        return 0

    def to_json(self) -> dict[str, Any]:
        return {
            "tool": "lora-check",
            "command": "verify",
            "endpoint": self.endpoint,
            "verdict": self.verdict,
            "probes": self.probes,
            "repeats": self.repeats,
            "requests": self.requests,
            "api": self.api,
            "base": self.base.to_json(),
            "adapters": [a.to_json() for a in self.adapters],
            "notes": self.notes,
        }


# ---------------------------------------------------------------------------
# Discovery
# ---------------------------------------------------------------------------


def discover(
    models: Sequence[ModelInfo], base_model: str | None, adapters: Sequence[str] | None
) -> tuple[str, list[str]]:
    """Pick the base model id and the adapter ids to verify.

    Adapters are the ``/v1/models`` entries with a ``parent``.  If the server
    does not report parents, name the adapters with ``--adapter`` and the
    base with ``--base-model``.
    """
    listed_adapters = [m for m in models if m.is_adapter]
    if adapters:
        chosen = list(dict.fromkeys(adapters))
    else:
        chosen = [m.id for m in listed_adapters if base_model is None or m.parent == base_model]
        if not chosen:
            chosen = [m.id for m in listed_adapters]
    if not chosen:
        raise EndpointError(
            "no LoRA adapters found: /v1/models lists no entry with a 'parent' field. "
            "Name them with --adapter (and the base with --base-model)."
        )
    if base_model is None:
        parents = {m.parent for m in listed_adapters if m.id in chosen and m.parent}
        if len(parents) == 1:
            base_model = parents.pop()
        else:
            bases = [m.id for m in models if not m.is_adapter and m.id not in chosen]
            if len(bases) != 1:
                raise EndpointError(
                    "cannot tell which model is the base "
                    f"(candidates: {', '.join(bases) or 'none'}); pass --base-model"
                )
            base_model = bases[0]
    if base_model in chosen:
        raise EndpointError(f"{base_model!r} is both the base model and an adapter to check")
    return base_model, chosen


# ---------------------------------------------------------------------------
# The run
# ---------------------------------------------------------------------------


@dataclass
class _Samples:
    """Everything observed for one model."""

    model_id: str
    isolated: dict[str, list[Dist]] = field(default_factory=dict)
    burst: dict[str, Dist] = field(default_factory=dict)
    sequential: dict[str, tuple[Dist, str]] = field(default_factory=dict)  # (answer, predecessor)
    mixed: dict[str, Dist] = field(default_factory=dict)
    salted: dict[str, Dist] = field(default_factory=dict)
    error: str | None = None
    salt_error: str | None = None
    within: dict[str, float] = field(default_factory=dict)
    noise: float = 0.0
    burst_noise: float = 0.0


@dataclass(frozen=True)
class _Comparison:
    """How two models' isolated answers relate."""

    differ: int
    median: float
    top: float
    threshold: float
    effect: Effect
    weak: bool

    def evidence(self, probes: int) -> dict[str, Any]:
        return {
            "probes_differing": self.differ,
            "probes": probes,
            "median_distance": round(self.median, 6),
            "max_distance": round(self.top, 6),
            "threshold": round(self.threshold, 6),
            "cross_validated_rms": round(self.effect.rms, 6),
            "cross_validated_t": round(min(self.effect.t, 1e6), 2),
        }


def _schedule(names: list[str], probes: list[Probe]) -> list[tuple[str, Probe]]:
    """Sequential order that changes model on every request.

    Rotating the model order per probe makes every model follow every other
    model across the run.
    """
    order: list[tuple[str, Probe]] = []
    last: str | None = None
    for i, probe in enumerate(probes):
        shift = i % len(names)
        rotated = names[shift:] + names[:shift]
        if (i // len(names)) % 2 == 1:
            rotated.reverse()
        if last is not None and rotated[0] == last and len(rotated) > 1:
            rotated[0], rotated[1] = rotated[1], rotated[0]
        for name in rotated:
            order.append((name, probe))
            last = name
    return order


class _Run:
    def __init__(
        self, client: Client, config: VerifyConfig, base: str, adapters: list[str]
    ) -> None:
        self.client = client
        self.cfg = config
        self.base = base
        self.adapter_names = adapters
        self.samples: dict[str, _Samples] = {base: _Samples(model_id=base)}
        for name in adapters:
            model_id = config.adapter_model_format.format(base=base, adapter=name)
            self.samples[name] = _Samples(model_id=model_id)
        self.notes: list[str] = []

    # -- helpers -----------------------------------------------------------

    def _say(self, message: str) -> None:
        if self.cfg.progress is not None:
            self.cfg.progress(message)

    def _alive(self) -> list[str]:
        return [name for name, s in self.samples.items() if s.error is None]

    def _ask(self, name: str, probe: Probe, salt: str | None = None) -> Dist:
        return self.client.next_token(self.samples[name].model_id, probe.prompt, cache_salt=salt)

    def _asker(self, name: str) -> Callable[[Probe], Dist]:
        def ask(probe: Probe) -> Dist:
            return self._ask(name, probe)

        return ask

    def _threshold(self, name: str, probe_id: str) -> float:
        s = self.samples[name]
        return max(
            self.cfg.min_delta,
            self.cfg.noise_multiplier * max(s.noise, s.within.get(probe_id, 0.0)),
        )

    def _deviation(self, name: str, probe_id: str, answer: Dist) -> float:
        return mean(distance(answer, ref) for ref in self.samples[name].isolated[probe_id])

    # -- phases ------------------------------------------------------------

    def collect(self) -> None:
        probes = self.cfg.probes
        self._say(
            f"phase 1/4: isolated answers ({len(self.samples)} models x {len(probes)} probes x {self.cfg.repeats})"
        )
        for name, s in self.samples.items():
            try:
                self._ask(name, probes[0])  # warm-up, discarded
                for _ in range(self.cfg.repeats):
                    for probe in probes:
                        s.isolated.setdefault(probe.id, []).append(self._ask(name, probe))
            except EndpointError as exc:
                s.error = str(exc)
                continue
            s.within = {pid: max_within_distance(group) for pid, group in s.isolated.items()}
            s.noise = percentile(list(s.within.values()), 90.0)
        if self.samples[self.base].error is not None:
            raise EndpointError(f"base model {self.base!r} failed: {self.samples[self.base].error}")

        if not self.cfg.interleave:
            return
        workers = max(1, self.cfg.concurrency)
        self._say("phase 2/4: same-model concurrent bursts")
        with ThreadPoolExecutor(max_workers=workers) as pool:
            for name in self._alive():
                s = self.samples[name]
                try:
                    self._ask(name, probes[0])  # switch models before the burst
                    answers = list(pool.map(self._asker(name), probes))
                except EndpointError as exc:
                    s.error = str(exc)
                    continue
                s.burst = {p.id: a for p, a in zip(probes, answers)}
                s.burst_noise = percentile(
                    [self._deviation(name, pid, a) for pid, a in s.burst.items()], 90.0
                )

            alive = self._alive()
            self._say("phase 3/4: interleaved sequential pass and mixed concurrent burst")
            previous = alive[-1]
            try:
                for name, probe in _schedule(alive, probes):
                    self.samples[name].sequential[probe.id] = (self._ask(name, probe), previous)
                    previous = name
                tasks = [(name, probe) for probe in probes for name in alive]
                answers = list(pool.map(lambda task: self._ask(task[0], task[1]), tasks))
                for (name, probe), answer in zip(tasks, answers):
                    self.samples[name].mixed[probe.id] = answer
            except EndpointError as exc:
                self.notes.append(f"interleaving phase aborted: {exc}")
                for name in alive:
                    self.samples[name].sequential.clear()
                    self.samples[name].mixed.clear()

            if self.cfg.cache_salt:
                self._say("phase 4/4: cache-bypass control (cache_salt)")
                for name in self._alive():
                    s = self.samples[name]
                    try:
                        for probe in probes:
                            s.salted[probe.id] = self._ask(name, probe, salt=uuid.uuid4().hex)
                    except EndpointError as exc:
                        s.salted.clear()
                        s.salt_error = str(exc)

    # -- checks ------------------------------------------------------------

    def _compare(self, a: str, b: str) -> _Comparison:
        sa, sb = self.samples[a], self.samples[b]
        threshold = max(self.cfg.min_delta, self.cfg.noise_multiplier * max(sa.noise, sb.noise))
        probes = self.cfg.probes
        cross = [mean_cross_distance(sa.isolated[p.id], sb.isolated[p.id]) for p in probes]
        effect = cross_validated_effect(
            [sa.isolated[p.id] for p in probes], [sb.isolated[p.id] for p in probes]
        )
        return _Comparison(
            differ=sum(1 for d in cross if d > threshold),
            median=median(cross),
            top=max(cross),
            threshold=threshold,
            effect=effect,
            weak=effect.significant(self.cfg.t_critical, self.cfg.min_delta),
        )

    def check_applied(self, name: str) -> CheckResult:
        n = len(self.cfg.probes)
        cmp = self._compare(name, self.base)
        evidence = cmp.evidence(n)
        if cmp.differ == 0 and cmp.weak:
            return CheckResult(
                "applied",
                WARN,
                f"no single probe differs from base by more than the noise threshold "
                f"({cmp.threshold:.4f}), but across repeats there is a consistent difference of about "
                f"{cmp.effect.rms:.4f} nats (t={cmp.effect.t:.1f}): applied, with an effect below the "
                "endpoint's run-to-run noise. Use more sensitive probes (`lora-check mine`)",
                evidence,
            )
        if cmp.differ == 0:
            return CheckResult(
                "applied",
                FAIL,
                f"indistinguishable from the base model on all {n} probes "
                f"(max distance {cmp.top:.4f} <= threshold {cmp.threshold:.4f}, no consistent "
                f"difference across repeats): the adapter has no effect",
                evidence,
            )
        if cmp.differ < self.cfg.weak_fraction * n:
            return CheckResult(
                "applied",
                WARN,
                f"differs from base on only {cmp.differ}/{n} probes (median distance {cmp.median:.4f}, "
                f"threshold {cmp.threshold:.4f}): a narrow adapter, or one that is only partly applied",
                evidence,
            )
        return CheckResult(
            "applied",
            PASS,
            f"differs from base on {cmp.differ}/{n} probes "
            f"(median distance {cmp.median:.4f}, threshold {cmp.threshold:.4f})",
            evidence,
        )

    def check_distinct(self, name: str) -> CheckResult:
        others = [o for o in self.adapter_names if o != name and self.samples[o].error is None]
        if not others:
            return CheckResult("distinct", SKIP, "only one adapter to compare")
        twins: list[str] = []
        closest: tuple[float, str] | None = None
        for other in others:
            cmp = self._compare(name, other)
            if cmp.differ == 0 and not cmp.weak:
                twins.append(other)
            if closest is None or cmp.median < closest[0]:
                closest = (cmp.median, other)
        assert closest is not None
        evidence = {
            "twins": twins,
            "closest": closest[1],
            "closest_median_distance": round(closest[0], 6),
        }
        if twins:
            return CheckResult(
                "distinct",
                FAIL,
                f"behaves identically to {', '.join(repr(t) for t in twins)} on every probe: "
                "the same weights are served under several names (mix-up, or all of them ignored)",
                evidence,
            )
        return CheckResult(
            "distinct",
            PASS,
            f"differs from all {len(others)} other adapter(s) (closest: {closest[1]!r}, "
            f"median distance {closest[0]:.4f})",
            evidence,
        )

    def _closest_other(self, name: str, probe_id: str, answer: Dist) -> str | None:
        """Another model whose own isolated answer this answer matches."""
        best: tuple[float, str] | None = None
        for other in self._alive():
            if other == name:
                continue
            d = mean(distance(answer, ref) for ref in self.samples[other].isolated[probe_id])
            if d <= self._threshold(other, probe_id) and (best is None or d < best[0]):
                best = (d, other)
        return best[1] if best else None

    def check_stable(self, name: str) -> CheckResult:
        s = self.samples[name]
        if not self.cfg.interleave:
            return CheckResult("stable", SKIP, "interleaving disabled")
        if not s.sequential and not s.mixed:
            return CheckResult("stable", SKIP, "interleaving phase did not complete")
        flagged: list[dict[str, Any]] = []
        worst = 0.0
        for mode, answers in (
            ("sequential", {k: v[0] for k, v in s.sequential.items()}),
            ("concurrent", s.mixed),
        ):
            for pid, answer in answers.items():
                threshold = self._threshold(name, pid)
                if mode == "concurrent":
                    threshold = max(threshold, self.cfg.noise_multiplier * s.burst_noise)
                deviation = self._deviation(name, pid, answer)
                worst = max(worst, deviation / threshold)
                if deviation > threshold:
                    item: dict[str, Any] = {
                        "probe": pid,
                        "mode": mode,
                        "deviation": round(deviation, 6),
                        "threshold": round(threshold, 6),
                        "matches": self._closest_other(name, pid, answer),
                    }
                    if mode == "sequential":
                        item["after"] = s.sequential[pid][1]
                    flagged.append(item)
        total = len(s.sequential) + len(s.mixed)
        evidence: dict[str, Any] = {
            "answers_checked": total,
            "answers_moved": len(flagged),
            "worst_deviation_over_threshold": round(worst, 3),
            "examples": flagged[:5],
        }
        if not flagged:
            return CheckResult(
                "stable",
                PASS,
                f"{total} interleaved answers all match the isolated ones "
                f"(worst deviation {worst:.2f}x threshold)",
                evidence,
            )
        seq = sum(1 for f in flagged if f["mode"] == "sequential")
        conc = len(flagged) - seq
        matched: dict[str, int] = {}
        for f in flagged:
            if f["matches"]:
                matched[f["matches"]] = matched.get(f["matches"], 0) + 1
        evidence["matches_other_model"] = matched
        where = []
        if seq:
            where.append(f"{seq} after switching model")
        if conc:
            where.append(f"{conc} under mixed concurrency")
        detail = ""
        if matched:
            top_name = max(matched, key=lambda k: matched[k])
            detail = (
                f"; {sum(matched.values())} of them equal another model's answer "
                f"(most often {top_name!r}): cross-adapter leakage"
            )
        if len(flagged) == 1 and not matched:
            return CheckResult(
                "stable",
                WARN,
                f"1/{total} interleaved answers moved ({'; '.join(where)}) - possibly a noise outlier",
                evidence,
            )
        return CheckResult(
            "stable",
            FAIL,
            f"{len(flagged)}/{total} answers changed depending on neighbouring requests "
            f"({', '.join(where)}){detail}",
            evidence,
        )

    def check_cache(self, name: str) -> CheckResult:
        s = self.samples[name]
        if not self.cfg.cache_salt or not self.cfg.interleave:
            return CheckResult("cache", SKIP, "cache-bypass control not requested (--cache-salt)")
        if s.salt_error is not None:
            return CheckResult(
                "cache", SKIP, f"server rejected the cache_salt parameter: {s.salt_error[:120]}"
            )
        if not s.salted:
            return CheckResult("cache", SKIP, "cache-bypass phase did not run")
        moved = []
        for pid, answer in s.salted.items():
            threshold = self._threshold(name, pid)
            deviation = self._deviation(name, pid, answer)
            if deviation > threshold:
                moved.append(
                    {
                        "probe": pid,
                        "deviation": round(deviation, 6),
                        "threshold": round(threshold, 6),
                    }
                )
        evidence = {
            "answers_checked": len(s.salted),
            "answers_moved": len(moved),
            "examples": moved[:5],
        }
        if not moved:
            return CheckResult(
                "cache",
                PASS,
                f"{len(s.salted)} answers unchanged when the prefix cache is bypassed",
                evidence,
            )
        if len(moved) == 1:
            return CheckResult(
                "cache",
                WARN,
                f"1/{len(s.salted)} answers changed with a fresh cache_salt",
                evidence,
            )
        return CheckResult(
            "cache",
            FAIL,
            f"{len(moved)}/{len(s.salted)} answers change when the prefix cache is bypassed with a fresh "
            "cache_salt: cached state computed for another adapter or older weights is being reused",
            evidence,
        )

    def check_reference(self, name: str) -> CheckResult:
        fingerprint = self.cfg.fingerprint
        if fingerprint is None:
            return CheckResult("reference", SKIP, "no fingerprint supplied (--fingerprint)")
        ref_name = self.cfg.name_map.get(name, name)
        if ref_name not in fingerprint.adapters:
            known = ", ".join(sorted(fingerprint.adapters)) or "none"
            return CheckResult(
                "reference",
                SKIP,
                f"fingerprint has no adapter {ref_name!r} (it has: {known}); use --map {name}=<name>",
            )
        outcome = compare_to_reference(
            fingerprint,
            ref_name,
            self.samples[name].isolated,
            self.samples[self.base].isolated,
            noise=max(self.samples[name].noise, self.samples[self.base].noise),
            abs_tolerance=self.cfg.reference_tolerance,
            noise_multiplier=self.cfg.noise_multiplier,
            scale_tolerance=self.cfg.scale_tolerance,
        )
        return CheckResult("reference", outcome.status, outcome.summary, outcome.evidence())

    def check_base_reference(self) -> CheckResult:
        fingerprint = self.cfg.fingerprint
        if fingerprint is None:
            return CheckResult("reference", SKIP, "no fingerprint supplied (--fingerprint)")
        from lora_check.fingerprint import _table_distance, detect_key_style

        observed = self.samples[self.base].isolated
        offset = _table_distance(observed, fingerprint.base, detect_key_style(observed))
        effects = [mean(fingerprint.sensitivity(a).values()) for a in fingerprint.adapters]
        smallest = min(effects) if effects else 0.0
        evidence = {"base_offset": round(offset, 6), "smallest_adapter_effect": round(smallest, 6)}
        tolerance = max(
            self.cfg.reference_tolerance, self.cfg.noise_multiplier * self.samples[self.base].noise
        )
        if offset <= tolerance:
            return CheckResult(
                "reference",
                PASS,
                f"base model matches the reference base (d={offset:.4f})",
                evidence,
            )
        if effects and offset > 0.5 * smallest:
            return CheckResult(
                "reference",
                WARN,
                f"base model is {offset:.4f} from the reference base - as large as an adapter's effect "
                f"({smallest:.4f}); different base weights, quantisation, tokenizer or template?",
                evidence,
            )
        return CheckResult(
            "reference",
            PASS,
            f"base model is {offset:.4f} from the reference base (numeric offset; tolerances widened to match)",
            evidence,
        )

    # -- assembly ----------------------------------------------------------

    def report(self) -> VerifyReport:
        base_s = self.samples[self.base]
        base = ModelReport(
            name=self.base, model_id=base_s.model_id, is_base=True, noise=base_s.noise
        )
        base.checks["stable"] = self.check_stable(self.base)
        base.checks["cache"] = self.check_cache(self.base)
        base.checks["reference"] = self.check_base_reference()
        adapters: list[ModelReport] = []
        for name in self.adapter_names:
            s = self.samples[name]
            rep = ModelReport(name=name, model_id=s.model_id, is_base=False, noise=s.noise)
            if s.error is not None:
                rep.checks["applied"] = CheckResult("applied", ERROR, f"requests failed: {s.error}")
            else:
                rep.checks["applied"] = self.check_applied(name)
                rep.checks["distinct"] = self.check_distinct(name)
                rep.checks["stable"] = self.check_stable(name)
                rep.checks["cache"] = self.check_cache(name)
                rep.checks["reference"] = self.check_reference(name)
            adapters.append(rep)
        if self.cfg.fingerprint is None:
            self.notes.append(
                "No fingerprint: PASS means each adapter is applied, distinct and stable - "
                "not that it is the right adapter at the right scale."
            )
        return VerifyReport(
            endpoint=self.client.base_url,
            base=base,
            adapters=adapters,
            probes=len(self.cfg.probes),
            repeats=self.cfg.repeats,
            requests=self.client.request_count,
            api=self.client.api,
            notes=self.notes,
        )


def run_verify(client: Client, config: VerifyConfig) -> VerifyReport:
    """Probe ``client``'s endpoint and return the per-adapter verdicts."""
    if not config.probes:
        raise EndpointError("no probes to send")
    if config.repeats < 2:
        raise EndpointError("--repeats must be at least 2 to measure run-to-run noise")
    models: list[ModelInfo] = []
    if config.adapters is None or config.base_model is None:
        models = client.list_models()
    base, adapters = discover(models, config.base_model, config.adapters)
    run = _Run(client, config, base, adapters)
    fingerprint = config.fingerprint
    if fingerprint is not None:
        if fingerprint.api != client.api:
            run.notes.append(
                f"fingerprint was computed for the {fingerprint.api!r} API but the endpoint is probed "
                f"through {client.api!r}; prompts may be templated differently."
            )
        if client.top_logprobs > fingerprint.top_k:
            run.notes.append(
                f"endpoint returns top-{client.top_logprobs} but the fingerprint stores only "
                f"top-{fingerprint.top_k}; comparison uses the shared part."
            )
    run.collect()
    return run.report()
