"""Reference fingerprints and the comparison of an endpoint against them.

A fingerprint is a small JSON file holding, for each probe, the top-K
next-token log-probabilities of the base model and of base+adapter as
computed by a trusted implementation (transformers + PEFT).  It can also hold
*variants*: the same adapter deliberately mis-applied in ways engines are
known to get wrong (a different global scale, ``rank_pattern`` /
``alpha_pattern`` ignored, rsLoRA ignored).  When the endpoint does not match
the reference, the closest variant names the likely cause.

This module has no heavy dependencies; computing a fingerprint from real
weights lives in :mod:`lora_check.reference`.
"""

from __future__ import annotations

import json
import math
from collections.abc import Mapping, Sequence
from dataclasses import dataclass, field
from pathlib import Path
from typing import Any

from lora_check.client import TOKEN_ID_PREFIX
from lora_check.dist import Dist, ScaleFit, distance, fit_scale, mean, truncate
from lora_check.probes import Probe

FORMAT = "lora-check/fingerprint"
FORMAT_VERSION = 1

TokenEntry = tuple[int | None, str, float]
"""(token id or None, decoded token text, log-probability)."""


class FingerprintError(ValueError):
    """Raised for unreadable or inconsistent fingerprint files."""


@dataclass(frozen=True)
class RefDist:
    """A reference top-K distribution that remembers ids and texts."""

    entries: tuple[TokenEntry, ...]

    def as_dist(self, key_style: str = "text", k: int = 0) -> Dist:
        """Convert to a :data:`Dist` keyed like the endpoint's answers."""
        out: Dist = {}
        for token_id, text, lp in self.entries:
            key = (
                f"{TOKEN_ID_PREFIX}{token_id}"
                if key_style == "id" and token_id is not None
                else text
            )
            # Different ids may decode to the same text; an endpoint that
            # reports text can only show one of them, the likelier.
            if key not in out or lp > out[key]:
                out[key] = lp
        return truncate(out, k) if k else out

    def to_json(self) -> list[list[Any]]:
        return [[i, t, round(lp, 6)] for i, t, lp in self.entries]

    @classmethod
    def from_json(cls, raw: object) -> RefDist:
        if not isinstance(raw, list):
            raise FingerprintError("distribution must be a list of [id, text, logprob]")
        entries: list[TokenEntry] = []
        for item in raw:
            if not isinstance(item, list) or len(item) != 3:
                raise FingerprintError("distribution entries must be [id, text, logprob]")
            token_id = int(item[0]) if item[0] is not None else None
            entries.append((token_id, str(item[1]), float(item[2])))
        return cls(tuple(entries))

    @classmethod
    def from_dist(cls, dist: Mapping[str, float]) -> RefDist:
        """Build from a plain distribution (no token ids)."""
        items = sorted(dist.items(), key=lambda kv: (-kv[1], kv[0]))
        return cls(tuple((None, token, lp) for token, lp in items))


DistTable = dict[str, RefDist]
"""Probe id -> reference distribution."""


@dataclass
class Variant:
    """The adapter applied in a deliberately wrong, named way."""

    name: str
    kind: str
    description: str
    dists: DistTable
    scale: float | None = None

    def to_json(self) -> dict[str, Any]:
        return {
            "name": self.name,
            "kind": self.kind,
            "description": self.description,
            "scale": self.scale,
            "dists": {pid: d.to_json() for pid, d in self.dists.items()},
        }

    @classmethod
    def from_json(cls, raw: Mapping[str, Any]) -> Variant:
        scale = raw.get("scale")
        return cls(
            name=str(raw["name"]),
            kind=str(raw.get("kind", "other")),
            description=str(raw.get("description", "")),
            scale=float(scale) if scale is not None else None,
            dists={str(pid): RefDist.from_json(d) for pid, d in dict(raw["dists"]).items()},
        )


@dataclass
class AdapterRef:
    """Reference behaviour of one adapter."""

    name: str
    reference: DistTable
    variants: list[Variant] = field(default_factory=list)
    config: dict[str, Any] = field(default_factory=dict)

    def to_json(self) -> dict[str, Any]:
        return {
            "config": self.config,
            "reference": {pid: d.to_json() for pid, d in self.reference.items()},
            "variants": [v.to_json() for v in self.variants],
        }

    @classmethod
    def from_json(cls, name: str, raw: Mapping[str, Any]) -> AdapterRef:
        return cls(
            name=name,
            reference={str(pid): RefDist.from_json(d) for pid, d in dict(raw["reference"]).items()},
            variants=[Variant.from_json(v) for v in raw.get("variants", [])],
            config=dict(raw.get("config", {})),
        )


@dataclass
class Fingerprint:
    """Everything `verify --fingerprint` needs."""

    base_model: str
    probes: list[Probe]
    base: DistTable
    adapters: dict[str, AdapterRef]
    api: str = "completions"
    top_k: int = 20
    meta: dict[str, Any] = field(default_factory=dict)

    def to_json(self) -> dict[str, Any]:
        return {
            "format": FORMAT,
            "version": FORMAT_VERSION,
            "base_model": self.base_model,
            "api": self.api,
            "top_k": self.top_k,
            "meta": self.meta,
            "probes": [{"id": p.id, "prompt": p.prompt} for p in self.probes],
            "base": {pid: d.to_json() for pid, d in self.base.items()},
            "adapters": {name: ref.to_json() for name, ref in self.adapters.items()},
        }

    @classmethod
    def from_json(cls, raw: object) -> Fingerprint:
        if not isinstance(raw, dict) or raw.get("format") != FORMAT:
            raise FingerprintError(f"not a lora-check fingerprint (missing format {FORMAT!r})")
        if int(raw.get("version", 0)) > FORMAT_VERSION:
            raise FingerprintError(
                f"fingerprint version {raw.get('version')} is newer than this lora-check supports"
            )
        try:
            probes = [Probe(id=str(p["id"]), prompt=str(p["prompt"])) for p in raw["probes"]]
            fingerprint = cls(
                base_model=str(raw.get("base_model", "")),
                probes=probes,
                base={str(pid): RefDist.from_json(d) for pid, d in dict(raw["base"]).items()},
                adapters={
                    str(name): AdapterRef.from_json(str(name), ref)
                    for name, ref in dict(raw.get("adapters", {})).items()
                },
                api=str(raw.get("api", "completions")),
                top_k=int(raw.get("top_k", 20)),
                meta=dict(raw.get("meta", {})),
            )
        except (KeyError, TypeError, ValueError) as exc:
            if isinstance(exc, FingerprintError):
                raise
            raise FingerprintError(f"malformed fingerprint: {exc}") from exc
        fingerprint.validate()
        return fingerprint

    def validate(self) -> None:
        ids = [p.id for p in self.probes]
        if len(set(ids)) != len(ids):
            raise FingerprintError("fingerprint probe ids are not unique")
        tables = [("base", self.base)]
        for name, ref in self.adapters.items():
            tables.append((f"adapter {name!r}", ref.reference))
            tables.extend((f"adapter {name!r} variant {v.name!r}", v.dists) for v in ref.variants)
        for label, table in tables:
            missing = [pid for pid in ids if pid not in table]
            if missing:
                raise FingerprintError(
                    f"fingerprint {label} lacks probes: {', '.join(missing[:5])}"
                )

    def save(self, path: str | Path) -> None:
        Path(path).write_text(
            json.dumps(self.to_json(), ensure_ascii=False, separators=(",", ":")) + "\n",
            encoding="utf-8",
        )

    @classmethod
    def load(cls, path: str | Path) -> Fingerprint:
        try:
            raw = json.loads(Path(path).read_text(encoding="utf-8"))
        except OSError as exc:
            raise FingerprintError(f"cannot read fingerprint {path}: {exc}") from exc
        except json.JSONDecodeError as exc:
            raise FingerprintError(f"{path}: invalid JSON: {exc}") from exc
        return cls.from_json(raw)

    def subset(self, probe_ids: Sequence[str]) -> Fingerprint:
        """A fingerprint restricted to the given probes (order preserved)."""
        wanted = list(probe_ids)
        by_id = {p.id: p for p in self.probes}
        unknown = [pid for pid in wanted if pid not in by_id]
        if unknown:
            raise FingerprintError(f"unknown probe ids: {', '.join(unknown[:5])}")

        def cut(table: DistTable) -> DistTable:
            return {pid: table[pid] for pid in wanted}

        return Fingerprint(
            base_model=self.base_model,
            probes=[by_id[pid] for pid in wanted],
            base=cut(self.base),
            adapters={
                name: AdapterRef(
                    name=name,
                    reference=cut(ref.reference),
                    variants=[
                        Variant(v.name, v.kind, v.description, cut(v.dists), v.scale)
                        for v in ref.variants
                    ],
                    config=ref.config,
                )
                for name, ref in self.adapters.items()
            },
            api=self.api,
            top_k=self.top_k,
            meta=self.meta,
        )

    def sensitivity(self, adapter: str) -> dict[str, float]:
        """Per-probe distance between the adapter reference and the base."""
        ref = self.adapters[adapter]
        return {
            p.id: distance(ref.reference[p.id].as_dist(), self.base[p.id].as_dist())
            for p in self.probes
        }


# ---------------------------------------------------------------------------
# Comparing an endpoint with a fingerprint
# ---------------------------------------------------------------------------


@dataclass(frozen=True)
class Candidate:
    """One explanation for what the endpoint is serving, with its distance."""

    label: str
    kind: str  # expected | base | adapter | variant
    distance: float
    description: str = ""
    scale: float | None = None


@dataclass
class ReferenceOutcome:
    """Result of comparing one served adapter with its reference."""

    status: str  # PASS | WARN | FAIL
    summary: str
    cause: str
    candidates: list[Candidate]
    tolerance: float
    effect: float
    base_offset: float | None
    fit: ScaleFit | None

    def evidence(self) -> dict[str, Any]:
        out: dict[str, Any] = {
            "cause": self.cause,
            "tolerance": round(self.tolerance, 6),
            "adapter_effect": round(self.effect, 6),
            "candidates": [
                {"label": c.label, "kind": c.kind, "distance": round(c.distance, 6)}
                for c in self.candidates
            ],
        }
        if self.base_offset is not None:
            out["base_offset"] = round(self.base_offset, 6)
        if self.fit is not None and self.fit.usable:
            out["fitted_scale"] = round(self.fit.scale, 4)
            out["fit_r2"] = round(self.fit.r2, 4)
        return out


def detect_key_style(observed: Mapping[str, Sequence[Mapping[str, float]]]) -> str:
    """``"id"`` if the endpoint reports ``token_id:<n>`` keys, else ``"text"``."""
    for repeats in observed.values():
        for dist in repeats:
            if dist:
                return "id" if all(k.startswith(TOKEN_ID_PREFIX) for k in dist) else "text"
    return "text"


def _table_distance(
    observed: Mapping[str, Sequence[Mapping[str, float]]], table: DistTable, key_style: str
) -> float:
    values: list[float] = []
    for pid, repeats in observed.items():
        ref = table[pid]
        for dist in repeats:
            values.append(distance(dist, ref.as_dist(key_style, len(dist))))
    return mean(values)


def scale_hypotheses(config: Mapping[str, Any]) -> list[tuple[float, str]]:
    """Scale factors that specific, known mistakes would produce.

    Derived only from the adapter's own config, using PEFT's documented
    scaling rules (``lora_alpha / r``, or ``lora_alpha / sqrt(r)`` with
    ``use_rslora``).
    """
    out: list[tuple[float, str]] = []
    try:
        r = float(config.get("r", 0))
        alpha = float(config.get("lora_alpha", 0))
    except (TypeError, ValueError):
        return out
    if r <= 0 or alpha <= 0:
        return out
    if config.get("use_rslora"):
        out.append(
            (
                1.0 / math.sqrt(r),
                "use_rslora ignored (lora_alpha/r used instead of lora_alpha/sqrt(r))",
            )
        )
    else:
        out.append(
            (math.sqrt(r), "adapter treated as rsLoRA (lora_alpha/sqrt(r) instead of lora_alpha/r)")
        )
    if abs(alpha - r) > 1e-9:
        nominal = alpha / math.sqrt(r) if config.get("use_rslora") else alpha / r
        out.append((1.0 / nominal, "lora_alpha ignored (scale 1.0 used)"))
    return out


def _explain_scale(scale: float, config: Mapping[str, Any]) -> str:
    for factor, text in scale_hypotheses(config):
        if factor > 0 and abs(scale - factor) <= 0.1 * factor:
            return f"; {factor:.3g}x is what you get with {text}"
    return ""


def _estimate_scale(
    fingerprint: Fingerprint,
    ref: AdapterRef,
    observed: Mapping[str, Sequence[Mapping[str, float]]],
    key_style: str,
) -> ScaleFit:
    """Estimate the scale the endpoint applies the adapter at.

    The references at known scales (0 = base, 1 = the adapter, plus any stored
    ``scale=`` variants) form a grid.  Within each pair of neighbouring grid
    points the observed answers are fitted as an interpolation between the
    two; the best-explained bracket gives the estimate.  With no stored
    variants this reduces to a single linear fit between base and adapter,
    which is exact only to first order.
    """
    grid: list[tuple[float, DistTable]] = [(0.0, fingerprint.base), (1.0, ref.reference)]
    grid.extend((v.scale, v.dists) for v in ref.variants if v.kind == "scale" and v.scale)
    grid.sort(key=lambda item: item[0])
    best: ScaleFit | None = None
    for index in range(len(grid) - 1):
        (lo, lo_table), (hi, hi_table) = grid[index], grid[index + 1]
        triples: list[tuple[Dist, Dist, Mapping[str, float]]] = []
        for pid, repeats in observed.items():
            lo_dist = lo_table[pid].as_dist(key_style)
            hi_dist = hi_table[pid].as_dist(key_style)
            triples.extend((lo_dist, hi_dist, dist) for dist in repeats)
        local = fit_scale(triples)
        if not local.usable:
            continue
        last = index == len(grid) - 2
        if local.scale < -0.15 or (local.scale > 1.15 and not last):
            continue
        candidate = ScaleFit(lo + local.scale * (hi - lo), local.r2, local.points)
        if best is None or candidate.r2 > best.r2:
            best = candidate
    return best if best is not None else ScaleFit(float("nan"), 0.0, 0)


def compare_to_reference(
    fingerprint: Fingerprint,
    adapter: str,
    observed: Mapping[str, Sequence[Mapping[str, float]]],
    observed_base: Mapping[str, Sequence[Mapping[str, float]]] | None,
    noise: float,
    abs_tolerance: float = 0.02,
    noise_multiplier: float = 3.0,
    scale_tolerance: float = 0.15,
) -> ReferenceOutcome:
    """Decide whether the endpoint serves ``adapter`` as the fingerprint says.

    ``observed`` maps probe id to the endpoint's repeated answers for the
    adapter; ``observed_base`` does the same for the base model and is used to
    measure the engine's numeric offset from the reference implementation
    (dtype, kernels, quantisation) so that the tolerance adapts to it.
    """
    ref = fingerprint.adapters[adapter]
    key_style = detect_key_style(observed)

    base_offset: float | None = None
    if observed_base:
        base_offset = _table_distance(observed_base, fingerprint.base, key_style)
    effect = mean(fingerprint.sensitivity(adapter).values())
    # The fixed tolerance may not swallow the adapter itself: cap it at a tenth
    # of the adapter's effect, then widen it by whatever the endpoint's noise
    # and its measured numeric offset from the reference demand.
    tolerance = max(
        min(abs_tolerance, 0.1 * effect),
        noise_multiplier * noise,
        noise_multiplier * (base_offset or 0.0),
    )
    candidates = [
        Candidate("reference", "expected", _table_distance(observed, ref.reference, key_style)),
        Candidate("base model", "base", _table_distance(observed, fingerprint.base, key_style)),
    ]
    for other_name, other in fingerprint.adapters.items():
        if other_name != adapter:
            candidates.append(
                Candidate(
                    f"adapter {other_name!r}",
                    "adapter",
                    _table_distance(observed, other.reference, key_style),
                )
            )
    for variant in ref.variants:
        candidates.append(
            Candidate(
                f"variant {variant.name!r}",
                "variant",
                _table_distance(observed, variant.dists, key_style),
                description=variant.description,
                scale=variant.scale,
            )
        )
    expected = candidates[0]
    ranked = sorted(candidates, key=lambda c: c.distance)
    best = ranked[0]

    fit = _estimate_scale(fingerprint, ref, observed, key_style)
    # A scale estimate is only trustworthy when the adapter's effect clearly
    # dominates the numeric disagreement between engine and reference.
    fit_trusted = fit.usable and fit.r2 >= 0.6 and effect >= 5.0 * tolerance / noise_multiplier

    def outcome(status: str, summary: str, cause: str) -> ReferenceOutcome:
        return ReferenceOutcome(
            status=status,
            summary=summary,
            cause=cause,
            candidates=ranked,
            tolerance=tolerance,
            effect=effect,
            base_offset=base_offset,
            fit=fit,
        )

    # A wrong candidate must beat the reference clearly, not by noise.
    margin = max(noise, 0.25 * tolerance)
    clearly_better = best is not expected and (
        best.distance < 0.5 * expected.distance and expected.distance - best.distance > margin
    )
    within = expected.distance <= tolerance
    scale_off = fit_trusted and abs(fit.scale - 1.0) > scale_tolerance

    unreliable = (
        base_offset is not None
        and base_offset > max(abs_tolerance, noise_multiplier * noise)
        and base_offset > 0.25 * effect
    )
    if unreliable:
        return outcome(
            "WARN",
            f"endpoint base model is {base_offset:.4f} from the reference base, comparable to "
            f"the adapter's own effect ({effect:.4f}): different base weights, quantisation, "
            "tokenizer or prompt template - the adapter cannot be validated against this fingerprint",
            "base-mismatch",
        )
    if within and not clearly_better and not scale_off:
        extra = f", fitted scale {fit.scale:.2f}" if fit_trusted else ""
        return outcome(
            "PASS",
            f"matches reference (d={expected.distance:.4f}, tolerance {tolerance:.4f}{extra})",
            "match",
        )

    d_exp = f"d={expected.distance:.4f} to the reference"
    firm = best is not expected and best.distance <= tolerance
    estimate = f"; estimated scale {fit.scale:.2f}x (R2={fit.r2:.2f})" if fit_trusted else ""
    if firm and best.kind == "base":
        return outcome(
            "FAIL",
            f"endpoint matches the BASE model (d={best.distance:.4f}), not the adapter ({d_exp}): "
            "the adapter is loaded but ignored",
            "adapter-ignored",
        )
    if firm and best.kind == "adapter":
        return outcome(
            "FAIL",
            f"endpoint matches {best.label} (d={best.distance:.4f}), not this adapter ({d_exp}): "
            "adapters are swapped or mis-routed",
            "wrong-adapter",
        )
    if firm and best.kind == "variant":
        note = _explain_scale(best.scale, ref.config) if best.scale is not None else ""
        return outcome(
            "FAIL",
            f"endpoint matches {best.label} (d={best.distance:.4f}), not the reference ({d_exp}): "
            f"{best.description}{note}{estimate}",
            "wrong-scale" if best.scale is not None else "wrong-application",
        )
    if scale_off:
        return outcome(
            "FAIL",
            f"adapter is applied at about {fit.scale:.2f}x its intended strength "
            f"(fit against the stored references, R2={fit.r2:.2f}; {d_exp})"
            f"{_explain_scale(fit.scale, ref.config)}: scaling / lora_alpha is handled differently",
            "wrong-scale",
        )
    if clearly_better:
        causes = {
            "base": "adapter-ignored",
            "adapter": "wrong-adapter",
            "variant": "wrong-application",
        }
        return outcome(
            "FAIL",
            f"endpoint is much closer to {best.label} (d={best.distance:.4f}) than to the reference "
            f"({d_exp}), without matching it exactly (tolerance {tolerance:.4f}){estimate}",
            causes[best.kind],
        )
    return outcome(
        "FAIL",
        f"endpoint matches no known reference ({d_exp}, tolerance {tolerance:.4f}; closest is "
        f"{best.label} at d={best.distance:.4f}{estimate}): different adapter weights or revision, a stale "
        "cache, partial application, or a tokenizer/template mismatch",
        "unknown-mismatch",
    )
