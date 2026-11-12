"""Static checks of a PEFT LoRA adapter before it is served.

Generic rules come from the PEFT documentation of ``adapter_config.json``.
Engine rules state how a specific engine treats a feature and each carries
the URL it was taken from and whether that is documentation, source code or
an issue.  Nothing here executes a model.
"""

from __future__ import annotations

import math
import re
from collections.abc import Mapping
from dataclasses import dataclass, field
from typing import Any

from lora_check.lint.io import Source, SourceError, TensorInfo

ERROR = "error"
WARNING = "warning"
INFO = "info"
_ORDER = {ERROR: 0, WARNING: 1, INFO: 2}

CHECKED = "2026-11-12"
"""Date on which the cited engine sources were last read."""

ENGINES = ("vllm", "sglang")


@dataclass(frozen=True)
class Rule:
    """A lint rule.  Engine rules must cite a source."""

    id: str
    title: str
    engine: str | None = None
    source: str | None = None
    source_kind: str | None = None  # docs | source | issue
    fixed_in: str | None = None
    """First engine release in which the behaviour is fixed, when verified."""


_PEFT_DOC = "https://huggingface.co/docs/peft/main/en/package_reference/lora"
_VLLM_HELPER = "https://github.com/vllm-project/vllm/blob/main/vllm/lora/peft_helper.py"
_VLLM_DOC = "https://docs.vllm.ai/en/latest/features/lora.html"
_SGL_DOC = "https://docs.sglang.io/advanced_features/lora.html"
_SGL_MANAGER = (
    "https://github.com/sgl-project/sglang/blob/main/python/sglang/srt/lora/lora_manager.py"
)
_SGL_LORA = "https://github.com/sgl-project/sglang/blob/main/python/sglang/srt/lora/lora.py"

RULES: dict[str, Rule] = {
    r.id: r
    for r in (
        Rule(
            "L001",
            "adapter_config.json is missing or unreadable",
            source=_PEFT_DOC,
            source_kind="docs",
        ),
        Rule("L002", "no adapter weights file", source=_PEFT_DOC, source_kind="docs"),
        Rule("L003", "weights are a pickle (.bin) and cannot be inspected safely"),
        Rule("L004", "adapter is not a LoRA (peft_type)", source=_PEFT_DOC, source_kind="docs"),
        Rule(
            "L010", "required config field missing or invalid", source=_PEFT_DOC, source_kind="docs"
        ),
        Rule(
            "L020",
            "per-module scaling differs (rank_pattern / alpha_pattern)",
            source=_PEFT_DOC,
            source_kind="docs",
        ),
        Rule("L021", "rank-stabilised scaling (use_rslora)", source=_PEFT_DOC, source_kind="docs"),
        Rule("L022", "DoRA adapter (use_dora)", source=_PEFT_DOC, source_kind="docs"),
        Rule(
            "L023", "adapter trains base-model biases (bias)", source=_PEFT_DOC, source_kind="docs"
        ),
        Rule(
            "L024",
            "full modules saved next to the LoRA (modules_to_save)",
            source=_PEFT_DOC,
            source_kind="docs",
        ),
        Rule(
            "L025",
            "initialisation that modifies the base weights (init_lora_weights)",
            source=_PEFT_DOC,
            source_kind="docs",
        ),
        Rule(
            "L026",
            "LoRA on nn.Parameter targets (target_parameters)",
            source=_PEFT_DOC,
            source_kind="docs",
        ),
        Rule("L030", "weights do not match the config", source=_PEFT_DOC, source_kind="docs"),
        Rule("L031", "unexpected weight key layout"),
        Rule("L040", "adapter modules not found in the base checkpoint"),
        Rule("L041", "adapter and base module shapes disagree"),
        Rule("L042", "base_model_name_or_path differs from --base"),
        Rule("L043", "adapter and base module paths differ by a prefix"),
        Rule(
            "L050",
            "adapter changes the tokenizer or embeddings",
            source=_PEFT_DOC,
            source_kind="docs",
        ),
        Rule("V101", "vLLM does not support DoRA", "vllm", _VLLM_HELPER, "source"),
        Rule("V102", "vLLM does not support adapter bias", "vllm", _VLLM_HELPER, "source"),
        Rule(
            "V103",
            "vLLM only supports modules_to_save for classifier heads",
            "vllm",
            _VLLM_HELPER,
            "source",
        ),
        Rule(
            "V104",
            "vLLM rejects inits that modify the base weights",
            "vllm",
            _VLLM_HELPER,
            "source",
        ),
        Rule(
            "V105", "vLLM does not implement this PEFT LoRA feature", "vllm", _VLLM_HELPER, "source"
        ),
        Rule("V106", "rank exceeds vLLM's default --max-lora-rank", "vllm", _VLLM_DOC, "docs"),
        Rule(
            "V107",
            "vLLM applied one lora_alpha/r scale to every module",
            "vllm",
            "https://github.com/vllm-project/vllm/issues/59799",
            "issue",
            fixed_in="0.32.0",
        ),
        Rule(
            "V108",
            "vLLM validates only the last component of module names",
            "vllm",
            "https://github.com/vllm-project/vllm/issues/34186",
            "issue",
        ),
        Rule(
            "V109",
            "MoE expert LoRA needs extra vLLM settings and is unsafe with DP > 1",
            "vllm",
            "https://github.com/vllm-project/vllm/issues/61453",
            "issue",
        ),
        Rule(
            "V110",
            "rsLoRA was not supported by old vLLM releases",
            "vllm",
            "https://github.com/vllm-project/vllm/issues/10798",
            "issue",
        ),
        Rule("S201", "SGLang does not support DoRA adapters", "sglang", _SGL_MANAGER, "source"),
        Rule(
            "S202",
            "SGLang does not support adapters that add tokens",
            "sglang",
            _SGL_MANAGER,
            "source",
        ),
        Rule(
            "S203",
            "SGLang uses a single lora_alpha/r scale per adapter",
            "sglang",
            _SGL_LORA,
            "source",
        ),
        Rule(
            "S204",
            "SGLang LoRA on the embedding layer is under development",
            "sglang",
            _SGL_DOC,
            "docs",
        ),
        Rule(
            "S205",
            "SGLang must be started with matching rank and target modules",
            "sglang",
            _SGL_DOC,
            "docs",
        ),
    )
}


@dataclass
class Finding:
    """One lint result."""

    rule: str
    severity: str
    message: str

    @property
    def info(self) -> Rule:
        return RULES[self.rule]

    def to_json(self) -> dict[str, Any]:
        rule = self.info
        return {
            "rule": self.rule,
            "severity": self.severity,
            "title": rule.title,
            "message": self.message,
            "engine": rule.engine,
            "source": rule.source,
            "source_kind": rule.source_kind,
        }


@dataclass
class LintReport:
    """Result of :func:`lint_adapter`."""

    adapter: str
    base: str | None
    engines: list[str]
    engine_version: str | None
    summary: dict[str, Any] = field(default_factory=dict)
    findings: list[Finding] = field(default_factory=list)

    def add(self, rule: str, severity: str, message: str) -> None:
        self.findings.append(Finding(rule, severity, message))

    def count(self, severity: str) -> int:
        return sum(1 for f in self.findings if f.severity == severity)

    def exit_code(self, strict: bool = False) -> int:
        if self.count(ERROR) or (strict and self.count(WARNING)):
            return 1
        return 0

    def sorted_findings(self) -> list[Finding]:
        return sorted(self.findings, key=lambda f: (_ORDER[f.severity], f.rule))

    def to_json(self) -> dict[str, Any]:
        return {
            "tool": "lora-check",
            "command": "lint",
            "adapter": self.adapter,
            "base": self.base,
            "engines": self.engines,
            "engine_version": self.engine_version,
            "sources_checked": CHECKED,
            "errors": self.count(ERROR),
            "warnings": self.count(WARNING),
            "summary": self.summary,
            "findings": [f.to_json() for f in self.sorted_findings()],
        }


# ---------------------------------------------------------------------------
# Helpers
# ---------------------------------------------------------------------------

_KEY_RE = re.compile(
    r"^(?P<module>.+?)\.(?P<part>lora_A|lora_B|lora_embedding_A|lora_embedding_B|lora_magnitude_vector)"
    r"(?:\.(?P<adapter>[^.]+))??(?:\.(?:weight|bias))?$"
)
PEFT_PREFIX = "base_model.model."
_BASE_MODIFYING_INITS = ("pissa", "olora", "corda", "loftq", "astra")
_EMBED_HINTS = ("embed", "wte", "lm_head", "shared")


def parse_version(text: str) -> tuple[int, ...]:
    """``"0.30.0rc1"`` -> ``(0, 30, 0)``; unparsable parts are dropped."""
    parts: list[int] = []
    for piece in text.strip().lstrip("v").split("."):
        match = re.match(r"\d+", piece)
        if not match:
            break
        parts.append(int(match.group()))
    return tuple(parts)


def match_pattern(pattern: Mapping[str, Any], module: str) -> Any:
    """PEFT's rule for ``rank_pattern``/``alpha_pattern``: a key matches if it
    is a (regex) suffix of the module name starting at a dot boundary; the
    first matching key wins."""
    for key, value in pattern.items():
        try:
            if re.match(rf"(.*\.)?({key})$", module):
                return value
        except re.error:
            if module == key or module.endswith("." + key):
                return value
    return None


@dataclass
class _Module:
    name: str
    a: TensorInfo | None = None
    b: TensorInfo | None = None
    embedding: bool = False
    dora: bool = False

    @property
    def rank(self) -> int | None:
        return self.a.shape[0] if self.a is not None and self.a.shape else None


def _collect_modules(
    tensors: Mapping[str, TensorInfo],
) -> tuple[dict[str, _Module], list[str], bool]:
    """Group LoRA tensors by module.  Returns (modules, other keys, has_prefix)."""
    modules: dict[str, _Module] = {}
    others: list[str] = []
    prefixed = 0
    for key, info in tensors.items():
        stripped = key
        if key.startswith(PEFT_PREFIX):
            stripped = key[len(PEFT_PREFIX) :]
            prefixed += 1
        match = _KEY_RE.match(stripped)
        if not match:
            others.append(stripped)
            continue
        module = modules.setdefault(match["module"], _Module(match["module"]))
        part = match["part"]
        if part in ("lora_A", "lora_embedding_A"):
            module.a = info
        elif part in ("lora_B", "lora_embedding_B"):
            module.b = info
        else:
            module.dora = True
        if part.startswith("lora_embedding"):
            module.embedding = True
    return modules, others, prefixed * 2 >= max(1, len(tensors))


def _scaling(alpha: float, rank: float, rslora: bool) -> float:
    return alpha / math.sqrt(rank) if rslora else alpha / rank


def _fmt_list(items: list[str], limit: int = 4) -> str:
    shown = ", ".join(items[:limit])
    return shown + (f" (+{len(items) - limit} more)" if len(items) > limit else "")


def _affected(rule: Rule, version: str | None) -> bool | None:
    """True/False if the version settles it, None if unknown."""
    if rule.fixed_in is None or not version:
        return None
    return parse_version(version) < parse_version(rule.fixed_in)


# ---------------------------------------------------------------------------
# The linter
# ---------------------------------------------------------------------------


def lint_adapter(
    adapter: Source,
    base: Source | None = None,
    engine: str | None = None,
    engine_version: str | None = None,
) -> LintReport:
    """Run all static checks on ``adapter`` (optionally against ``base``)."""
    engines = [engine] if engine else list(ENGINES)
    report = LintReport(
        adapter=adapter.label,
        base=base.label if base else None,
        engines=engines,
        engine_version=engine_version if engine else None,
    )
    try:
        files = adapter.files()
        config = adapter.read_json("adapter_config.json")
    except SourceError as exc:
        report.add("L001", ERROR, str(exc))
        return report
    if not isinstance(config, dict):
        report.add(
            "L001", ERROR, "adapter_config.json not found; this does not look like a PEFT adapter"
        )
        return report

    peft_type = str(config.get("peft_type", "LORA")).upper()
    if peft_type != "LORA":
        report.add(
            "L004",
            ERROR,
            f"peft_type is {peft_type!r}; lora-check and the multi-LoRA engines only handle LORA",
        )

    r = config.get("r")
    alpha = config.get("lora_alpha")
    for name, value in (("r", r), ("lora_alpha", alpha)):
        if not isinstance(value, (int, float)) or isinstance(value, bool) or value <= 0:
            report.add("L010", ERROR, f"{name} must be a positive number, got {value!r}")
    if "target_modules" not in config and not config.get("target_parameters"):
        report.add("L010", ERROR, "target_modules is missing")
    r_num = float(r) if isinstance(r, (int, float)) and not isinstance(r, bool) and r > 0 else None
    alpha_num = (
        float(alpha)
        if isinstance(alpha, (int, float)) and not isinstance(alpha, bool) and alpha > 0
        else None
    )
    rslora = bool(config.get("use_rslora"))
    rank_pattern = config.get("rank_pattern") or {}
    alpha_pattern = config.get("alpha_pattern") or {}

    # -- weights -----------------------------------------------------------
    tensors: dict[str, TensorInfo] = {}
    weight_file = next((f for f in ("adapter_model.safetensors",) if f in files), None)
    if weight_file is None:
        if "adapter_model.bin" in files:
            report.add(
                "L003",
                WARNING,
                "only adapter_model.bin (pickle) is present; keys and shapes were not checked. "
                "Re-save with safetensors to enable the weight checks.",
            )
        else:
            report.add(
                "L002", ERROR, "neither adapter_model.safetensors nor adapter_model.bin found"
            )
    else:
        try:
            tensors = adapter.tensors(weight_file)
        except SourceError as exc:
            report.add("L002", ERROR, f"{weight_file}: {exc}")
    modules, others, prefixed = _collect_modules(tensors)

    # -- per-module scaling ------------------------------------------------
    scalings: dict[str, float] = {}
    nominal: float | None = None
    if r_num is not None and alpha_num is not None:
        nominal = _scaling(alpha_num, r_num, False)
        names = list(modules) or [
            str(m) for m in (config.get("target_modules") or []) if isinstance(m, str)
        ]
        for name in names:
            rank_m = match_pattern(rank_pattern, name)
            alpha_m = match_pattern(alpha_pattern, name)
            scalings[name] = _scaling(
                float(alpha_m) if isinstance(alpha_m, (int, float)) else alpha_num,
                float(rank_m) if isinstance(rank_m, (int, float)) and rank_m > 0 else r_num,
                rslora,
            )
    off = {
        name: s
        for name, s in scalings.items()
        if nominal and abs(s - nominal) > 1e-9 * max(1.0, nominal)
    }
    ranks = sorted({m.rank for m in modules.values() if m.rank})
    report.summary = {
        "r": r,
        "lora_alpha": alpha,
        "use_rslora": rslora,
        "use_dora": bool(config.get("use_dora")),
        "bias": config.get("bias", "none"),
        "target_modules": config.get("target_modules"),
        "base_model_name_or_path": config.get("base_model_name_or_path"),
        "lora_modules_in_weights": len(modules),
        "ranks_in_weights": ranks,
        "distinct_scalings": sorted({round(s, 6) for s in scalings.values()}),
    }

    _lint_features(
        report, config, modules, off, nominal, rslora, engine_version if engine else None
    )
    _lint_weights(report, config, modules, others, prefixed, tensors, rank_pattern)
    base_vocab = _lint_base(report, config, base, modules) if base is not None else None
    _lint_tokenizer(report, adapter, files, config, modules, others, tensors, base_vocab)
    if "sglang" in engines and (modules or config.get("target_modules")):
        report.add(
            "S205",
            INFO,
            "start SGLang with --max-lora-rank and --lora-target-modules covering this adapter if it is "
            "loaded dynamically; when they are inferred from --lora-paths, later adapters must fit that shape",
        )
    return report


def _lint_features(
    report: LintReport,
    config: Mapping[str, Any],
    modules: Mapping[str, _Module],
    off: Mapping[str, float],
    nominal: float | None,
    rslora: bool,
    version: str | None,
) -> None:
    vllm = "vllm" in report.engines
    sglang = "sglang" in report.engines
    patterns = bool(config.get("rank_pattern") or config.get("alpha_pattern"))

    if patterns:
        if off and nominal:
            factors = sorted({round(s / nominal, 4) for s in off.values()})
            detail = (
                f"{len(off)} module(s) have a scale different from lora_alpha/r = {nominal:g} "
                f"(by factors {', '.join(f'{f:g}x' for f in factors[:6])}), e.g. {_fmt_list(sorted(off), 2)}"
            )
            report.add("L020", INFO, f"rank_pattern/alpha_pattern present: {detail}")
            if vllm:
                rule = RULES["V107"]
                affected = _affected(rule, version)
                if affected is not False:
                    which = (
                        f"vLLM {version} is affected"
                        if affected
                        else "vLLM releases before 0.32.0 are affected (pass --engine-version to settle this)"
                    )
                    report.add(
                        "V107",
                        ERROR if affected else WARNING,
                        f"{which}: one lora_alpha/r scale was applied to every module, so those "
                        f"{len(off)} module(s) are served at the wrong strength with no warning. "
                        "Upgrade, or fold the per-module scale into lora_B and set lora_alpha = r.",
                    )
            if sglang:
                report.add(
                    "S203",
                    WARNING,
                    "SGLang computes a single scaling = lora_alpha / r per adapter and its LoRA loader "
                    "does not read rank_pattern or alpha_pattern, so those modules would be served at "
                    "the wrong strength. Confirm with `lora-check verify --fingerprint`.",
                )
        else:
            report.add(
                "L020",
                INFO,
                "rank_pattern/alpha_pattern present but every module ends up with the scale lora_alpha/r",
            )
    if rslora:
        r = config.get("r")
        report.add(
            "L021",
            INFO,
            f"use_rslora is set: the scale is lora_alpha/sqrt(r), not lora_alpha/r (a factor "
            f"{math.sqrt(float(r)):.3g}x apart for r={r})"
            if isinstance(r, (int, float)) and r > 0
            else "use_rslora is set: the scale is lora_alpha/sqrt(r), not lora_alpha/r",
        )
        if vllm and version and parse_version(version) <= (0, 6, 4):
            report.add(
                "V110",
                WARNING,
                f"vLLM {version}: rsLoRA adapters were reported to be served without the rsLoRA scale "
                "in 0.6.4.post1; current vLLM reads use_rslora",
            )
        if sglang and not patterns:
            report.add(
                "S203",
                WARNING,
                "SGLang computes scaling = lora_alpha / r and its LoRA loader does not read use_rslora, "
                "so this adapter would be served too weak. Confirm with `lora-check verify --fingerprint`.",
            )
    dora = bool(config.get("use_dora")) or any(m.dora for m in modules.values())
    if dora:
        report.add(
            "L022", INFO, "DoRA adapter: needs the magnitude vectors, not just lora_A/lora_B"
        )
        if vllm:
            report.add(
                "V101",
                ERROR,
                'vLLM rejects the adapter at load time: "vLLM does not yet support DoRA."',
            )
        if sglang:
            report.add(
                "S201",
                ERROR,
                "SGLang rejects the adapter: LoRA serving does not support DoRA adapters",
            )
    bias = config.get("bias", "none")
    if bias not in (None, "none"):
        report.add(
            "L023",
            WARNING,
            f"bias={bias!r}: base-model biases were trained, so the base model itself must change; "
            "per PEFT, even with the adapter disabled the model no longer matches the base",
        )
        if vllm:
            report.add(
                "V102",
                ERROR,
                'vLLM rejects the adapter at load time: "Adapter bias is not supported."',
            )
    saved = [str(m) for m in (config.get("modules_to_save") or [])]
    if saved:
        report.add(
            "L024",
            WARNING,
            f"modules_to_save={saved}: these modules are fully replaced, not low-rank; an engine that "
            "loads only lora_A/lora_B will silently drop them",
        )
        unsupported = [m for m in saved if m not in ("classifier", "score")]
        if vllm and unsupported:
            report.add(
                "V103",
                ERROR,
                f"vLLM only supports modules_to_save of classifier/score heads; unsupported: {unsupported}",
            )
    init = config.get("init_lora_weights", True)
    if isinstance(init, str) and init.lower().split("_niter_")[0] in _BASE_MODIFYING_INITS:
        report.add(
            "L025",
            WARNING,
            f"init_lora_weights={init!r} modifies the base weights when PEFT loads the adapter; serving "
            "it on the unmodified base gives different outputs unless the adapter was converted",
        )
        if vllm:
            report.add(
                "V104",
                ERROR,
                f"vLLM rejects init_lora_weights={init!r}; convert it to a plain LoRA first",
            )
    if vllm:
        extra = [
            name
            for name in (
                "lora_bias",
                "alora_invocation_tokens",
                "layer_replication",
                "use_bdlora",
                "use_qalora",
            )
            if config.get(name)
        ]
        if extra:
            report.add("V105", ERROR, f"vLLM rejects adapters using {', '.join(extra)}")
        ranks = [m.rank for m in modules.values() if m.rank]
        r = config.get("r")
        top = max(
            [
                *ranks,
                *(v for v in (config.get("rank_pattern") or {}).values() if isinstance(v, int)),
            ],
            default=0,
        )
        if isinstance(r, int):
            top = max(top, r)
        if top > 16:
            report.add(
                "V106",
                INFO,
                f"highest rank is {top}: start vLLM with --max-lora-rank {top} (the default is 16) "
                "or the adapter is refused",
            )
        if modules or config.get("target_modules"):
            report.add(
                "V108",
                INFO,
                "vLLM checks only the last component of module names when loading; an adapter whose "
                "module paths do not line up with the served model class can load cleanly and never be "
                "applied. Run `lora-check verify` after loading"
                + ("" if report.base else "; pass --base to compare paths"),
            )
    if config.get("target_parameters"):
        report.add(
            "L026",
            INFO,
            f"target_parameters={config.get('target_parameters')}: LoRA on raw parameters (MoE experts)",
        )
        if vllm:
            report.add(
                "V109",
                WARNING,
                "expert LoRA on MoE models: reports say models not flagged as 3D-MoE need both "
                "enable_mixed_moe_lora_format and is_3d_lora_weight (#59799), and that with data "
                "parallelism > 1 expert deltas can be dropped or cross-applied between replicas (#61453)",
            )


def _lint_weights(
    report: LintReport,
    config: Mapping[str, Any],
    modules: Mapping[str, _Module],
    others: list[str],
    prefixed: bool,
    tensors: Mapping[str, TensorInfo],
    rank_pattern: Mapping[str, Any],
) -> None:
    if not tensors:
        return
    if not modules:
        report.add("L030", ERROR, "the weights file contains no lora_A/lora_B tensors")
        return
    if not prefixed:
        report.add(
            "L031",
            WARNING,
            "weight keys do not start with 'base_model.model.' as PEFT writes them; engines may not find the modules",
        )
    incomplete = sorted(n for n, m in modules.items() if m.a is None or m.b is None)
    if incomplete:
        report.add(
            "L030", ERROR, f"modules with only one of lora_A/lora_B: {_fmt_list(incomplete)}"
        )
    r = config.get("r")
    wrong_rank: list[str] = []
    bad_pair: list[str] = []
    for name, module in modules.items():
        if (
            module.a is None
            or module.b is None
            or len(module.a.shape) < 2
            or len(module.b.shape) < 2
        ):
            continue
        expected = match_pattern(rank_pattern, name) or r
        if isinstance(expected, int) and module.a.shape[0] != expected:
            wrong_rank.append(f"{name} (has {module.a.shape[0]}, config says {expected})")
        if module.a.shape[0] != module.b.shape[1]:
            bad_pair.append(name)
    if wrong_rank:
        report.add(
            "L030",
            ERROR,
            f"tensor rank differs from r/rank_pattern: {_fmt_list(sorted(wrong_rank), 3)}",
        )
    if bad_pair:
        report.add(
            "L030", ERROR, f"lora_A and lora_B ranks disagree: {_fmt_list(sorted(bad_pair))}"
        )
    targets = config.get("target_modules")
    if isinstance(targets, list) and targets:
        leafs = {name.rsplit(".", 1)[-1] for name in modules}
        names = [str(t) for t in targets]
        unused = sorted(t for t in names if not any(n == t or n.endswith("." + t) for n in modules))
        if unused:
            report.add(
                "L030",
                WARNING,
                f"target_modules with no tensors in the weights: {_fmt_list(unused)}",
            )
        wanted = {t.rsplit(".", 1)[-1] for t in names}
        stray = sorted(leafs - wanted)
        if stray and not config.get("target_parameters"):
            report.add(
                "L030",
                WARNING,
                f"weights contain modules not named in target_modules: {_fmt_list(stray)}",
            )


def _base_tensors(base: Source) -> dict[str, TensorInfo]:
    names = [f for f in base.files() if f.endswith(".safetensors") and not f.startswith("adapter_")]
    tensors: dict[str, TensorInfo] = {}
    for name in names:
        tensors.update(base.tensors(name))
    return tensors


def _lint_base(
    report: LintReport,
    config: Mapping[str, Any],
    base: Source,
    modules: Mapping[str, _Module],
) -> int | None:
    """Compare module paths and shapes with the base checkpoint; returns its vocab size."""
    declared = config.get("base_model_name_or_path")
    if isinstance(declared, str) and declared and declared.rstrip("/") != base.label.rstrip("/"):
        report.add(
            "L042",
            INFO,
            f"adapter_config.json names base {declared!r} but it is being checked against {base.label!r}",
        )
    vocab: int | None = None
    try:
        base_config = base.read_json("config.json")
        if isinstance(base_config, dict) and isinstance(base_config.get("vocab_size"), int):
            vocab = base_config["vocab_size"]
        base_tensors = _base_tensors(base)
    except SourceError as exc:
        report.add("L040", WARNING, f"could not read the base model: {exc}")
        return vocab
    if not base_tensors:
        report.add(
            "L040",
            WARNING,
            f"{base.label} has no safetensors weights in its root; module paths and shapes were not compared",
        )
        return vocab
    if not modules:
        return vocab
    weights = {
        key[: -len(".weight")]: info
        for key, info in base_tensors.items()
        if key.endswith(".weight")
    }
    by_tail: dict[str, list[str]] = {}
    for path in weights:
        parts = path.split(".")
        for i in range(len(parts)):
            by_tail.setdefault(".".join(parts[i:]), []).append(path)

    missing: list[str] = []
    adapter_extra: dict[str, int] = {}
    base_extra: dict[str, int] = {}
    shape_bad: list[str] = []
    for name, module in modules.items():
        resolved: str | None = None
        if name in weights:
            resolved = name
        else:
            # Base path is the adapter path plus a leading prefix.
            longer = by_tail.get(name, [])
            if len(longer) == 1:
                resolved = longer[0]
                prefix = resolved[: -len(name)]
                base_extra[prefix] = base_extra.get(prefix, 0) + 1
            else:
                # Adapter path is the base path plus a leading prefix.
                parts = name.split(".")
                for i in range(1, len(parts)):
                    tail = ".".join(parts[i:])
                    if tail in weights:
                        resolved = tail
                        prefix = ".".join(parts[:i]) + "."
                        adapter_extra[prefix] = adapter_extra.get(prefix, 0) + 1
                        break
        if resolved is None:
            missing.append(name)
            continue
        shape = weights[resolved].shape
        if (
            module.a is not None
            and module.b is not None
            and len(shape) == 2
            and not module.embedding
        ):
            dims = sorted((module.a.shape[-1], module.b.shape[0]))
            if dims != sorted(shape):
                shape_bad.append(
                    f"{name}: LoRA maps {module.a.shape[-1]} -> {module.b.shape[0]}, base weight is {list(shape)}"
                )
    total = len(modules)
    if missing:
        report.add(
            "L040",
            ERROR,
            f"{len(missing)}/{total} adapter modules have no counterpart in the base checkpoint, e.g. "
            f"{_fmt_list(sorted(missing), 3)}. An engine would have nothing to attach them to.",
        )
    if base_extra:
        prefixes = ", ".join(f"'{p}' ({n} modules)" for p, n in sorted(base_extra.items()))
        report.add(
            "L043",
            WARNING,
            f"the base checkpoint nests the adapted modules under an extra prefix: {prefixes}. The adapter "
            "was probably trained against a different model class than the checkpoint layout (for example "
            "a text-only class of a multimodal model). This is the situation in which vLLM was reported to "
            "load an adapter without error and never apply it (vllm#34186): check it with `lora-check verify`.",
        )
    if adapter_extra:
        prefixes = ", ".join(f"'{p}' ({n} modules)" for p, n in sorted(adapter_extra.items()))
        report.add(
            "L043",
            INFO,
            f"adapter module paths carry a prefix the checkpoint keys lack: {prefixes}. This is normal when "
            "the checkpoint was saved from the bare backbone and the adapter was trained on a head model.",
        )
    if shape_bad:
        report.add(
            "L041", ERROR, f"shape mismatch with the base model: {_fmt_list(sorted(shape_bad), 3)}"
        )
    if not (missing or base_extra or shape_bad):
        report.summary["base_modules_matched"] = total
    return vocab


def _lint_tokenizer(
    report: LintReport,
    adapter: Source,
    files: list[str],
    config: Mapping[str, Any],
    modules: Mapping[str, _Module],
    others: list[str],
    tensors: Mapping[str, TensorInfo],
    base_vocab: int | None,
) -> None:
    sglang = "sglang" in report.engines
    notes: list[str] = []
    added_new: int | None = None
    if "added_tokens.json" in files:
        try:
            added = adapter.read_json("added_tokens.json")
        except SourceError:
            added = None
        if isinstance(added, dict) and added:
            if base_vocab is not None:
                added_new = sum(1 for v in added.values() if isinstance(v, int) and v >= base_vocab)
                notes.append(
                    f"added_tokens.json lists {len(added)} token(s), {added_new} beyond the base vocabulary "
                    f"of {base_vocab}"
                )
            else:
                notes.append(
                    f"added_tokens.json lists {len(added)} token(s) (pass --base to see which are new)"
                )
    tokenizer_files = [
        f
        for f in files
        if f in ("tokenizer.json", "tokenizer_config.json", "special_tokens_map.json")
    ]
    if tokenizer_files:
        notes.append(f"adapter ships its own tokenizer files ({', '.join(tokenizer_files)})")
    embed_saved = [
        m for m in (config.get("modules_to_save") or []) if any(h in str(m) for h in _EMBED_HINTS)
    ]
    if embed_saved:
        notes.append(f"modules_to_save includes embedding/output layers {embed_saved}")
    if config.get("trainable_token_indices"):
        notes.append("trainable_token_indices is set (selected token embeddings were trained)")
    full = [
        key
        for key in others
        if any(h in key for h in _EMBED_HINTS)
        and len(tensors.get(PEFT_PREFIX + key, tensors.get(key, TensorInfo("", ()))).shape) == 2
    ]
    for key in full[:2]:
        shape = tensors.get(PEFT_PREFIX + key, tensors.get(key, TensorInfo("", ()))).shape
        resized = (
            f", resized from {base_vocab}"
            if base_vocab is not None and shape[0] != base_vocab
            else ""
        )
        notes.append(f"weights contain a full matrix {key} with {shape[0]} rows{resized}")
    embedding_lora = sorted(n for n, m in modules.items() if m.embedding)
    if embedding_lora:
        notes.append(f"LoRA applied to embedding layers: {_fmt_list(embedding_lora, 2)}")
    if notes:
        report.add(
            "L050",
            WARNING,
            "; ".join(notes)
            + ". The server must use the adapter's tokenizer and embeddings; a server that "
            "keeps the base tokenizer will mis-tokenise new tokens.",
        )
    if sglang:
        if added_new or (
            added_new is None and any(n.startswith("added_tokens.json") for n in notes)
        ):
            report.add(
                "S202",
                ERROR if added_new else WARNING,
                "SGLang refuses adapters whose added_tokens.json adds ids beyond the base vocabulary "
                '("LoRA serving currently doesn\'t support adapters that add tokens to the vocabulary")'
                + ("" if added_new else "; pass --base to check whether these tokens are new"),
            )
        if embedding_lora or embed_saved:
            report.add(
                "S204",
                WARNING,
                "the SGLang LoRA documentation lists embedding-layer support as still under development",
            )
